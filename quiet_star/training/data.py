import os
from collections import defaultdict
from functools import partial
from typing import Callable

import numpy as np
import torch as t
from dataclasses import dataclass, field
from datasets import Dataset, load_dataset
from lightning import pytorch as pl
from toolz import pipe, compose
from toolz.curried import map
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase


@dataclass
class CollateQA:
	pad_token_id: int

	def __call__( self, b ):
		def pad( tensor, side, dim, pad_len, value ):
			pad_vals = [ (0, 0) ] * tensor.ndim

			if side == "left":
				pad_vals[ dim ] = (pad_len - tensor.shape[ dim ], 0)
			else:
				pad_vals[ dim ] = (0, pad_len - tensor.shape[ dim ])

			return np.pad( tensor, pad_vals, constant_values = value )

		def union_list_of_dicts( l ):
			dd = defaultdict( list )

			for d in l:
				for k, v in d.items():
					dd[ k ].append( v )

			return dict( dd )

		union = union_list_of_dicts( b )
		inp_id_pad_len = max( s.shape[ -1 ] for s in union[ "input_ids" ] )
		ans_pad_len = max( s.shape[ -1 ] for s in union[ "answer" ] )
		union = union | { "attention_mask": [ np.ones_like( s ) for s in union[ "input_ids" ] ] }

		pad_args = {
			"input_ids": partial(
				pad,
				side = "left",
				dim = -1,
				pad_len = inp_id_pad_len,
				value = self.pad_token_id ),
			"attention_mask": partial(
				pad,
				side = "left",
				dim = -1,
				pad_len = inp_id_pad_len,
				value = 0 ),
			"answer": partial(
				pad,
				side = "right",
				dim = -1,
				pad_len = ans_pad_len,
				value = self.pad_token_id )
		}

		ret = { k: pipe( v, map( pad_args[ k ] ), tuple, np.stack, t.from_numpy ) for k, v in union.items() }

		return ret


@dataclass
class TokenizeFn:
	tokenizer: PreTrainedTokenizerBase
	fn: Callable
	extra_args: tuple = ()
	extra_kwargs: dict = field( default_factory = dict )

	def __call__( self, *args, **kwargs ):
		return self.fn( self.tokenizer, *args, *self.extra_args, **kwargs, **self.extra_kwargs )


def do_tok_owm( tokenizer, b, max_length ):
	tokens = tokenizer(
		b[ "text" ], padding = "max_length", max_length = max_length, truncation = True,
		padding_side = "left", return_tensors = "pt", return_attention_mask = True, add_special_tokens = False )
	return { **tokens }


def preproc_gsm8k( b ):
	to_qa = lambda q, a: (f"Q: {q}\nA: ", a.split( "####" )[ -1 ] + "\n")
	qa_dict = lambda t: { "question": t[ 0 ], "answer": t[ 1 ] }
	uncurry = lambda f: lambda args: f( *args )
	map_keys = lambda f, *keys: lambda d: d | uncurry( f )( d[ k ] for k in keys )

	return map_keys( compose( qa_dict, to_qa ), "question", "answer" )( b )


def do_tok_gsm8k( tokenizer, b ):
	map_each = lambda f, *keys: lambda d: d | { k: f( d[ k ] ) for k in keys }
	tokenize = lambda s: [ tk for tk in tokenizer(
		s,
		return_tensors = "np",
		padding = False,
		truncation = False,
		add_special_tokens = False )[ "input_ids" ] ]

	return map_each( tokenize, "question", "answer" )( b )


def preproc_csqa( b ):
	select = lambda d, *keys: zip( *(d[ k ] for k in keys) )
	to_qa = lambda q, C, a: (
		"\n".join(
			[
				"Q: " + q,
				*(f"({l}) {c}" for l, c in select( C, "label", "text" )),
				"A: " ] ),
		a + "\n")
	qa_dict = lambda t: { "question": t[ 0 ], "answer": t[ 1 ] }
	uncurry = lambda f: lambda args: f( *args )
	map_keys = lambda f, *keys: lambda d: d | uncurry( f )( d[ k ] for k in keys )
	return map_keys( compose( qa_dict, to_qa ), "question", "choices", "answerKey" )( b )


def do_tok_csqa( tokenizer, b ):
	map_each = lambda f, *keys: lambda d: d | { k: f( d[ k ] ) for k in keys }
	tokenize = lambda s: [ tk for tk in tokenizer(
		s,
		return_tensors = "np",
		padding = False,
		truncation = False,
		add_special_tokens = False )[ "input_ids" ] ]
	return map_each( tokenize, "question", "answer" )( b )


class QuietStarDataModule( pl.LightningDataModule ):
	def __init__(
			self,
			tokenizer,
			train_batch_size = 2,
			test_val_batch_size = 1024,
			preproc_batch_size = 1024,
			max_length = 256,
			n_download_proc = 1,
			dataloader_workers = min( 16, os.cpu_count() ) ):
		super().__init__()
		self.tokenizer = tokenizer
		self.n_download_proc = n_download_proc
		self.max_length = max_length
		self.train_batch_size = train_batch_size
		self.test_val_batch_size = test_val_batch_size
		self.preproc_batch_size = preproc_batch_size
		self.dataloader_workers = dataloader_workers

	def process_owm( self, ds: Dataset ):
		# def tokenized( b ):
		# 	tokens = self.tokenizer(
		# 		b[ "text" ], padding = "max_length", max_length = self.max_length, truncation = True,
		# 		padding_side = "left", return_tensors = "pt", return_attention_mask = True, add_special_tokens = False )
		# 	return { **tokens }
		tokenized = TokenizeFn(
			tokenizer = self.tokenizer, fn = do_tok_owm, extra_kwargs = { "max_length": self.max_length } )

		ret = ds.map( tokenized, batched = True, batch_size = self.preproc_batch_size ).with_format( type = "pt" )

		return ret

	def process_gsm8k( self, ds: Dataset ):
		# to_qa = lambda q, a: (f"Q: {q}\nA: ", a.split( "####" )[ -1 ] + "\n")
		# qa_dict = lambda t: { "question": t[ 0 ], "answer": t[ 1 ] }
		# uncurry = lambda f: lambda args: f( *args )
		# map_keys = lambda f, *keys: lambda d: d | uncurry( f )( d[ k ] for k in keys )
		# map_each = lambda f, *keys: lambda d: d | { k: f( d[ k ] ) for k in keys }

		return (
			ds
			.map( preproc_gsm8k, batched = False )
			.map(
				TokenizeFn( tokenizer = self.tokenizer, fn = do_tok_gsm8k ), batched = True,
				batch_size = self.preproc_batch_size )
			.rename_column( "question", "input_ids" )
			.with_format( type = "np", columns = [ "input_ids", "answer" ] ))

	def process_csqa( self, ds: Dataset ):
		return (
			ds
			.map( preproc_csqa, batched = False )
			.map(
				TokenizeFn( tokenizer = self.tokenizer, fn = do_tok_csqa ), batched = True,
				batch_size = self.preproc_batch_size )
			.rename_column( "question", "input_ids" )
			.with_format( type = "np", columns = [ "input_ids", "answer" ] ))

	def get_train_dataset( self ):
		return self.process_owm(
			load_dataset(
				"open-web-math/open-web-math",
				split = "train[:1000]",
				num_proc = self.n_download_proc ) )

	def get_validation_datasets( self ):
		csqa = load_dataset( "tau/commonsense_qa", num_proc = self.n_download_proc )
		csqa.pop( "test" )  # Test set has no answers
		return (
			self.process_gsm8k( load_dataset( "gsm8k", name = "main", num_proc = self.n_download_proc ) ),
			self.process_csqa( csqa ),
		)

	def prepare_data( self ):
		self.train_dataset = self.get_train_dataset()
		rest = self.get_validation_datasets()
		self.val_datasets = (rest[ 0 ][ "train" ], rest[ 1 ][ "train" ])
		self.test_datasets = (rest[ 0 ][ "test" ], rest[ 1 ][ "validation" ])

	def train_dataloader( self ):
		return DataLoader(
			self.train_dataset,
			batch_size = self.train_batch_size,
			# collate_fn = collate_fn,
			num_workers = self.dataloader_workers,
			pin_memory = True,
			shuffle = False )

	def val_dataloader( self ):
		return tuple(
			DataLoader(
				ds,
				batch_size = self.test_val_batch_size,
				collate_fn = CollateQA( self.tokenizer.pad_token_id ),
				num_workers = self.dataloader_workers,
				pin_memory = True,
				shuffle = False )
				for ds in self.val_datasets )
