import os
from collections.abc import Callable

from datasets import Dataset, load_dataset
from lightning import pytorch as pl
from toolz import compose
from torch.utils.data import DataLoader


class Uncurry( Callable ):
	def __init__( self, f ):
		self.f = f

	def __call__( self, args ):
		return self.f( *args )


uncurry = Uncurry


def do_tok_owm( b, tokenizer, max_length ):
	tokens = tokenizer(
		b[ "text" ], padding = "max_length", max_length = max_length, truncation = True,
		padding_side = "left", return_tensors = "pt", return_attention_mask = True, add_special_tokens = False )
	return { **tokens }


def preproc_gsm8k( b ):
	to_qa = lambda q, a: (f"Q: {q}\nA: ", a.split( "####" )[ -1 ] + "\n")
	qa_dict = lambda t: { "question": t[ 0 ], "answer": t[ 1 ] }
	map_keys = lambda f, *keys: lambda d: d | uncurry( f )( d[ k ] for k in keys )

	return map_keys( compose( qa_dict, to_qa ), "question", "answer" )( b )


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
	map_keys = lambda f, *keys: lambda d: d | uncurry( f )( d[ k ] for k in keys )
	return map_keys( compose( qa_dict, to_qa ), "question", "choices", "answerKey" )( b )


def do_tok_eval( b, tokenizer, max_length, ans_len ):
	q, a = b[ "question" ], b[ "answer" ]
	q_out = tokenizer(
		q,
		return_tensors = "pt",
		padding = "max_length",
		max_length = max_length,
		truncation = True,
		padding_side = "left",
		add_special_tokens = False )
	a_out = tokenizer(
		a,
		return_tensors = "pt",
		padding = "max_length",
		max_length = ans_len,
		truncation = True,
		padding_side = "right",
		add_special_tokens = False )
	return {
		"input_ids": q_out[ "input_ids" ],
		"attention_mask": q_out[ "attention_mask" ],
		"answer": a_out[ "input_ids" ],
	}


class QuietStarDataModule( pl.LightningDataModule ):
	def __init__(
			self,
			tokenizer,
			train_batch_size = 2,
			test_val_batch_size = 512,
			preproc_batch_size = 1024,
			gsm8k_ans_len = 15,
			csqa_ans_len = 5,
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
		self.gsm8k_ans_len = gsm8k_ans_len
		self.csqa_ans_len = csqa_ans_len

	def process_owm( self, ds: Dataset ):
		ret = ds.map(
			do_tok_owm, batched = True, batch_size = self.preproc_batch_size,
			fn_kwargs = { "max_length": self.max_length, "tokenizer": self.tokenizer } ).with_format( type = "pt" )

		return ret

	def process_eval( self, preproc, ans_len, ds: Dataset ):
		ret = (
			ds
			.map( preproc, batched = False )
			.map(
				do_tok_eval,
				fn_kwargs = {
					"max_length": self.max_length,
					"ans_len": ans_len,
					"tokenizer": self.tokenizer },
				batched = True,
				batch_size = self.preproc_batch_size,
				desc = "Tokenizing" )
			.with_format( type = "pt", columns = [ "input_ids", "attention_mask", "answer" ] ))
		return ret

	def get_train_dataset( self ):
		return self.process_owm(
			load_dataset(
				"open-web-math/open-web-math",
				split = "train[:8000]",
				num_proc = self.n_download_proc ) )

	def get_validation_datasets( self ):
		csqa = load_dataset( "tau/commonsense_qa", num_proc = self.n_download_proc )
		gsm8k = load_dataset( "gsm8k", name = "main", num_proc = self.n_download_proc )
		csqa.pop( "test" )  # Test set has no answers
		return {
			"csqa": self.process_eval( preproc_csqa, self.csqa_ans_len, csqa ),
			"gsm8k": self.process_eval( preproc_gsm8k, self.gsm8k_ans_len, gsm8k )
		}

	def prepare_data( self ):
		self.train_dataset = self.get_train_dataset()
		rest = self.get_validation_datasets()
		self.val_datasets = { "CommonsenseQA": rest[ "csqa" ][ "train" ], "GSM8k": rest[ "gsm8k" ][ "train" ] }
		self.test_datasets = { "CommonsenseQA": rest[ "csqa" ][ "validation" ], "GSM8k": rest[ "gsm8k" ][ "test" ] }

	def train_dataloader( self ):
		return DataLoader(
			self.train_dataset,
			batch_size = self.train_batch_size,
			num_workers = self.dataloader_workers,
			pin_memory = True,
			shuffle = False )

	def val_dataloader( self ):
		return { k:
			DataLoader(
				ds,
				batch_size = self.test_val_batch_size,
				num_workers = self.dataloader_workers,
				pin_memory = True,
				shuffle = False )
			for k, ds in self.val_datasets.items() }
