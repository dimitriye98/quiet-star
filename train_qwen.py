import os
import time
import warnings
from collections import defaultdict
from contextlib import nullcontext
from functools import partial
from typing import Callable

import einx
import lightning.pytorch as pl
import numpy as np
import torch as t
from dataclasses import dataclass, field
from datasets import load_dataset, Dataset
from lightning.fabric.utilities import move_data_to_device
from peft import get_peft_model, LoraConfig
from pytorch_lightning.loggers import WandbLogger
from toolz.curried import compose, pipe, map
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, GenerationConfig, StaticCache, \
	CompileConfig, PreTrainedTokenizer, PreTrainedTokenizerBase
from transformers.cache_utils import StaticCacheConfig

from quiet_star.qwen import QwenThoughtModelConfig, QwenThoughtModel

# warnings.filterwarnings( "error", category = UserWarning, module = 'torch' )


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
	extra_kwargs: dict = field(default_factory=dict)

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

	return map_keys(compose(qa_dict, to_qa), "question", "answer")(b)

def do_tok_gsm8k( tokenizer, b ):
	map_each = lambda f, *keys: lambda d: d | { k: f( d[ k ] ) for k in keys }
	tokenize = lambda s: [ tk for tk in tokenizer(
		s,
		return_tensors = "np",
		padding = False,
		truncation = False,
		add_special_tokens = False )[ "input_ids" ] ]

	return map_each( tokenize, "question", "answer" )(b)

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
	return map_keys( compose( qa_dict, to_qa ), "question", "choices", "answerKey" )(b)

def do_tok_csqa( tokenizer, b ):
	map_each = lambda f, *keys: lambda d: d | { k: f( d[ k ] ) for k in keys }
	tokenize = lambda s: [ tk for tk in tokenizer(
		s,
		return_tensors = "np",
		padding = False,
		truncation = False,
		add_special_tokens = False )[ "input_ids" ] ]
	return map_each( tokenize, "question", "answer" )(b)

class QuietStarDataModule( pl.LightningDataModule ):
	def __init__(
			self,
			tokenizer,
			train_batch_size = 2,
			test_val_batch_size = 256,
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
		tokenized = TokenizeFn( tokenizer = self.tokenizer, fn = do_tok_owm, extra_kwargs = { "max_length": self.max_length } )

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
			.map( TokenizeFn( tokenizer = self.tokenizer, fn = do_tok_gsm8k ), batched = True, batch_size = self.preproc_batch_size )
			.rename_column( "question", "input_ids" )
			.with_format( type = "np", columns = [ "input_ids", "answer" ] ))

	def process_csqa( self, ds: Dataset ):

		return (
			ds
			.map( preproc_csqa, batched = False )
			.map( TokenizeFn( tokenizer = self.tokenizer, fn = do_tok_csqa ), batched = True, batch_size = self.preproc_batch_size )
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


class QuietStar( pl.LightningModule ):
	def __init__(
			self,
			config,
			model_factory,
			validation_pad_token,
			learning_rate = 1e-6,
			weight_decay = 0.001,
			adam_epsilon = 1e-6,
			warmup_steps = 0,
			train_batch_size = 2,
			test_val_batch_size = 8,
			look_ahead = 4,
			n_thoughts = 2,
			thought_depth = 12,
			**kwargs ):
		super().__init__()
		self.save_hyperparameters( ignore = [ "validation_pad_token" ] )
		self.validation_pad_token = validation_pad_token
		self.test_val_batch_size = test_val_batch_size
		self.outputs = defaultdict( list )

	def configure_model( self ):
		self.model = self.hparams.model_factory( self.hparams.config )
		self.model.logger = self.log_dict

	def configure_optimizers( self ):
		"""Prepare optimizer and schedule (linear warmup and decay)."""
		model = self.model
		no_decay = [ "bias", "LayerNorm.weight" ]
		optimizer_grouped_parameters = [
			{
				"params": [ p for n, p in model.named_parameters() if not any( nd in n for nd in no_decay ) ],
				"weight_decay": self.hparams.weight_decay,
			},
			{
				"params": [ p for n, p in model.named_parameters() if any( nd in n for nd in no_decay ) ],
				"weight_decay": 0.0,
			},
		]
		optimizer = AdamW(
			optimizer_grouped_parameters, lr = self.hparams.learning_rate, eps = self.hparams.adam_epsilon )

		scheduler = get_linear_schedule_with_warmup(
			optimizer,
			num_warmup_steps = self.hparams.warmup_steps,
			num_training_steps = self.trainer.estimated_stepping_batches,
		)
		scheduler = { "scheduler": scheduler, "interval": "step", "frequency": 1 }
		return [ optimizer ], [ scheduler ]

	def on_train_epoch_start( self ):
		self.training_cache = StaticCache(
			max_cache_len = 256 * ( 3 + self.hparams.look_ahead + self.hparams.thought_depth),
			batch_size = self.hparams.train_batch_size * self.hparams.n_thoughts,
			device = self.device,
			dtype = self.dtype,
			config = self.model.config )

	def training_step( self, batch, batch_idx ):
		loss = self.model(
			input_ids = batch[ "input_ids" ], labels = batch[ "input_ids" ],
			attention_mask = batch[ "attention_mask" ], past_key_values = self.training_cache ).loss
		self.training_cache = self.training_cache.reset()
		if hasattr(self, "profiler"):
			self.profiler.step()
		return loss

	@property
	def validation_gen_config( self ):
		if not hasattr( self, "_validation_gen_config" ):
			self._validation_gen_config = GenerationConfig(
				max_new_tokens = 10,
				output_logits = True,
				return_dict_in_generate = True,
				pad_token_id = self.validation_pad_token,
				num_logits_to_keep = 1,
				# cache_implementation = "static",
				# cache_config = StaticCacheConfig( batch_size = self.test_val_batch_size, max_cache_len = 1 ), # NB: max_cache_len has no effect due to bug in transformers
				# compile_config = CompileConfig(
				# 	fullgraph = False
				# )
			)
		return self._validation_gen_config

	def validation_step( self, batch, batch_idx, dataloader_idx = 0 ):
		output = self.model.generate(
			inputs = batch[ "input_ids" ],
			attention_mask = batch[ "attention_mask" ],
			generation_config = self.validation_gen_config )
		logs = t.cat( [ l.unsqueeze( 0 ) for l in output.logits ], dim = 0 )
		logs = einx.rearrange( "c ... v -> ... c v", logs )
		batch[ "output" ] = logs
		self.outputs[ dataloader_idx ].append( batch )
		if hasattr(self, "profiler"):
			self.profiler.step()

	def on_validation_epoch_end( self ):
		flat_outputs = [ ]
		for lst in self.outputs.values():
			flat_outputs.extend( lst )

		with (t.no_grad()):
			flat_outputs = move_data_to_device( flat_outputs, "cpu" )

			logs, ans = [ ], [ ]

			pad_to_length = 0
			for b in flat_outputs:
				a, l = b[ "answer" ], b[ "output" ]
				assert a.shape[ 0 ] == l.shape[ 0 ]
				a_len = a.shape[ -1 ]
				l_ = l[ ..., : a_len, : ]  # trim to answer length
				pad_to_length = max( pad_to_length, a_len )
				logs.append( l_ )
				ans.append( a )

			def pad_right( tensor, dim, pad_len, value ):
				pad_vals = [ (0, 0) ] * tensor.ndim
				# ugly index manipulation because torch.nn.functional.pad is weird
				pad_vals[ -1 - (dim % tensor.ndim) ] = (0, pad_len - tensor.shape[ dim ])
				pad_vals = tuple( p for tup in pad_vals for p in tup )
				return t.nn.functional.pad( tensor, pad_vals, value = value )

			apply_pad = lambda l, d, p, v: [ pad_right( tensor, d, p, v ) for tensor in l ]
			logs = apply_pad( logs, -2, pad_to_length, t.nan )
			ans = apply_pad( ans, -1, pad_to_length, self.validation_pad_token )
			logs, ans = t.cat( logs, dim = 0 ), t.cat( ans, dim = 0 )

			def compute_losses( l, a ):
				l_ = einx.rearrange( "... v -> (...) v", l )
				a_ = einx.rearrange( "... -> (...)", a )
				losses = t.nn.functional.cross_entropy( l_, a_, reduction = "none" ).reshape_as( a )
				losses = einx.where( "... s, , ... s", a == self.validation_pad_token, t.nan, losses )
				return losses

			losses = compute_losses( logs, ans )
			losses = losses.reshape_as( ans )

			loss = losses.nan_to_num( 1 ).prod( dim = - 1 ).mean()
			self.log( "val_loss", loss, prog_bar = True )


tokenizer = AutoTokenizer.from_pretrained( "Qwen/Qwen2.5-0.5B" )
tokenizer.padding_side = "left"
tokenizer.add_special_tokens( { "additional_special_tokens": [ "<|startthought|>", "<|endthought|>" ] } )

config = QwenThoughtModelConfig.from_pretrained(
	"Qwen/Qwen2.5-0.5B",
	start_thought_token = tokenizer.convert_tokens_to_ids( "<|startthought|>" ),
	end_thought_token = tokenizer.convert_tokens_to_ids( "<|endthought|>" ),
	initial_start_thought_token = tokenizer.convert_tokens_to_ids( "---" ),
	initial_end_thought_token = tokenizer.convert_tokens_to_ids( "---" ),
	host_thought_token_embeddings = True,
	torch_dtype = "bfloat16"
)


def model_factory( config ):
	peft_config = LoraConfig(
		r = 8,
		lora_alpha = 16,
		lora_dropout = 0.1,
		use_rslora = True,
		target_modules = "all-linear",
	)

	model = QwenThoughtModel.from_pretrained( "Qwen/Qwen2.5-0.5B", config = config )

	# Fix a bug with the underlying model
	model._lm_model.resize_token_embeddings( len( tokenizer ) )
	out_embeds = t.nn.Linear( model.config.hidden_size, model.config.vocab_size, bias = False )
	out_embeds.weight = model._lm_model.get_input_embeddings().weight
	model._lm_model.set_output_embeddings( out_embeds )

	print( "Tied LM head output weights to input weights" )

	model._lm_model = get_peft_model( model._lm_model, peft_config )

	return model


dm = QuietStarDataModule(
	tokenizer,
	n_download_proc = 1,
)

model = QuietStar( config, model_factory, validation_pad_token = tokenizer.pad_token_id, test_val_batch_size = dm.test_val_batch_size )

checkpoint_callback = pl.callbacks.ModelCheckpoint(
	every_n_train_steps = 10,
	save_top_k = -1,
)

run_id = int( time.time() )
trace_dir = os.path.join( ".trace", str( run_id ) )
os.makedirs( trace_dir, exist_ok = True )

def trace_handler( prof ):
	print("Saving trace...", flush = True)
	st = time.process_time()
	prof.export_chrome_trace( os.path.join( trace_dir, f"{prof.step_num}.json.gz" ) )
	et = time.process_time()
	print(f"Trace saved! Took {et - st:.2f} seconds", flush = True)

do_profile = False

t.cuda.memory._record_memory_history(stacks="python")

with t.profiler.profile(
		activities = [
			t.profiler.ProfilerActivity.CPU,
			# t.profiler.ProfilerActivity.CUDA,
		],
		schedule = t.profiler.schedule( skip_first = 0, wait = 0, warmup = 10, active = 1, repeat = 1 ),
		on_trace_ready = trace_handler,
		# profile_memory = True,
		with_stack = True,
		record_shapes = True,
		experimental_config = t._C._profiler._ExperimentalConfig( verbose = True )
) if do_profile else nullcontext() as prof:
	if do_profile:
		model.profiler = prof

	root_dir = "gs://ddanilovic-llm-storage"
	logger = WandbLogger(log_model="all")
	logger.watch( model )
	trainer = pl.Trainer(
		default_root_dir = root_dir,
		logger = logger,
		precision = "bf16-true",
		max_epochs = 1,
		check_val_every_n_epoch = None,
		val_check_interval = 100,
		# accelerator = "auto",
		# devices = 1,
		accumulate_grad_batches = 4,
		callbacks = [ checkpoint_callback ],
		# enable_progress_bar = False,
	)

	try:
		# trainer.fit( model, datamodule = dm )
		trainer.validate( model, datamodule = dm )
	finally:
		t.cuda.memory._dump_snapshot( os.path.join( trace_dir, "memory_snapshot.json" ) )
