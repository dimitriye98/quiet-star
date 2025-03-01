import os
import queue
import time
from collections import defaultdict
from contextlib import nullcontext
from multiprocessing import Process, Queue
from os import cpu_count
from threading import Thread
from typing import Callable

import einx
import lightning.pytorch as pl
import torch
import torch as t
from deepspeed.ops.adam import DeepSpeedCPUAdam
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning_utilities import apply_to_collection
from peft import get_peft_model, LoraConfig
from transformers import AutoTokenizer, GenerationConfig, StaticCache

from quiet_star.qwen import QwenThoughtModelConfig, QwenThoughtModel
from quiet_star.training.data import QuietStarDataModule


# warnings.filterwarnings( "error", category = UserWarning, module = 'torch' )


def be_nice():
	# os.nice( 10 )
	pass


class ToCPUWorker( Thread ):
	def __init__( self, q_in, q_out ):
		super().__init__()
		self.q_in = q_in
		self.q_out = q_out
		self.daemon = True

	@staticmethod
	def transfer_tensor( tensor ):
		print( "Transferring tensor", flush = True )
		stream = t.cuda.Stream()
		with t.cuda.stream( stream ):
			receive = t.empty( tensor.shape, dtype = tensor.dtype, device = "cpu", pin_memory = True )
			receive.copy_( tensor )
			# We have to copy again, as the receiving tensor *must* be pinned
			# for the transfer to not block the GPU
			# however, holding pinned memory can be expensive,
			# so we only keep the data in pinned memory if instructed
			# if not pin_memory:
			unpinned = t.empty( tensor.shape, dtype = tensor.dtype, device = "cpu" )
			unpinned.copy_( receive )
			print( "Transferred tensor", flush = True )
			return unpinned

	@classmethod
	def do_transfer( cls, data ):
		return apply_to_collection( data, t.Tensor, cls.transfer_tensor )

	def run( self ):
		while True:
			data = self.q_in.get()
			self.q_out.put( self.do_transfer( data ) )


class EvaluationWorker( Process ):
	def __init__( self, q_in, q_out ):
		super().__init__()
		self.q_in = q_in
		self.q_out = q_out

	@staticmethod
	def evaluate( data ):
		return data

	def run( self ):
		while True:
			data = self.q_in.get()
			self.q_out.put( self.evaluate( data ) )


class Evaluator:
	def __init__(
			self, *,
			factory_transfer_worker: Callable[ [ Queue, Queue ], ToCPUWorker ] = ToCPUWorker,
			n_transfer_worker = None,
			factory_evaluator_worker: Callable[ [ Queue, Queue ], EvaluationWorker ] = EvaluationWorker,
			n_evaluator_worker = None ):
		if n_transfer_worker is None:
			n_transfer_worker = min(16, cpu_count())
		if n_evaluator_worker is None:
			n_evaluator_worker = min(16, cpu_count())

		self.counter = 0
		self.q_in = queue.Queue()
		self.q_mid = Queue()
		self.q_out = Queue()

		self.transfer_workers = [ factory_transfer_worker( self.q_in, self.q_mid ) for _ in range( n_transfer_worker ) ]
		self.evaluator_workers = [ factory_evaluator_worker( self.q_mid, self.q_out ) for _ in
			range( n_evaluator_worker ) ]

		for w in self.transfer_workers:
			w.daemon = True
			w.start()

		for w in self.evaluator_workers:
			w.daemon = True
			w.start()

	# Enqueues data for transfer to CPU and evaluation in a separate process
	def submit( self, data ):
		self.counter += 1
		self.q_in.put( data )

	# Blocks until all enqueued data has been evaluated
	def collect( self ):
		ret = [ ]
		while self.counter > 0:
			ret.append( self.q_out.get() )
			self.counter -= 1
		return ret


class QuietStarEvaluationWorker( EvaluationWorker ):
	def __init__( self, q_in, q_out, pad_token ):
		super().__init__( q_in, q_out )
		self.pad_token = pad_token

	def evaluate( self, batch ):
		print( "Doing evaluation", flush = True )
		print( f"Batch: {batch}", flush = True )
		evalstarttime = time.time()

		stat = "loss nc"

		@torch.no_grad()
		def compute_losses( l, a ):
			l_ = einx.rearrange( "... v -> (...) v", l )
			print( f"{stat} l_ transform {time.time() - evalstarttime}", flush = True )
			a_ = einx.rearrange( "... -> (...)", a )
			print( f"{stat} a_ transform {time.time() - evalstarttime}", flush = True )
			losses = t.nn.functional.cross_entropy( l_, a_, reduction = "none" )
			print( f"{stat} losses {time.time() - evalstarttime}", flush = True )
			losses = einx.rearrange( "(b... s) -> b... s", losses, **einx.solve( "b... s", a ) )
			print( f"{stat} losses rearrange {time.time() - evalstarttime}", flush = True )
			losses = einx.where( "... s, , ... s", a == self.pad_token, t.nan, losses )
			print( f"{stat} losses where {time.time() - evalstarttime}", flush = True )
			ret = losses.nanmean( dim = -1 )
			print( f"{stat} losses nanmean {time.time() - evalstarttime}", flush = True )
			return ret

		def check_accuracy( s, a ):
			return ((s == a) | (a == self.pad_token)).all( dim = -1 ).to( t.float32 )

		lossnc = compute_losses( batch[ "output_logits" ], batch[ "answer" ] )
		stat = "loss wc"
		losswc = compute_losses( batch[ "confident_output_logits" ], batch[ "answer" ] )

		ret = {
			"loss (no confidence)": lossnc,
			"one-shot accuracy (no confidence)": check_accuracy( batch[ "output_ids" ], batch[ "answer" ] ),
			"loss (with confidence)": losswc,
			"one-shot accuracy (with confidence)": check_accuracy( batch[ "confident_output_ids" ], batch[ "answer" ] ),
		}

		print( f"Evaluation took {time.time() - evalstarttime} seconds", flush = True )

		return ret


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
			look_ahead = 4,
			n_thoughts = 2,
			thought_depth = 12,
			confidence_loss_beta_step_up = 1e3,
			confidence_loss_beta_step_up_start = 2000,
			confidence_loss_beta_step_up_end = 3000,
			**kwargs ):
		super().__init__()
		self.save_hyperparameters()
		self.validation_pad_token = validation_pad_token
		self.outputs = defaultdict( list )
		self.initial_confidence_loss_beta = config.confidence_loss_beta

	def configure_model( self ):
		self.model = self.hparams.model_factory( self.hparams.config )
		self.model.logger = self.log_dict
		self.evaluator = Evaluator(
			factory_evaluator_worker = lambda q_in, q_out: QuietStarEvaluationWorker(
				q_in, q_out, self.hparams.validation_pad_token ) )

	def configure_optimizers( self ):
		"""Prepare optimizer and schedule (linear warmup and decay)."""
		# model = self.model
		# no_decay = [ "bias", "LayerNorm.weight" ]
		# optimizer_grouped_parameters = [
		# 	{
		# 		"params": [ p for n, p in model.named_parameters() if not any( nd in n for nd in no_decay ) ],
		# 		"weight_decay": self.hparams.weight_decay,
		# 	},
		# 	{
		# 		"params": [ p for n, p in model.named_parameters() if any( nd in n for nd in no_decay ) ],
		# 		"weight_decay": 0.0,
		# 	},
		# ]
		# optimizer = AdamW(
		# 	optimizer_grouped_parameters, lr = self.hparams.learning_rate, eps = self.hparams.adam_epsilon )
		#
		# scheduler = get_linear_schedule_with_warmup(
		# 	optimizer,
		# 	num_warmup_steps = self.hparams.warmup_steps,
		# 	num_training_steps = self.trainer.estimated_stepping_batches,
		# )
		# scheduler = { "scheduler": scheduler, "interval": "step", "frequency": 1 }
		# return [ optimizer ], [ scheduler ]
		return DeepSpeedCPUAdam(
			self.model.parameters(),
			lr = self.hparams.learning_rate,
			eps = self.hparams.adam_epsilon,
			weight_decay = self.hparams.weight_decay
		)

	def on_train_epoch_start( self ):
		t.cuda.memory._record_memory_history( stacks = "python" )

	def training_step( self, batch, batch_idx ):
		cache_len, batch_size = self.model.get_cache_size( *batch[ "input_ids" ].shape )
		training_cache = StaticCache(
			max_cache_len = cache_len,
			max_batch_size = batch_size,
			device = self.device,
			dtype = self.dtype,
			config = self.model.config )
		if self.hparams.confidence_loss_beta_step_up_start < self.current_epoch <= self.hparams.confidence_loss_beta_step_up_end:
			# Linear ramp from start to end
			now = self.current_epoch
			start = self.hparams.confidence_loss_beta_step_up_start
			end = self.hparams.confidence_loss_beta_step_up_end
			lerp = (now - start) / (end - start)
			lower = self.initial_confidence_loss_beta
			upper = self.hparams.confidence_loss_beta_step_up * lower
			self.model.confidence_loss_beta = lower * (1 - lerp) + upper * lerp
		loss = self.model(
			input_ids = batch[ "input_ids" ], labels = batch[ "input_ids" ],
			attention_mask = batch[ "attention_mask" ], past_key_values = training_cache ).loss
		if hasattr( self, "profiler" ):
			self.profiler.step()
		return loss

	@property
	def validation_gen_config( self ):
		if not hasattr( self, "_validation_gen_config" ):
			self._validation_gen_config = GenerationConfig(
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

	# def on_validation_epoch_start(self):
	# 	self.evaluation_pool = ProcessPoolExecutor( max_workers = cpu_count() - 1, initializer = be_nice )

	def validation_step( self, batch, batch_idx, dataloader_idx = 0 ):
		output_sans_confidence = self.model.generate(
			inputs = batch[ "input_ids" ],
			attention_mask = batch[ "attention_mask" ],
			generation_config = self.validation_gen_config,
			max_new_tokens = batch[ "answer" ].shape[ -1 ] )

		logs_sc = t.cat( [ l.unsqueeze( 0 ) for l in output_sans_confidence.logits ], dim = 0 )
		logs_sc = einx.rearrange( "c ... v -> ... c v", logs_sc )
		output_ids_sc = output_sans_confidence.sequences[ ..., -batch[ "answer" ].shape[ -1 ]: ]
		del output_sans_confidence

		output_with_confidence = self.model.generate(
			inputs = batch[ "input_ids" ],
			attention_mask = batch[ "attention_mask" ],
			generation_config = self.validation_gen_config,
			max_new_tokens = batch[ "answer" ].shape[ -1 ],
			confidence_parameter = 0.7, comparison_mode = True )

		logs_wc = t.cat( [ l.unsqueeze( 0 ) for l in output_with_confidence.logits ], dim = 0 )
		logs_wc = einx.rearrange( "c ... v -> ... c v", logs_wc )
		output_ids_wc = output_with_confidence.sequences[ ..., -batch[ "answer" ].shape[ -1 ]: ]

		print("Submitting to evaluator", flush = True)

		self.evaluator.submit(
			batch | {
				"output_ids": output_ids_sc,
				"output_logits": logs_sc,
				"confident_output_ids": output_ids_wc,
				"confident_output_logits": logs_wc } )

		if hasattr( self, "profiler" ):
			self.profiler.step()

	def on_validation_epoch_end( self ):
		print( "Awaiting validation results...", flush = True )
		flat_outputs = self.evaluator.collect()

		flat_dict = defaultdict( list )
		for d in flat_outputs:
			for k, v in d.items():
				flat_dict[ k ].append( v )

		means = { f"validation mean {k}": t.cat( v, dim = 0 ).mean() for k, v in flat_dict.items() }

		self.log_dict( means, prog_bar = True )


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
		r = 32,
		lora_alpha = 64,
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

	print(
		f"Model has {model.num_parameters()} parameters, of which "
		f"{model.num_parameters( only_trainable = True )} are trainable" )

	return model


effective_batch_size = 8
acc_grad = 8
assert effective_batch_size % acc_grad == 0

dm = QuietStarDataModule(
	tokenizer,
	n_download_proc = 16,
	train_max_length = 128,
	test_val_max_length = 256,
	train_batch_size = effective_batch_size // acc_grad,
)

model = QuietStar(
	config,
	model_factory,
	validation_pad_token = tokenizer.pad_token_id,
	train_batch_size = dm.train_batch_size,
	test_val_batch_size = dm.test_val_batch_size )

checkpoint_callback = pl.callbacks.ModelCheckpoint(
	every_n_train_steps = 50,
	save_top_k = -1,
)

run_id = int( time.time() )
trace_dir = os.path.join( ".trace", str( run_id ) )
os.makedirs( trace_dir, exist_ok = True )


def trace_handler( prof ):
	print( "Saving trace...", flush = True )
	st = time.process_time()
	prof.export_chrome_trace( os.path.join( trace_dir, f"{prof.step_num}.json.gz" ) )
	et = time.process_time()
	print( f"Trace saved! Took {et - st:.2f} seconds", flush = True )


do_profile = False

# t.cuda.memory._record_memory_history( stacks = "python" )

with t.profiler.profile(
		activities = [
			t.profiler.ProfilerActivity.CPU,
			t.profiler.ProfilerActivity.CUDA,
		],
		schedule = t.profiler.schedule( skip_first = 0, wait = 0, warmup = 4, active = 3, repeat = 1 ),
		on_trace_ready = trace_handler,
		profile_memory = True,
		with_stack = True,
		record_shapes = True,
		experimental_config = t._C._profiler._ExperimentalConfig( verbose = True )
) if do_profile else nullcontext() as prof:
	if do_profile:
		model.profiler = prof

	root_dir = "gs://ddanilovic-llm-storage"
	logger = WandbLogger( log_model = "all" )
	logger.watch( model )
	trainer = pl.Trainer(
		default_root_dir = root_dir,
		logger = logger,
		precision = "bf16-true",
		max_epochs = 1,
		# max_steps = 2,
		check_val_every_n_epoch = None,
		val_check_interval = 250,
		accelerator = "gpu",
		devices = 1,
		accumulate_grad_batches = acc_grad,
		# accumulate_grad_batches = 1,
		strategy = DeepSpeedStrategy(
			stage = 2,
			offload_optimizer = True,
			offload_parameters = True,
			remote_device = "nvme",
			offload_optimizer_device = "nvme",
			nvme_path = "/gradients"
		),
		callbacks = [ checkpoint_callback ],
		# enable_progress_bar = False,
	)

	try:
		trainer.fit( model, datamodule = dm )
	# trainer.validate( model, datamodule = dm )
	except t.OutOfMemoryError as e:
		print( "OOM...", flush = True )
	finally:
		t.cuda.memory._dump_snapshot( os.path.join( trace_dir, "memory_snapshot.json" ) )
		prof.step()
		if prof is not None:
			prof.export_memory_timeline( os.path.join( trace_dir, "memory_timeline.html" ) )
