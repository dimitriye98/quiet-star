import os
import time
from collections import defaultdict
from concurrent.futures.process import ProcessPoolExecutor
from contextlib import nullcontext
from os import cpu_count

import einx
import lightning.pytorch as pl
import torch as t
from peft import get_peft_model, LoraConfig
from pytorch_lightning.loggers import WandbLogger
from toolz import compose_left
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, GenerationConfig, StaticCache

from quiet_star.broker import CollectionBroker
from quiet_star.futures.collected import CollectedFuture
from quiet_star.qwen import QwenThoughtModelConfig, QwenThoughtModel
from quiet_star.training.data import QuietStarDataModule, uncurry


# warnings.filterwarnings( "error", category = UserWarning, module = 'torch' )


def be_nice():
	os.nice( 10 )


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
		self.save_hyperparameters( ignore = [ "validation_pad_token" ] )
		self.validation_pad_token = validation_pad_token
		self.broker = CollectionBroker()
		self.evaluation_pool = ProcessPoolExecutor( max_workers = cpu_count() - 1, initializer = be_nice )
		self.outputs = defaultdict( list )
		self.initial_confidence_loss_beta = config.confidence_loss_beta

	def __del__( self ):
		self.evaluation_pool.shutdown( wait = False, cancel_futures = True )

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
			max_cache_len = 256 * (3 + self.hparams.look_ahead + self.hparams.thought_depth),
			batch_size = self.hparams.train_batch_size * self.hparams.n_thoughts,
			device = self.device,
			dtype = self.dtype,
			config = self.model.config )

	def training_step( self, batch, batch_idx ):
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
			attention_mask = batch[ "attention_mask" ], past_key_values = self.training_cache ).loss
		self.training_cache = self.training_cache.reset()
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

	def validation_step( self, batch, batch_idx, dataloader_idx = 0 ):
		output_sans_confidence = self.model.generate(
			inputs = batch[ "input_ids" ],
			attention_mask = batch[ "attention_mask" ],
			generation_config = self.validation_gen_config,
			max_new_tokens = batch[ "answer" ].shape[ -1 ] )

		logs_sc = t.cat( [ l.unsqueeze( 0 ) for l in output_sans_confidence.logits ], dim = 0 )
		logs_sc = einx.rearrange( "c ... v -> ... c v", logs_sc )
		future_output_ids_sc = self.broker( output_sans_confidence.sequences[ ..., -batch[ "answer" ].shape[ -1 ]: ] )
		future_logs_sc = self.broker( logs_sc )
		del output_sans_confidence, logs_sc

		output_with_confidence = self.model.generate(
			inputs = batch[ "input_ids" ],
			attention_mask = batch[ "attention_mask" ],
			generation_config = self.validation_gen_config,
			max_new_tokens = batch[ "answer" ].shape[ -1 ],
			confidence_parameter = 0.7, comparison_mode = True )

		logs_wc = t.cat( [ l.unsqueeze( 0 ) for l in output_with_confidence.logits ], dim = 0 )
		logs_wc = einx.rearrange( "c ... v -> ... c v", logs_wc )
		future_output_ids_wc = self.broker( output_with_confidence.sequences[ ..., -batch[ "answer" ].shape[ -1 ]: ] )
		future_logs_wc = self.broker( logs_wc )
		future_batch = self.broker( batch )

		tupled_future = CollectedFuture(
			(
				future_batch,
				future_output_ids_sc,
				future_logs_sc,
				future_output_ids_wc,
				future_logs_wc) )

		tupled_future.add_done_callback(
			compose_left(
				lambda f: f.result(),
				uncurry(
					lambda b, oidsc, lsc, oidwc, lwc:
					b | {
						"output_ids": oidsc,
						"output_logits": lsc,
						"confident_output_ids": oidwc,
						"confident_output_logits": lwc } ),
				lambda b: self.evaluation_pool.submit( self.evaluate, b, self.validation_pad_token ),
				self.outputs[ dataloader_idx ].append ) )

		if hasattr( self, "profiler" ):
			self.profiler.step()

	@staticmethod
	def evaluate( batch, pad_token ):
		def compute_losses( l, a ):
			l_ = einx.rearrange( "... v -> (...) v", l )
			a_ = einx.rearrange( "... -> (...)", a )
			losses = t.nn.functional.cross_entropy( l_, a_, reduction = "none" )
			losses = einx.rearrange( "(b... s) -> b... s", losses, **einx.solve( "b... s", a ) )
			losses = einx.where( "... s, , ... s", a == pad_token, t.nan, losses )
			return losses.nanmean( dim = -1 )

		def check_accuracy( s, a ):
			return ((s == a) | (a == pad_token)).all( dim = -1 ).to( t.float32 )

		return {
			"loss (no confidence)": compute_losses( batch[ "output_logits" ], batch[ "answer" ] ),
			"one-shot accuracy (no confidence)": check_accuracy( batch[ "output_ids" ], batch[ "answer" ] ),
			"loss (with confidence)": compute_losses( batch[ "confident_output_logits" ], batch[ "answer" ] ),
			"one-shot accuracy (with confidence)": check_accuracy( batch[ "confident_output_ids" ], batch[ "answer" ] ),
		}

	def on_validation_epoch_end( self ):
		flat_outputs = [ f.result() for l in self.outputs.values() for f in l ]

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

t.cuda.memory._record_memory_history( stacks = "all" )

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
	logger = WandbLogger( log_model = "all" )
	logger.watch( model )
	trainer = pl.Trainer(
		default_root_dir = root_dir,
		logger = logger,
		precision = "bf16-true",
		max_epochs = 1,
		check_val_every_n_epoch = None,
		val_check_interval = 250,
		# accelerator = "auto",
		# devices = 1,
		accumulate_grad_batches = acc_grad,
		callbacks = [ checkpoint_callback ],
		# enable_progress_bar = False,
	)

	try:
		trainer.fit( model, datamodule = dm )
	# trainer.validate( model, datamodule = dm )
	except t.OutOfMemoryError as e:
		print( "OOM, dumping memory snapshot...", flush = True )
		t.cuda.memory._dump_snapshot( os.path.join( trace_dir, "memory_snapshot.json" ) )
