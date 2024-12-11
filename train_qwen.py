from collections import defaultdict

import einx
import lightning.pytorch as pl
import torch as t
from datasets import load_dataset, Dataset
from peft import BOFTConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer

from quiet_star.qwen import QwenThoughtModelConfig, QwenThoughtModel


class QuietStarDataModule( pl.LightningDataModule ):
	def __init__(
			self, tokenizer, train_batch_size = 2, test_val_batch_size = 8, max_length = 256, n_download_proc = 1 ):
		super().__init__()
		self.tokenizer = tokenizer
		self.n_download_proc = n_download_proc
		self.max_length = max_length
		self.train_batch_size = train_batch_size
		self.test_val_batch_size = test_val_batch_size

	def process_owm( self, ds: Dataset ):
		def tokenized( b ):
			tokens = self.tokenizer(
				b[ "text" ], padding = "max_length", max_length = self.max_length, truncation = True,
				padding_side = "left", return_tensors = "pt", return_attention_mask = True, add_special_tokens = False )
			return tokens

		return ds.map( tokenized, batched = True, batch_size = self.train_batch_size )

	def process_gsm8k( self, ds: Dataset ):
		to_answer = lambda q, a: (s := f"Q: {q}\nA: ", a.split( "####" )[ -1 ], len( s ))
		all_prompts = lambda b: [ to_answer( q, a ) for q, a in zip( b[ "question" ], b[ "answer" ] ) ]

		def tokenize( b ):
			return self.tokenizer(
				b, padding = "max_length", max_length = self.max_length, truncation = False,
				padding_side = "left", return_tensors = "pt", return_attention_mask = True, add_special_tokens = False )

		def tokenized( b ):
			questions, answers, answer_locations = [ ], [ ], [ ]
			for q, a, l in all_prompts( b ):
				questions.append( q )
				answers.append( a )
				answer_locations.append( l )

			tokens = tokenize( questions )
			tokens[ "answers" ] = tokenize( answers )[ "input_ids" ]
			return tokens

		return ds.map( tokenized, batched = True, batch_size = self.test_val_batch_size )

	def process_csqa( self, ds: Dataset ):
		def construct_question( q, choices ):
			choice_list = "\n".join(
				[ f"({label}) {choice}" for label, choice in zip( choices[ "label" ], choices[ "text" ] ) ] )
			return f"Q: {q}\n{choice_list}"

		to_answer = lambda q, c, a: (s := f"{construct_question( q, c )}\nA: ", a, len( s ))
		all_prompts = lambda b: [ to_answer( q, c, a ) for q, c, a in
			zip( b[ 'question' ], b[ 'choices' ], b[ 'answerKey' ] ) ]

		def tokenize( b ):
			return self.tokenizer(
				b, padding = "max_length", max_length = self.max_length, truncation = False,
				padding_side = "left", return_tensors = "pt", return_attention_mask = True, add_special_tokens = False )

		def tokenized( b ):
			questions, answers, answer_locations = [ ], [ ], [ ]
			for q, a, l in all_prompts( b ):
				questions.append( q )
				answers.append( a )
				answer_locations.append( l )

			tokens = tokenize( questions )
			tokens[ "answers" ] = tokenize( answers )[ "input_ids" ]
			return tokens

		return ds.map( tokenized, batched = True, batch_size = self.test_val_batch_size )

	def get_train_datasets( self ):
		return self.process_owm(
			load_dataset(
				"open-web-math/open-web-math", split = "train[:1000]", num_proc = self.n_download_proc ) ),

	def get_validation_datasets( self ):
		return (
			self.process_gsm8k( load_dataset( "gsm8k", name = "main", num_proc = self.n_download_proc ) ),
			self.process_csqa( load_dataset( "tau/commonsense_qa", num_proc = self.n_download_proc ) ),
		)

	def prepare_data( self ):
		self.train_dataset = self.get_train_datasets()
		rest = self.get_validation_datasets()
		self.val_datasets = (rest[ 0 ][ "train" ], rest[ 1 ][ "validation" ])
		self.test_datasets = (rest[ 0 ][ "test" ], rest[ 1 ][ "test" ])

	def train_dataloader( self ):
		return DataLoader( self.train_dataset, batch_size = self.train_batch_size, shuffle = True )

	def val_dataloader( self ):
		return (
			DataLoader( self.val_datasets[ 0 ], batch_size = self.test_val_batch_size, shuffle = False ),
			DataLoader( self.val_datasets[ 1 ], batch_size = self.test_val_batch_size, shuffle = False ),
		)


class QuietStar( pl.LightningModule ):
	def __init__(
			self, config, model_factory, learning_rate = 1e-6, weight_decay = 0.001, adam_epsilon = 1e-6,
			warmup_steps = 0 ):
		super().__init__()
		self.save_hyperparameters()
		self.outputs = defaultdict( list )

	def configure_model( self ):
		self.model = self.hparams.model_factory( self.hparams.config )
		self.model.logger = self.logger

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

	def training_step( self, batch, batch_idx ):
		loss = self.model(
			input_ids = batch[ "input_ids" ], labels = batch[ "input_ids" ],
			attention_mask = batch[ "attention_mask" ] ).loss
		return loss

	def validation_step( self, batch, batch_idx, dataloader_idx = 0 ):
		output = self.model.generate( inputs = batch[ "input_ids" ], max_new_tokens = 10, output_logits = True )
		self.outputs[ dataloader_idx ].append( (batch, output) )

	def on_validation_epoch_end( self ):
		flat_outputs = [ ]
		for lst in self.outputs.values():
			flat_outputs.extend( lst )

		with t.no_grad():
			flat_outputs = [ (b.to( "cpu" ), o.to( "cpu" )) for b, o in flat_outputs ]

			log_to_check, a_tok = [ ], [ ]

			for b, o in flat_outputs:
				for a, logits in zip( b[ "answers" ], o ):
					a_tok.append( a.unsqueeze( 0 ) )
					log_to_check.append( logits[ : len( a_tok ) ].unsqueeze( 0 ) )

			logs, ans = t.cat( log_to_check, dim = 0 ), t.cat( a_tok, dim = 0 )

			losses = t.nn.functional.cross_entropy(
				einx.rearrange( "... v -> (...) v", logs ), einx.rearrange( "... -> (...)", ans ), reduction = "none" )
			losses = losses.reshape_as( ans )
			loss = losses.nansum( dim = -1 ).mean()
			self.log( "val_loss", loss, prog_bar = True )


tokenizer = AutoTokenizer.from_pretrained( "Qwen/Qwen2.5-0.5B" )
tokenizer.padding_side = "left"
tokenizer.add_special_tokens( { "additional_special_tokens": [ "<|startthought|>", "<|endthought|>" ] } )

config = QwenThoughtModelConfig.from_pretrained( "Qwen/Qwen2.5-0.5B",
	start_thought_token = tokenizer.convert_tokens_to_ids( "<|startthought|>" ),
	end_thought_token = tokenizer.convert_tokens_to_ids( "<|endthought|>" ),
	initial_start_thought_token = tokenizer.convert_tokens_to_ids("---"),
	initial_end_thought_token = tokenizer.convert_tokens_to_ids("---"),
	host_thought_token_embeddings = True
)


def model_factory( config ):
	boft_config = BOFTConfig(
		boft_block_size = 4,
		boft_n_butterfly_factor = 2,
		boft_dropout = 0.1,
		bias = "boft_only",
		target_modules = "all-linear"
	)

	model = QwenThoughtModel.from_pretrained( "Qwen/Qwen2.5-0.5B", config = config )

	model.resize_token_embeddings( len( tokenizer ) )

	model._lm_model = get_peft_model( model._lm_model, boft_config )

	return model


dm = QuietStarDataModule(
	tokenizer,
	n_download_proc = 1,
)

model = QuietStar( config, model_factory )

checkpoint_callback = pl.callbacks.ModelCheckpoint(
	every_n_train_steps = 10,
	save_top_k = -1,
)

trainer = pl.Trainer(
	default_root_dir = "gs://ddanilovic-llm-storage",
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

trainer.fit( model, datamodule = dm )
