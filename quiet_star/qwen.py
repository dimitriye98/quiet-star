import torch as t
from peft import BOFTConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from quiet_star.thought_model import ThoughtModel, ThoughtModelConfig


class QwenThoughtModelConfig( Qwen2Config, ThoughtModelConfig ):
	def __init__( self, **kwargs ):
		super().__init__( **kwargs )
		self.host_thought_token_embeddings = kwargs.pop( "host_thought_token_embeddings", False )


class QwenThoughtModel( ThoughtModel ):
	config_class = QwenThoughtModelConfig

	def __init__( self, config: QwenThoughtModelConfig ):
		super().__init__( config )
		self._lm_model = Qwen2ForCausalLM( config )

		# if config.initial_start_thought_token is not None:
		# 	self._lm_model.get_input_embeddings().weight.data[ config.initial_start_thought_token ] = self.start_thought_token
		# if config.initial_end_thought_token is not None:
		# 	self._lm_model.get_input_embeddings().weight.data[ config.initial_end_thought_token ] = self.end_thought_token
		# if config.host_thought_token_embeddings:
		# 	self.start_token_embedding = t.nn.Parameter( self._lm_model.get_input_embeddings().weight.data[ config.initial_start_thought_token ] )
		# 	self.end_token_embedding = t.nn.Parameter( self._lm_model.get_input_embeddings().weight.data[ config.initial_end_thought_token ] )
		#
		# if config.tie_word_embeddings:
		# 	output_embeds = t.nn.Linear( config.hidden_size, config.vocab_size, bias = False )
		# 	output_embeds.weight = self._lm_model.get_input_embeddings().weight
		#
		# 	self._lm_model.set_output_embeddings( output_embeds )

	def _init_weights( self, module ):
		if module == self:
			print("Setting initial weights for Qwen model")
			if self.config.host_thought_token_embeddings:
				self.start_token_embedding = t.nn.Parameter( self._lm_model.get_input_embeddings().weight.data[ self.config.initial_start_thought_token ] )
				self.end_token_embedding = t.nn.Parameter( self._lm_model.get_input_embeddings().weight.data[ self.config.initial_end_thought_token ] )
		else:
			super()._init_weights( module )

	def forward( self, *args, **kwargs ):
		if self.config.host_thought_token_embeddings:
			self._lm_model.get_input_embeddings().weight.data[ self.start_thought_token ] = self.start_token_embedding
			self._lm_model.get_input_embeddings().weight.data[ self.end_thought_token ] = self.end_token_embedding
		return super().forward( *args, **kwargs )

	@property
	def lm_model( self ):
		return self._lm_model

# class QwenBOFTThoughtModelConfig( QwenThoughtModelConfig, ThoughtModelConfig, BOFTConfig):
# 	def __init__( self, **kwargs ):
# 		super().__init__( **kwargs )
#
#
# class QwenBOFTThoughtModel( QwenThoughtModel ):
# 	config_class = QwenBOFTThoughtModelConfig
#
# 	def __init__( self, config: QwenBOFTThoughtModelConfig ):
# 		super().__init__( config )
# 		self._lm_model = get_peft_model(self._lm_model, config)
# 		self.start_token_embedding = self._lm_model.get_input_embeddings().weight.data[ self.start_thought_token ]
# 		self.end_token_embedding = self._lm_model.get_input_embeddings().weight.data[ self.end_thought_token ]
#
# 	def forward(
# 			self, *args, **kwargs ):
# 		self._lm_model.get_input_embeddings().weight.data[ self.start_thought_token ] = self.start_token_embedding
# 		self._lm_model.get_input_embeddings().weight.data[ self.end_thought_token ] = self.end_token_embedding
# 		return super().forward( *args, **kwargs )
