import dataclasses
import inspect
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Sequence, Dict

import torch
import torch as t
import wrapt
from dataclasses import dataclass
from toolz import memoize

pymin = min
pymax = max
from einx.op import *
from torch import Tensor, LongTensor, FloatTensor
from torch.nn.functional import gumbel_softmax, relu, cross_entropy, pad
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig, Cache, DynamicCache, StaticCache, \
	GenerationConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging

logger = logging.get_logger( __name__ )


@wrapt.decorator
def inherit_device_and_dtype( wrapped, instance, args, kwargs ):
	old_dtype = t.get_default_dtype()
	t.set_default_dtype( instance.dtype )
	with t.device( instance.device ):
		ret = wrapped( *args, **kwargs )
	t.set_default_dtype( old_dtype )
	return ret


_config_params = {
	"thought_depth": 12,
	"start_thought_token": None,
	"end_thought_token": None,
	"thought_temperature": 1.0,
	"reinforce_temperature": 3.0,
	"n_thoughts": 2,
	"look_ahead": 4,
	"base_loss_beta": 1.0,
	"policy_loss_beta": 1e6,
	"mixer_head_activation": "relu",
	"mixer_head_hidden_layers": 1
}

compilation_cache = { }


@wrapt.decorator
def compile_on_first_call( wrapped, instance, args, kwargs ):
	if wrapped not in compilation_cache:
		ret = wrapped( *args, **kwargs )
		# Check again to handle recursive calls
		if wrapped not in compilation_cache:
			compilation_cache[ wrapped ] = t.compile( wrapped )
		return ret
	else:
		return compilation_cache[ wrapped ]( *args, **kwargs )


@t.compiler.substitute_in_graph( compile_on_first_call )
@wrapt.decorator
def _compile_on_first_call( wrapped, instance, args, kwargs ):
	f = compilation_cache.get( wrapped, None )
	if f is not None:
		return f( *args, **kwargs )

	file, line = None, None
	try:
		file = inspect.getsourcefile( wrapped )
		_, line = inspect.getsourcelines( wrapped )
	except (TypeError, OSError):
		pass
	file = file if file is not None else ""
	line = ":" + str( line ) if line is not None else ""
	raise RuntimeError( f"call function {wrapped.__name__} at {file}{line} before attempting to compile" )


class ThoughtModelConfig( PretrainedConfig, ABC ):

	def __init__( self, **kwargs ):
		super().__init__( **kwargs )
		for k, v in _config_params.items():
			val = kwargs.pop( k, v )
			setattr( self, k, val )
		self.mixer_head_hidden_size = kwargs.pop( "mixer_head_hidden_size", self.hidden_size )
		self.initial_start_thought_token = kwargs.pop( "initial_start_thought_token", None )
		self.initial_end_thought_token = kwargs.pop( "initial_end_thought_token", None )


class WeightedMixerHead( t.nn.Module ):
	def __init__( self, config ):
		super().__init__()
		hs, ms, nl = config.hidden_size, config.mixer_head_hidden_size, config.mixer_head_hidden_layers
		input_layer = [ t.nn.Linear( 2 * hs, ms ) ]
		mid_layer = [ t.nn.Linear( ms, ms ), ACT2FN[ config.mixer_head_activation ] ] * nl
		output_layer = [ ACT2FN[ config.mixer_head_activation ], t.nn.Linear( ms, 1 ) ]

		self.mlp = t.nn.Sequential( *input_layer, *mid_layer, *output_layer )

	def forward( self, pre_thought_hidden_state, post_thought_hidden_state ):
		catted_states = t.cat( (pre_thought_hidden_state, post_thought_hidden_state), dim = -1 )
		return self.mlp( catted_states )


class ThoughtModel( PreTrainedModel, GenerationMixin, ABC ):
	base_model_prefix = "_lm_model"
	_supports_cache_class = True
	_supports_static_cache = True

	def _init_weights( self, module ):
		std = self.config.initializer_range
		if isinstance( module, t.nn.Linear ):
			module.weight.data.normal_( mean = 0.0, std = std )
			if module.bias is not None:
				module.bias.data.zero_()
		elif isinstance( module, t.nn.Embedding ):
			module.weight.data.normal_( mean = 0.0, std = std )
			if module.padding_idx is not None:
				module.weight.data[ module.padding_idx ].zero_()

	def __init__( self, config ):
		super().__init__( config )
		self.thought_depth = config.thought_depth
		self.start_thought_token = config.start_thought_token
		self.end_thought_token = config.end_thought_token
		self.thought_temperature = config.thought_temperature
		self.reinforce_temperature = config.reinforce_temperature
		self.n_thoughts = config.n_thoughts
		self.look_ahead = config.look_ahead
		self.base_loss_beta = config.base_loss_beta
		self.policy_loss_beta = config.policy_loss_beta
		self.mixer_head = WeightedMixerHead( config )

	@property
	@abstractmethod
	def lm_model( self ):
		...

	@dataclass
	class ForwardParams:
		input_ids: LongTensor = None
		attention_mask: Optional[ Tensor ] = None
		position_ids: Optional[ LongTensor ] = None
		past_key_values: Optional[ Cache | Tuple[ Tuple[ FloatTensor ] ] ] = None
		inputs_embeds: Optional[ FloatTensor ] = None
		labels: Optional[ LongTensor ] = None
		use_cache: Optional[ bool ] = None
		output_attentions: Optional[ bool ] = None
		output_hidden_states: Optional[ bool ] = None
		return_dict: Optional[ bool ] = None
		cache_position: Optional[ LongTensor ] = None
		num_logits_to_keep: int = None
		thought_mask: Optional[ Tensor ] = None

		def as_kwargs( self ):
			return { f.name: getattr( self, f.name ) for f in dataclasses.fields( self ) }

	@staticmethod
	def crop_cache( cache: Cache, l: int ):
		if isinstance( cache, DynamicCache ):
			cache.crop( l )
		elif isinstance( cache, StaticCache ):
			# Relies on unexposed internals, might break if they change
			for layer_idx in range( len( cache.key_cache ) ):
				if cache.key_cache[ layer_idx ] is not None:
					cache.key_cache[ layer_idx ][ :, :, l + 1: ].zero_()
				if cache.value_cache[ layer_idx ] is not None:
					cache.value_cache[ layer_idx ][ :, :, l + 1: ].zero_()
		else:
			raise NotImplementedError( "Only DynamicCache and StaticCache are currently supported" )

	# """Reorders the cache for beam search, given the selected beam indices."""
	# for layer_idx in range(len(cache.key_cache)):
	# 	if cache.key_cache[ layer_idx ]:
	# 		device = cache.key_cache[layer_idx].device
	# 		# cache.key_cache[layer_idx] = cache.key_cache[layer_idx].index_select(0, beam_idx.to(device))
	# 		cache.key_cache[layer_idx] = get_at("... [l], i, ... i")
	# 	if cache.value_cache[ layer_idx ]:
	# 		device = cache.value_cache[layer_idx].device
	# 		# cache.value_cache[layer_idx] = cache.value_cache[layer_idx].index_select(0, beam_idx.to(device))

	def prepare_inputs_for_generation(
			self,
			input_ids: t.LongTensor,
			past_key_values: Optional[ Cache ] = None,
			attention_mask: Optional[ LongTensor ] = None,
			inputs_embeds: Optional[ FloatTensor ] = None,
			cache_position: Optional[ LongTensor ] = None,
			num_logits_to_keep: int = 1,
			**kwargs,
	):
		return super().prepare_inputs_for_generation(
			input_ids, past_key_values, attention_mask, inputs_embeds, cache_position,
			num_logits_to_keep = num_logits_to_keep, **kwargs )

	# @t.no_grad()
	# @t.compiler.disable()
	# def inference_forward( self, params: ForwardParams ):
	# 	if params.inputs_embeds is not None:
	# 		raise NotImplementedError( "Input embeddings not supported for thoughtful inference" )
	# 	if params.output_attentions is not None:
	# 		raise NotImplementedError( "Output attentions not supported for thoughtful inference" )
	# 	if params.output_hidden_states is not None:
	# 		raise NotImplementedError( "Output hidden states not supported for thoughtful inference" )
	# 	if params.num_logits_to_keep is None or params.num_logits_to_keep > 1:
	# 		raise NotImplementedError(
	# 			"Only single-token inference is supported for thoughtful inference, as it is sequential by nature" )
	#
	# 	return_legacy_cache = False
	# 	if params.use_cache and not isinstance( params.past_key_values, Cache ):
	# 		return_legacy_cache = True
	# 		if params.past_key_values is None:
	# 			params.past_key_values = DynamicCache()
	# 		else:
	# 			params.past_key_values = DynamicCache.from_legacy_cache( params.past_key_values )
	# 	# logger.warning_once(
	# 	# 	"We detected that you are passing `params.past_key_values` as a tuple of tuples. This is deprecated and "
	# 	# 	"will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
	# 	# 	"(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
	# 	# )
	#
	# 	# context = t.empty(
	# 	# 	params.input_ids.shape[ :-1 ] + (params.input_ids.shape[ -1 ] + self.thought_depth + 2,),
	# 	# 	dtype = params.input_ids.dtype )
	# 	# context[ ..., :params.input_ids.shape[ -1 ] ] = params.input_ids
	# 	context = pad( params.input_ids, (0, self.thought_depth + 2) )
	#
	# 	if params.past_key_values is not None:
	# 		if isinstance( params.past_key_values, StaticCache ):
	# 			mask_offset = params.past_key_values.get_max_cache_shape()
	# 		else:
	# 			mask_offset = params.past_key_values.get_seq_length()
	# 	else:
	# 		mask_offset = 0
	#
	# 	if params.attention_mask is None:
	# 		# padding_mask = t.ones( context.shape[ :-1 ] + (mask_len,), dtype = self.dtype )
	# 		if params.past_key_values is not None:
	# 			padding_mask = pad( t.ones_like( context ), (0, mask_offset), value = 1 )
	# 		else:
	# 			padding_mask = t.ones_like( context )
	# 	else:
	# 		# if params.past_key_values is not None:
	# 		# 	padding_mask = pad(t.ones_like(context), (0, mask_offset), value = 1)
	# 		# 	padding_mask[..., :params.attention_mask.size(-1)] = params.attention_mask
	# 		#
	# 		# padding_mask = t.empty( context.shape[ :-1 ] + (mask_len,), dtype = params.attention_mask.dtype )
	# 		# padding_mask[ ..., :params.attention_mask.shape[ -1 ] ] = params.attention_mask
	# 		# padding_mask[ ..., params.attention_mask.shape[ -1 ]: ] = 1
	# 		padding_mask = pad( t.ones_like( context ), (0, mask_offset), value = 1 )
	# 		padding_mask[ ..., :params.attention_mask.shape[ -1 ] ] = params.attention_mask
	# 		padding_mask[ ..., params.attention_mask.shape[ -1 ]: ] = 1
	#
	# 	offset = params.input_ids.shape[ -1 ]
	# 	context[ ..., offset ] = self.start_thought_token
	#
	# 	if params.position_ids is None:
	# 		first_index = padding_mask.argmax( dim = -1 )
	# 		# params.position_ids = t.ones_like( context )
	# 		params.position_ids = (rearrange(
	# 			"l -> b... l",
	# 			t.arange( context.shape[ -1 ] ),
	# 			b = context.shape[ :-1 ] ) - first_index.unsqueeze( -1 )).contiguous()
	# 		# Adjust by first unmasked index and clamp to 0, since we assume left-padding
	#
	# 		params.position_ids = params.position_ids.maximum( t.zeros_like( params.position_ids ) )
	# 	else:
	# 		first_index = params.position_ids[ ..., -1 ]
	# 		additional_position_ids = (rearrange(
	# 			"l -> b... l",
	# 			t.arange( context.shape[ -1 ] - params.position_ids.shape[ -1 ] ),
	# 			b = params.position_ids.shape[ :-1 ] ) + first_index.unsqueeze( -1 ) + 1).contiguous()
	#
	# 		params.position_ids = t.cat( (params.position_ids, additional_position_ids), dim = -1 )
	#
	# 	if params.use_cache and params.cache_position is None:
	# 		past_seen_tokens = params.past_key_values.get_seq_length() if params.past_key_values is not None else 0
	# 		params.cache_position = t.arange(
	# 			past_seen_tokens, past_seen_tokens + offset, device = self.device
	# 		)
	#
	# 	init_params = replace(
	# 		params,
	# 		position_ids = params.position_ids[ ..., :offset ],
	# 		attention_mask = padding_mask[ ..., :mask_offset + offset ],
	# 		# past_key_values (use initial)
	# 		# inputs_embeds (asserted unsupported)
	# 		# labels (always None if this is inference)
	# 		# use_cache (constant)
	# 		output_attentions = False,
	# 		output_hidden_states = True,
	# 		return_dict = True,
	# 		# cache_position (use initial)
	# 		num_logits_to_keep = 1,
	# 	)
	#
	# 	init_output = self.lm_model( **init_params.as_kwargs() )
	# 	init_logits = init_output.logits
	# 	init_hidden = init_output.hidden_states[ -1 ][ ..., -1, : ].clone()
	# 	params.past_key_values = init_output.past_key_values
	# 	init_cache_len = params.past_key_values.get_seq_length() if params.past_key_values is not None else None
	#
	# 	del init_params, init_output
	#
	# 	if params.cache_position is not None:
	# 		# Exactly one unseen token, the forced start token
	# 		params.cache_position = params.cache_position[ ..., -1: ] + 1
	#
	# 	for i in range( 1, self.thought_depth + 1 ):
	# 		if params.cache_position is not None:
	# 			# Exactly one unseen token
	# 			inp = context[ ..., offset + i - 1:offset + i ]
	# 			pos = params.position_ids[ ..., offset + i - 1:offset + i ]
	# 		# inp = get_at( "... [l], c -> ... c", context, params.cache_position )
	# 		# pos = get_at( "... [l], c -> ... c", params.position_ids, params.cache_position )
	# 		else:
	# 			inp = context[ ..., :offset + i ]
	# 			pos = params.position_ids[ ..., :offset + i ]
	#
	# 		gen_params = replace(
	# 			params,
	# 			input_ids = inp,
	# 			attention_mask = padding_mask[ ..., :mask_offset + offset + i ],
	# 			position_ids = pos,
	# 			# past_key_values (updated in loop)
	# 			# inputs_embeds (asserted unsupported)
	# 			# labels (always None if this is inference)
	# 			# use_cache (constant)
	# 			output_attentions = False,
	# 			output_hidden_states = False,
	# 			return_dict = True,
	# 			# cache_position (updated in loop)
	# 			num_logits_to_keep = 1,
	# 		)
	#
	# 		outputs = self.lm_model( **gen_params.as_kwargs() )
	#
	# 		if params.cache_position is not None:
	# 			# Exactly one unseen token
	# 			params.cache_position = params.cache_position[ ..., -1: ] + 1
	#
	# 		params.past_key_values = outputs.past_key_values
	# 		logits = outputs.logits
	# 		logits[ ..., self.start_thought_token ] = -t.inf
	# 		logits[ ..., self.end_thought_token ] = -t.inf
	#
	# 		if self.thought_temperature == 0.0:
	# 			next_tokens = logits[ ..., 0 ].argmax( dim = -1 )
	# 		else:
	# 			next_tokens = gumbel_softmax(
	# 				logits[ ..., 0 ], tau = self.thought_temperature, hard = True ).argmax( dim = -1 )
	#
	# 		context[ ..., offset + i ] = next_tokens
	#
	# 	context[ ..., offset + self.thought_depth + 1 ] = self.end_thought_token
	#
	# 	if params.cache_position is not None:
	# 		# One more unseen token, the forced end token
	# 		params.cache_position = t.cat( (params.cache_position, params.cache_position[ ..., -1: ] + 1), dim = -1 )
	#
	# 		# Exactly two unseen tokens
	# 		inp = context[ ..., offset + self.thought_depth:offset + self.thought_depth + 2 ]
	# 		pos = params.position_ids[ ..., offset + self.thought_depth:offset + self.thought_depth + 2 ]
	# 	else:
	# 		inp = context
	# 		pos = params.position_ids
	#
	# 	params_final = replace(
	# 		params,
	# 		input_ids = inp,
	# 		attention_mask = padding_mask,
	# 		position_ids = pos,
	# 		# past_key_values (updated in loop)
	# 		# inputs_embeds (asserted unsupported)
	# 		# labels (always None if this is inference)
	# 		# use_cache (constant)
	# 		output_attentions = False,
	# 		output_hidden_states = True,
	# 		return_dict = True,
	# 		# cache_position (updated in loop)
	# 		num_logits_to_keep = 1,
	# 	)

	# final_outputs = self.lm_model( **params_final.as_kwargs() )
	# params.past_key_values = final_outputs.past_key_values
	# if isinstance( params.past_key_values, DynamicCache ) or isinstance( params.past_key_values, StaticCache ):
	# 	self.crop_cache( params.past_key_values, init_cache_len )
	# else:
	# 	warnings.warn(
	# 		f"Cache is of unsupported type {type( params.past_key_values )} and can't be cropped, falling back to invalidating it instead" )
	# 	params.past_key_values = None
	#
	# alpha = self.mixer_head( init_hidden, final_outputs.hidden_states[ -1 ][ ..., -1, : ].clone() )
	# logits = alpha * init_logits + (1 - alpha) * final_outputs.logits
	#
	# if params.past_key_values:
	# 	if return_legacy_cache:
	# 		params.past_key_values = params.past_key_values.to_legacy_cache()
	#
	# if params.return_dict:
	# 	return CausalLMOutputWithPast(
	# 		logits = logits,
	# 		past_key_values = params.past_key_values,
	# 	)
	# else:
	# 	return (logits,) + (pkv,) if (pkv := params.past_key_values) is not None else ()

	@torch.no_grad()
	def inference_forward( self, params ):
		input_ids = params.input_ids
		if params.inputs_embeds is not None:
			raise NotImplementedError(
				"The default implementation of inference_forward does not support inputs_embeds" )
		if input_ids is None:
			raise ValueError( "input_ids must be provided for inference" )
		with t.device( input_ids.device ):
			attention_mask = params.attention_mask

			if attention_mask is None:
				# StaticCache
				if isinstance( params.past_key_values, StaticCache ):
					target_length = params.past_key_values.get_max_cache_shape()
				# DynamicCache or no cache
				else:
					past_seen_tokens = params.past_key_values.get_seq_length() if params.past_key_values is not None else 0
					target_length = past_seen_tokens + input_ids.shape[ -1 ] + 1
				attention_mask = t.ones( (*input_ids.shape[ :-1 ], target_length) )

			if attention_mask.shape[ :-1 ] != input_ids.shape[ :-1 ]:
				raise NotImplementedError(
					"The default implementation of inference_forward does not support custom 4d attention masks" )

			return_legacy_cache = False
			if params.use_cache and not isinstance( params.past_key_values, Cache ):
				return_legacy_cache = True
				if params.past_key_values is None:
					params.past_key_values = DynamicCache()
				else:
					params.past_key_values = DynamicCache.from_legacy_cache( params.past_key_values )

			cache = params.past_key_values
			cache_pos = params.cache_position

			position_ids = params.position_ids
			if cache_pos is None:
				past_seen_tokens = params.past_key_values.get_seq_length() if params.past_key_values is not None else 0
				cache_pos = torch.arange(
					past_seen_tokens, past_seen_tokens + input_ids.shape[-1]
				)
			if position_ids is None:
				position_ids = cache_pos.unsqueeze(0)

			extend = lambda f, T, l: t.cat( (T, f( (*T.shape[ :-1 ], l), dtype = T.dtype )), dim = -1 )
			context = extend( t.empty, input_ids, self.thought_depth + 2 )
			context_mask = extend( t.ones, attention_mask, self.thought_depth + 2 )
			offset = input_ids.shape[ -1 ]
			mask_offset = attention_mask.shape[ -1 ]

			# if position_ids is None:
			# 	position_ids = t.arange( input_ids.shape[ -1 ], device = input_ids.device ).tile(
			# 		(*input_ids.shape[ :-1 ], 1) )

			pos_id_ext = t.arange( self.thought_depth + 2, device = position_ids.device ).tile(
				(*position_ids.shape[ :-1 ], 1) ) + position_ids[ ..., -1, None ] + 1
			context_position_ids = t.cat( (position_ids, pos_id_ext), dim = -1 )

			outputs = self.lm_model(
				input_ids = context[ ..., :offset ],
				position_ids = context_position_ids[ ..., :offset ],
				attention_mask = attention_mask[ ..., :mask_offset ],
				past_key_values = cache,
				use_cache = params.use_cache,
				output_attentions = False,
				output_hidden_states = True,
				return_dict = True,
				cache_position = cache_pos,
				num_logits_to_keep = 1,
			)

			context[ ..., offset ] = self.start_thought_token
			init_cache_len = params.past_key_values.get_seq_length() if params.past_key_values is not None else None

			# Get the logits and the hidden state so we can let the outputs deallocate
			pl, ph = outputs.logits, outputs.hidden_states[ -1 ][ ..., -1, : ].clone()

			def update_cache_pos( p ):
				if p is not None:
					return p[ ..., -1: ] + 1

			cache_pos = update_cache_pos( cache_pos )

			def sample( logits ):
				logits[ ..., self.start_thought_token ] = -t.inf
				logits[ ..., self.end_thought_token ] = -t.inf
				if self.thought_temperature == 0.0:
					tok = logits[ ..., 0 ].argmax( dim = -1 )
				else:
					tok = gumbel_softmax(
						logits[ ..., 0 ],
						tau = self.thought_temperature,
						hard = True ).argmax( dim = -1 )

				return tok

			for i in range( 1, self.thought_depth + 1 ):
				if cache_pos is not None:
					inp = context[ ..., offset + i - cache_pos.shape[ -1 ]:offset + i ]
					pos = context_position_ids[ ..., offset + i - cache_pos.shape[ -1 ]:offset + i ]
				else:
					inp = context[ ..., :offset + i ]
					pos = context_position_ids[ ..., :offset + i ]

				outputs = self.lm_model(
					input_ids = inp,
					position_ids = pos,
					attention_mask = context_mask[..., :mask_offset + i ],
					past_key_values = cache,
					use_cache = params.use_cache,
					output_attentions = False,
					output_hidden_states = True,
					return_dict = True,
					cache_position = cache_pos,
					num_logits_to_keep = 1,
				)

				context[ ..., offset + i ] = sample( outputs.logits )
				cache_pos = update_cache_pos( cache_pos )

			i += 1
			context[ ..., offset + i ] = self.end_thought_token
			i += 1

			if cache_pos is not None:
				cache_pos = t.cat( (cache_pos, cache_pos[ ..., -1: ] + 1), dim = -1 )
				inp = context[ ..., offset + i - cache_pos.shape[ -1 ]:offset + i ]
				pos = context_position_ids[ ..., offset + i - cache_pos.shape[ -1 ]:offset + i ]
			else:
				inp = context[ ..., :offset + i ]
				pos = context_position_ids[ ..., :offset + i ]

			outputs = self.lm_model(
				input_ids = inp,
				position_ids = pos,
				attention_mask = context_mask[..., :mask_offset + i ],
				past_key_values = cache,
				use_cache = params.use_cache,
				output_attentions = False,
				output_hidden_states = True,
				return_dict = True,
				cache_position = cache_pos,
				num_logits_to_keep = 1,
			)

			ql, qh = outputs.logits, outputs.hidden_states[ -1 ][ ..., -1, : ]

			alpha = self.mixer_head( ph, qh )

			logits = pl * (1 - alpha) + ql * alpha

			if isinstance( params.past_key_values, DynamicCache ) or isinstance( params.past_key_values, StaticCache ):
				self.crop_cache( params.past_key_values, init_cache_len )
			else:
				warnings.warn(
					f"Cache is of unsupported type {type( params.past_key_values )} and can't be cropped, falling back to invalidating it instead" )
				params.past_key_values = None

			if params.return_dict:
				return CausalLMOutputWithPast(
					logits = logits,
					past_key_values = outputs.past_key_values,
				)
			else:
				return (logits,) + ((pkv,) if (pkv := outputs.past_key_values) is not None else tuple())

	@staticmethod
	@t.compiler.disable
	@memoize
	def _prepare_thought_mask_encoding( d: int, l: int ):
		if d == 1:
			return t.ones( l, l ).tril().unsqueeze( 0 ).unsqueeze( 0 )

		thought_thought = multiply( "d..., l... -> d... l...", t.ones( d, d - 1 ).tril( -1 ), t.eye( l ) )
		causal = rearrange( "l... -> D l..l", t.ones( l, l ).tril(), D = d )
		return rearrange( "D l L, d D l L -> (1 + d) D l L", causal, thought_thought )

	@classmethod
	@compile_on_first_call
	def prepare_thought_mask( cls, d: int, l: int, padding_mask: Tensor ):
		assert padding_mask.shape[ -1 ] == l
		thought_mask = cls._prepare_thought_mask_encoding( d, l )
		return multiply( "b... l, d D l L -> b... d D l L", padding_mask, thought_mask )

	@staticmethod
	@compile_on_first_call
	def flatten_thought_mask( thought_mask: Tensor, k = None ):
		k = tuple() if k is None else tuple( k )
		return rearrange( "b... k... d D l L -> b... k... (d l) (D L)", thought_mask, k = k )

	@staticmethod
	def cache_layers_to_cache_positions( l: int, cache_layers: Tensor ):
		return (t.arange( l ) + l * cache_layers.reshape( -1, 1 )).reshape( -1 )

	@compile_on_first_call
	@inherit_device_and_dtype
	def broadcast_logits(
			self, thought_state: LongTensor, mask: Tensor, params: ForwardParams, new_layers ):
		d, l = thought_state.shape[ -2 ], thought_state.shape[ -1 ]
		cache_position = t.arange( 0, l ) + l * new_layers

		thought_mask = self.prepare_thought_mask( d, l, mask )
		thought_mask = t.zeros_like( thought_mask ).masked_fill( ~(thought_mask.to( t.bool )), -t.inf )

		causal_mask = rearrange(
			"b... d... l... -> (b...) k (d l)...", thought_mask, d = (d, d), l = (l, l),
			k = self.lm_model.config.num_attention_heads )

		layer_shape = thought_state.shape[ :-2 ] + thought_state.shape[ -1: ]

		outputs: CausalLMOutputWithPast = self.lm_model(
			rearrange( "b... d l -> (b...) (d l)", thought_state, d = d, l = l ),
			attention_mask = causal_mask,

			# past_key_values = params.past_key_values,
			# use_cache = params.use_cache,
			# cache_position = params.cache_position,

			return_dict = True,
			num_logits_to_keep = l * num_layers_to_keep,
			output_hidden_states = True )

		# params.past_key_values = outputs.past_key_values

		logits_out = rearrange(
			"(b...) (d l) v -> b... d l v", outputs.logits, **solve( "b... _ l", thought_state ) )
		hidden_out = rearrange(
			"(b...) (d l) e -> b... d l e", outputs.hidden_states[ -1 ][ ..., -l * num_layers_to_keep:, : ],
			d = num_layers_to_keep,
			**solve( "b... _ l", thought_state ) )

		return logits_out, hidden_out

	@compile_on_first_call
	@t.no_grad()
	def broadcast_tokens( self, thought_state: LongTensor, mask: Tensor, params: ForwardParams, d = None ):
		if d is None:
			logits, _ = self.broadcast_logits( thought_state, mask, params )
		else:
			logits, _ = self.broadcast_logits( thought_state, mask, params, num_layers_to_keep = 0 )

		logits[ ..., self.start_thought_token ] = -t.inf
		logits[ ..., self.end_thought_token ] = -t.inf

		if self.thought_temperature == 0.0:
			return logits.argmax( dim = -1 )
		else:
			return gumbel_softmax( logits, tau = self.thought_temperature, dim = -1, hard = True ).argmax( dim = -1 )

	# @t.no_grad()
	# @compile_on_first_call
	# def generate_thoughts( self, inputs: LongTensor, mask: Tensor, params: ForwardParams ):
	# 	input_layer = rearrange( "b... l -> b... n 1 l", inputs, n = self.n_thoughts )
	# 	mask = rearrange( "b... l -> b... n l", mask, n = self.n_thoughts )
	# 	global_mask = rearrange(
	# 		"b... n l -> b... n d l", t.zeros_like( mask ), d = self.thought_depth + 3 + self.look_ahead )
	#
	# 	layer_shape = list( input_layer.shape )
	# 	thought_state_shape = copy( layer_shape )
	# 	thought_state_shape[ -2 ] = self.thought_depth + 3 + self.look_ahead
	#
	# 	thought_state = t.empty( thought_state_shape, dtype = t.long )
	# 	set_at( "b... [d] l, [i], b... i l", thought_state, [ 0 ], input_layer )
	# 	set_at( "b... [d] l, [i], i", thought_state, [ 1 ], [ self.start_thought_token ] )
	# 	for i in range( 2, self.thought_depth + 2 ):
	# 		set_at(
	# 			"b... [d] l, [i], b... i l",
	# 			thought_state, [ i ],
	# 			self.broadcast_tokens( thought_state, global_mask, params, d = i ) )
	# 		set_at(
	# 			"b... [d] l, [i], i",
	# 			global_mask, [ i ], [ 1 ] )
	# 	thought_state = set_at(
	# 		"b... [d] l, [i], i",
	# 		thought_state,
	# 		[ self.thought_depth + 2 ],
	# 		[ self.end_thought_token ] )
	# 	set_at(
	# 		"b... [d] l, [i], i",
	# 		global_mask, [ self.thought_depth + 2 ], [ 1 ] )
	#
	# 	return thought_state

	class ThoughtState:
		@property
		def l( self ):
			return self._l

		@property
		def d( self ):
			return self._d

		@property
		def n( self ):
			return self._n

		def __init__(
				self,
				model,
				d: int,
				n: int,
				token_dtype,
				logit_dtype,
				hidden_dtype,
				hidden_container_provider,
				params ):
			self._model = model
			padding_mask = params.attention_mask
			self._l = padding_mask.shape[ -1 ]
			self._d = d
			self._n = n
			self._raw_tensor = rearrange(
				"... l -> ... n d l", t.empty_like( padding_mask, dtype = token_dtype ), n = self.n,
				d = self.d ).contiguous()
			self._logits = rearrange(
				"... -> ... v", t.empty_like( self._raw_tensor, dtype = logit_dtype ),
				v = self._model.lm_model.config.vocab_size ).contiguous()
			self._token_layers = t.full( (d,), False, dtype = t.bool )
			self._logit_layers = t.full_like( self._token_layers, False )
			self._new_lower = 0
			self._new_upper = 0
			self._layer_dim = self._raw_tensor.ndim - 2
			self.hidden_states = hidden_container_provider( self.l, self.d, self.n, hidden_dtype )
			self._logit_getter = self._LogitGetter( self )
			self._thought_mask = self.ThoughtMask(
				self.l, self.d, self._model.lm_model.config.num_attention_heads,
				rearrange( "b... l -> b... n l", padding_mask, n = self.n ) )
			self._params = params

		@property
		def thought_mask( self ):
			return self._thought_mask

		class ThoughtMask:
			def __init__( self, l: int, d: int, k: int, padding_mask: Tensor ):
				self._k = k
				self._l = l
				self._thought_mask = self.prepare_thought_mask( d, l, padding_mask )

			def __getitem__( self, idx ):
				d, D = idx

				return self.invert_mask( self.flatten_thought_mask( self._thought_mask[ ..., d, D, :, : ], self._k ) )

			@staticmethod
			@t.compiler.disable
			@memoize
			def _prepare_thought_mask_encoding(
					d: int, l: int, device = t.get_default_device() ):
				with t.device( device ):
					if d == 1:
						ret = t.ones( l, l ).tril().unsqueeze( 0 ).unsqueeze( 0 )
					else:
						thought_thought = multiply(
							"d..., l... -> d... l...", t.ones( d, d - 1 ).tril( -1 ), t.eye( l ) )
						causal = rearrange( "l... -> d l...", t.ones( l, l ).tril(), d = d )
						ret = rearrange( "d l L, d D l L -> d (1 + D) l L", causal, thought_thought )
					return ret

			@classmethod
			def prepare_thought_mask( cls, d: int, l: int, padding_mask: Tensor ):
				assert padding_mask.shape[ -1 ] == l
				thought_mask = cls._prepare_thought_mask_encoding( d, l, padding_mask.device )
				return multiply( "b... l, d D l L -> b... d D l L", padding_mask, thought_mask )

			@staticmethod
			def flatten_thought_mask( thought_mask: Tensor, k = None ):
				k = tuple() if k is None else (k,)
				return rearrange( "b... d D l L -> (b...) (k...) (d l) (D L)", thought_mask, k = k )

			@staticmethod
			def invert_mask( mask: Tensor ):
				return t.zeros_like( mask ).masked_fill( mask == 0, t.finfo( mask.dtype ).min )

		def _broadcast_logits( self, layer ):
			if layer == 0:
				raise ValueError(
					"this abstraction is generative, that is logits beget tokens, not the other way around, so layer 0 is the input" )
			if isinstance( layer, slice ):
				for i in range(
						layer.start if layer.start is not None else 0, layer.stop if layer.stop is not None else self.d,
						layer.step if layer.step is not None else 1 ):
					self._broadcast_logits( i )
				return
			elif isinstance( layer, Sequence ):
				for i in layer:
					self._broadcast_logits( i )
				return

			if self._logit_layers[ layer ]:
				return
			i = 1

			if not self._logit_layers[ layer - 1 ] and layer > 1:
				self._broadcast_logits( layer - 1 )

			while (~self._token_layers[ :layer ]).any():
				self._sample_tokens( layer - i )
				i += 1
				assert i < layer

			assert self._new_lower < self._new_upper

			# While we could process in parallel here in some cases, an increase
			# in batch size would be roughly equivalent, and easier to implement
			# so long as we're not memory constrained. Something to consider
			# if the GPU isn't saturated but VRAM is.
			upper = pymin( self._new_upper, layer )

			tm = self.thought_mask[ self._new_lower:upper, : ]

			if isinstance( self._params.past_key_values, StaticCache ):
				cache_position = self.flatten_cache_position( t.arange( self._new_lower, upper ) )
			else:
				cache_position = None

			outputs = self._model.lm_model(
				input_ids = rearrange( "b... d l -> (b...) (d l)", self[ self._new_lower:upper ] ),
				attention_mask = tm,
				# position_ids: Optional[ LongTensor ] = None
				past_key_values = self._params.past_key_values,
				# inputs_embeds: Optional[ FloatTensor ] = None
				# labels: Optional[ LongTensor ] = None
				use_cache = self._params.use_cache,
				# output_attentions: Optional[ bool ] = None
				output_hidden_states = True,
				return_dict = True,
				cache_position = cache_position,
				num_logits_to_keep = self.l
			)

			self._new_lower = upper
			self._params.past_key_values = outputs.past_key_values
			self.hidden_states[ layer ] = rearrange(
				"(b n) l e -> b n l e", outputs.hidden_states[ -1 ], l = self.l, n = self.n )
			self._logit_layers[ layer ] = True

			log = outputs.logits
			log[ ..., self._model.start_thought_token ] = -t.inf
			log[ ..., self._model.end_thought_token ] = -t.inf
			self.logits[ layer ] = rearrange( "(b n) l v -> b n l v", outputs.logits, l = self.l, n = self.n )

		@t.no_grad()
		def _sample_tokens( self, layer ):
			if isinstance( layer, slice ):
				for i in range(
						layer.start if layer.start is not None else 0, layer.stop if layer.stop is not None else self.d,
						layer.step if layer.step is not None else 1 ):
					self._sample_tokens( i )
				return
			elif isinstance( layer, Sequence ):
				for i in layer:
					self._sample_tokens( i )
				return

			if self._token_layers[ layer ]:
				return
			if not self._logit_layers[ layer ]:
				self._broadcast_logits( layer )

			logits = self.logits[ layer ]

			if self._model.thought_temperature == 0.0:
				toks = logits.argmax( dim = -1 )
			else:
				toks = gumbel_softmax( logits, tau = self._model.thought_temperature, dim = -1, hard = True ).argmax(
					dim = -1 )

			self[ layer ] = toks
			self._new_upper = layer + 1

		def flatten_cache_position( self, cache_layers: Tensor ):
			return (t.arange( self.l ) + self.l * cache_layers.reshape( -1, 1 )).reshape( -1 )

		def _get( self, table, gen, layer ):
			gen( layer )
			dim = self._layer_dim % table.ndim
			idx = [ slice( None, None ) ] * table.ndim
			idx[ dim ] = layer
			return table[ idx ]

		def _get_logits( self, idx ):
			return self._get( self._logits, self._broadcast_logits, idx )

		def _get_tokens( self, idx ):
			return self._get( self._raw_tensor, self._sample_tokens, idx )

		def __getitem__( self, idx ):
			return self._get_tokens( idx )

		class _LogitGetter:
			def __init__( self, ts ):
				self._ts = ts

			def __getitem__( self, idx ):
				return self._ts._get_logits( idx )

			def __setitem__( self, idx, value ):
				idx_list = [ slice( None, None ) ] * self._ts._logits.ndim
				if idx != 0:
					self._ts._sample_tokens( idx - 1 )  # ensure we're in a valid state
				idx_list[ self._ts._layer_dim % self._ts._logits.ndim ] = idx
				self._ts._logits[ tuple( idx_list ) ] = value
				self._ts._logit_layers[ idx ] = True

		@property
		def logits( self ):
			return self._logit_getter

		def __setitem__( self, idx, value ):
			idx_list = [ slice( None, None ) ] * self._raw_tensor.ndim
			if idx != 0:
				self._sample_tokens( idx - 1 )  # ensure we're in a valid state
			idx_list[ self._layer_dim % self._raw_tensor.ndim ] = idx
			self._raw_tensor[ tuple( idx_list ) ] = value
			self._token_layers[ idx ] = True
			if self._new_upper < idx + 1:
				self._new_upper = idx + 1
			else:
				warnings.warn( "setting an already set layer breaks the cache and is undefined behavior" )

	def generate_thoughts( self, inputs: LongTensor, params: ForwardParams ):
		thought_state = rearrange(
			"b... l -> b... n d l", t.empty_like( inputs ), n = self.n_thoughts,
			d = self.thought_depth + 3 + self.look_ahead )
		thought_state[ ..., 0, : ] = inputs
		thought_state[ ..., 1, : ] = self.start_thought_token

		thought_state = self.broadcast_tokens( thought_state, params ).with_new_layers( 0, 1 )[ 2 ]

		for i in range( 2, self.thought_depth + 2 ):
			thought_state[ ..., i + 1, : ] = self.broadcast_tokens( thought_state, params ).with_new_layers( i )[
				i + 1 ]

	Loss = Tuple[ str, Tensor, Tensor ]
	Metric = Tuple[ str, Tensor ]
	Losses = Sequence[ Loss ]
	Metrics = Sequence[ Metric ]

	def _loss_callback(
			self, *, thought_state, targets_loss, targets_thought, pl, ph, ql, qh, alpha, mask, **kwargs
	) -> None | Tuple[ Losses, Metrics ]:
		pass

	def calculate_loss( self, pl, ph, thought_state, labels, mask, params ):
		d, l = thought_state.shape[ -2 ], thought_state.shape[ -1 ]
		# sliding_labels = rearrange(
		# 	"b... l s -> b... n s l", labels[ ..., 1: ].unfold( -1, l, 1 ),
		# 	n = self.n_thoughts )
		sliding_labels = rearrange(
			"b... d l -> b... n d l", labels[ ..., 1: ].unfold( -1, l, 1 ), n = self.n_thoughts )
		# sliding_mask = rearrange(
		# 	"b... l s -> b... n s l", mask[ ..., 1: ].unfold( -1, l, 1 ),
		# 	n = self.n_thoughts )
		sliding_mask = rearrange( "b... d l -> b... n d l", mask[ ..., 1: ].unfold( -1, l, 1 ), n = self.n_thoughts )
		pl = rearrange(
			"b... d v l -> b... n d l v", pl[ ..., 1:, : ].unfold( -2, l, 1 ),
			n = self.n_thoughts )
		ph = rearrange(
			"b... d e l -> b... n d l e", ph[ ..., 1:, : ].unfold( -2, l, 1 ),
			n = self.n_thoughts )

		set_at( "b... n [d] l, s, b... n s l", thought_state, t.arange( self.look_ahead ), sliding_labels )
		# Repoint at the slice to allow garbage collection of redundant tensors
		del sliding_labels
		targets_loss = thought_state[ ..., -self.look_ahead:, : ]  # b n d l

		thought_mask = self.prepare_thought_mask( d, l, sliding_mask )
		# thought_mask = thought_mask.to( t.bool ) & rearrange(
		# 	"b... n d D l L, b... n s l -> b... n d (D + s) l L", lambda s: t.ones( s ), sliding_mask, d = d, D = d,
		# 	l = l,
		# 	L = l ).to( t.bool )

		ql, qh = self.broadcast_logits(
			thought_state, mask[ ..., :-self.look_ahead ], params, num_layers_to_keep = thought_state.shape[ -2 ] )

		alpha = self.mixer_head( ph, qh[ ..., -self.look_ahead:, : ] )

		final_logits = alpha * ql[ ..., -self.look_ahead:, : ] + (1 - alpha) * pl

		logits_loss = final_logits[ ..., -self.look_ahead:, :, : ]  # b n d l v
		logits_thought = ql[ ..., 2:self.thought_depth + 2, :, : ]  # b n d l v
		targets_thought = thought_state[ ..., 2:self.thought_depth + 2, : ]  # b n d l

		base_loss = self.compute_cross_entropy_loss( logits_loss, targets_loss, mask = sliding_mask )
		policy_loss, policy_metrics = self.compute_policy_loss(
			base_loss, logits_thought, targets_thought, temperature = self.reinforce_temperature )

		losses = (("base_loss", base_loss, self.base_loss_beta), ("policy_loss", policy_loss, self.policy_loss_beta))

		callback_result = self._loss_callback(
			thought_state = thought_state,
			targets_loss = targets_loss,
			targets_thought = targets_thought,
			pl = pl,
			ph = ph,
			ql = ql,
			qh = qh,
			alpha = alpha,
			mask = sliding_mask )

		if callback_result is not None:
			additional_losses, additional_metrics = callback_result
		else:
			additional_losses, additional_metrics = (), ()

		if callback_result is not None:
			losses = tuple( losses ) + tuple( additional_losses )

		aggregate_loss = None
		for _, loss, beta in losses:
			if aggregate_loss is None:
				aggregate_loss = beta * loss
			else:
				aggregate_loss += beta * loss

		if self.logger is not None:
			with t.no_grad():
				thoughtful_loss = self.compute_cross_entropy_loss(
					ql[ ..., -self.look_ahead:, :, : ], targets_loss, mask = sliding_mask )

				metrics_to_log = (
					("thoughtful_loss", thoughtful_loss),
					("alpha", alpha),
				)

				if policy_metrics is not None:
					metrics_to_log += policy_metrics

				metrics_to_log += additional_metrics

				metrics_to_log += tuple( (name, loss) for name, loss, _ in losses )

				log_dict = OrderedDict()
				for name, metric in metrics_to_log:
					log_dict[ f"{name}_avg" ] = metric.mean()
					log_dict[ f"{name}_max" ] = metric.max()
					log_dict[ f"{name}_min" ] = metric.min()
				log_dict[ "aggregate_loss" ] = aggregate_loss

				self.logger( log_dict, prog_bar = True )

		return aggregate_loss

	@staticmethod
	def compute_cross_entropy_loss( logits, targets, *, mask = None, temperature = 0.0 ):
		if temperature != 0.0:
			logits = logits / temperature

		# loss = vmap( "... [l v], ... [l] -> ... [l]", logits, targets, op = cross_entropy, kwargs = { "reduction": "none" } )
		l, tg = rearrange( "... l v, ... l -> (... l) v, (... l)", logits, targets )
		ls = cross_entropy( l, tg, reduction = "none" )
		loss = rearrange( "(B... l) -> B... l", ls, **solve( "B... l", targets ) )

		if mask is not None:
			loss.masked_fill_( ~(mask.to( t.bool )), t.nan )

		return reduce( "... d l -> ... l", loss, op = t.nanmean )

	@classmethod
	def compute_policy_loss( cls, base_loss, logits_thought, targets_thought, *, temperature = 0.0 ):
		with t.no_grad():
			r_mean = -reduce( "b... n l -> b... 1 l", base_loss, op = t.nanmean )
			reward = relu( -base_loss - r_mean )

		policy_loss = reward * cls.compute_cross_entropy_loss(
			logits_thought, targets_thought,
			temperature = temperature )

		return policy_loss, (("r_mean", r_mean), ("reward", reward))

	# TODO: Replace with a better tensor proxy
	class ForgetfulDict:
		def __init__( self, allowed_keys, default_value = None ):
			self.allowed_keys = allowed_keys
			self.default_value = default_value
			self.dict = { }

		def __getitem__( self, key ):
			if key in self.dict:
				return self.dict[ key ]
			else:
				return self.default_value

		def __setitem__( self, key, value ):
			if key in self.allowed_keys:
				self.dict[ key ] = value

	@compile_on_first_call
	def training_forward( self, params: ForwardParams ):
		ts = self.ThoughtState(
			self,
			self.thought_depth + 3 + self.look_ahead,
			self.n_thoughts,
			params.input_ids.dtype,
			self.dtype,
			self.dtype,
			lambda *args: self.ForgetfulDict(
				{ 1 } | { i for i in range( self.thought_depth + 3, self.thought_depth + 3 + self.look_ahead ) },
			),
			params )
		ts[ 0 ] = rearrange( "b... l -> b... n l", params.input_ids, n = self.n_thoughts )
		ts[ 1 ] = self.start_thought_token
		ts[ self.thought_depth + 2 ] = self.end_thought_token
		lab_len = ts.l - self.look_ahead
		labels = params.labels[ ..., 1: ].unfold( -1, lab_len, 1 )
		pl = ts.logits[ 1 ]
		ph = ts.hidden_states[ 1 ]
		ql, qh = [ ], [ ]
		padded_labels = pad( labels, (0, self.look_ahead) )
		for i in range( self.look_ahead ):
			j = i + self.thought_depth + 3
			ql.append( ts.logits[ j ] )
			qh.append( ts.hidden_states[ j ] )
			ts[ j ] = padded_labels[ ..., i, : ]

		ql = rearrange( "d b... l v-> b... d l v", t.stack( ql ) )[ ..., :-self.look_ahead, : ]
		qh = rearrange( "d b... l e -> b... d l e", t.stack( qh ) )[ ..., :-self.look_ahead, : ]

		pl = rearrange( "b... d v l -> b... d l v", pl[ ..., 1:, : ].unfold( -2, lab_len, 1 ) )
		ph = rearrange( "b... d e l -> b... d l e", ph[ ..., 1:, : ].unfold( -2, lab_len, 1 ) )

		alpha = self.mixer_head( ph, qh )

		logits_loss = ql * alpha + pl * (1 - alpha)

		sliding_mask = rearrange(
			"b... d l -> b... n d l", params.attention_mask[ ..., 1: ].unfold( -1, lab_len, 1 ), n = self.n_thoughts )

		targets_loss = rearrange( "b... d l -> b... n d l", labels, n = self.n_thoughts )
		base_loss = self.compute_cross_entropy_loss( logits_loss, targets_loss, mask = sliding_mask )
		logits_thought = ts.logits[ 2:self.thought_depth + 2 ]
		targets_thought = ts[ 2:self.thought_depth + 2 ]
		policy_loss, policy_metrics = self.compute_policy_loss(
			base_loss, logits_thought[ ..., :-self.look_ahead, : ], targets_thought[ ..., :-self.look_ahead ],
			temperature = self.reinforce_temperature )

		losses = (("base_loss", base_loss, self.base_loss_beta), ("policy_loss", policy_loss, self.policy_loss_beta))

		callback_result = self._loss_callback(
			thought_state = ts,
			targets_loss = targets_loss,
			targets_thought = targets_thought,
			pl = pl,
			ph = ph,
			ql = ql,
			qh = qh,
			alpha = alpha,
			mask = sliding_mask )

		if callback_result is not None:
			additional_losses, additional_metrics = callback_result
		else:
			additional_losses, additional_metrics = (), ()

		if callback_result is not None:
			losses = tuple( losses ) + tuple( additional_losses )

		aggregate_loss = None
		for _, loss, beta in losses:
			if aggregate_loss is None:
				aggregate_loss = beta * loss
			else:
				aggregate_loss += beta * loss

		aggregate_loss = aggregate_loss.mean()

		if self.logger is not None:
			with t.no_grad():
				thoughtful_loss = self.compute_cross_entropy_loss(
					ql[ ..., -self.look_ahead:, :, : ], targets_loss, mask = sliding_mask )

				metrics_to_log = (
					("thoughtful_loss", thoughtful_loss),
					("alpha", alpha),
				)

				if policy_metrics is not None:
					metrics_to_log += policy_metrics

				metrics_to_log += additional_metrics

				metrics_to_log += tuple( (name, loss) for name, loss, _ in losses )

				log_dict = OrderedDict()
				for name, metric in metrics_to_log:
					log_dict[ f"{name}_avg" ] = metric.mean()
				log_dict[ f"{name}_max" ] = metric.max()
				log_dict[ f"{name}_min" ] = metric.min()
				log_dict[ "aggregate_loss" ] = aggregate_loss

				self.logger( log_dict, prog_bar = True )

		return CausalLMOutputWithPast(
			loss = aggregate_loss,
			logits = logits_loss[ ..., -1, :, : ],
			past_key_values = params.past_key_values
		)

	# if params.num_logits_to_keep is None:
	# 	thought_state = self.generate_thoughts(
	# 		params.input_ids[ ..., :-self.look_ahead ], params.attention_mask[ ..., :-self.look_ahead ],
	# 		params )
	# else:
	# 	thought_state = self.generate_thoughts( params.input_ids, params.attention_mask, params )
	#
	# pl, ph = self.broadcast_logits(
	# 	rearrange( "b... l -> b... 1 l", params.input_ids ), params.attention_mask, params )
	#
	# if params.num_logits_to_keep is None:
	# 	base_loss = self.calculate_loss( pl, ph, thought_state, params.labels, params.attention_mask, params )
	# 	logits = None
	# else:
	# 	base_loss = self.calculate_loss(
	# 		pl, ph, thought_state[ ..., :-self.look_ahead ], params.labels, params.attention_mask, params )
	#
	# 	ql, qh = self.broadcast_logits(
	# 		thought_state[ ..., :self.thought_depth, : ], params.attention_mask, params )
	#
	# 	alpha = self.mixer_head( ph, qh )
	# 	logits = alpha * ql + (1 - alpha) * pl
	#
	# return CausalLMOutputWithPast(
	# 	loss = base_loss,
	# 	logits = logits,
	# 	past_key_values = params.past_key_values
	# )

	@inherit_device_and_dtype
	def forward( self, *args, **kwargs ) -> Union[ Tuple, CausalLMOutputWithPast ]:
		params = self.ForwardParams( *args, **kwargs )
		if params.labels is not None:
			if params.use_cache:
				# Caching doesn't work with
				# our training method
				params.use_cache = False
				params.past_key_values = None
				params.cache_position = None
			return self.training_forward( params )
		else:
			return self.inference_forward( params )

	def _prepare_cache_for_generation(
			self, generation_config: GenerationConfig, model_kwargs: Dict, assistant_model: "PreTrainedModel",
			batch_size: int, max_cache_length: int, device: t.device ) -> bool:
		return super()._prepare_cache_for_generation(
			generation_config, model_kwargs, assistant_model, batch_size, max_cache_length + self.thought_depth + 2,
			device )