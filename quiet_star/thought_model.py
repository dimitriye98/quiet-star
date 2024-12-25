import dataclasses
import inspect
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Sequence, Dict, Any

import torch as t
import wrapt
from dataclasses import dataclass
from toolz import memoize

pymin = min
pymax = max
from einx.op import *
from torch import Tensor, LongTensor, FloatTensor
from torch.nn.functional import gumbel_softmax, relu, cross_entropy, binary_cross_entropy
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
	"mixer_head_hidden_layers": 1,
	"confidence_head_activation": "relu",
	"confidence_head_hidden_layers": 3,
	"confidence_loss_beta": 1e4,
	"confidence_head_output_activation": "sigmoid",
	"confidence_parameter": 0.0,
}


class ThoughtModelConfig( PretrainedConfig, ABC ):

	def __init__( self, **kwargs ):
		super().__init__( **kwargs )
		for k, v in _config_params.items():
			val = kwargs.pop( k, v )
			setattr( self, k, val )
		self.mixer_head_hidden_size = kwargs.pop( "mixer_head_hidden_size", self.hidden_size )
		self.confidence_head_hidden_size = kwargs.pop( "confidence_head_hidden_size", self.hidden_size )
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


class ConfidenceHead( t.nn.Module ):
	def __init__( self, config ):
		super().__init__()
		hs, ms, nl = config.hidden_size, config.confidence_head_hidden_size, config.confidence_head_hidden_layers
		input_layer = [ t.nn.Linear( 2 * hs + 1, ms ) ]
		mid_layer = [ t.nn.Linear( ms, ms ), ACT2FN[ config.confidence_head_activation ] ] * nl
		output_layer = [ ACT2FN[ config.confidence_head_activation ], t.nn.Linear( ms, 1 ),
			ACT2FN[ config.confidence_head_output_activation ] ]

		self.mlp = t.nn.Sequential( *input_layer, *mid_layer, *output_layer )

	def forward( self, pre_thought_hidden_state, post_thought_hidden_state, mixer_value ):
		catted_states = t.cat( (pre_thought_hidden_state, post_thought_hidden_state, mixer_value), dim = -1 )
		return self.mlp( catted_states )


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


@dataclass
class ThoughtfulLMOutputWithPast( CausalLMOutputWithPast ):
	thoughts: Tensor = None
	logits_without_confidence: Tensor = None


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
		self.confidence_head = ConfidenceHead( config )
		self.confidence_loss_beta = config.confidence_loss_beta
		self.confidence_parameter = config.confidence_parameter
		self.loss_fns = [
			self.calculate_base_loss,
			self.calculate_policy_loss,
			self.calculate_confidence_loss
		]

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
		num_logits_to_keep: Optional[ int ] = None
		thought_mask: Optional[ Tensor ] = None
		confidence_parameter: Optional[ float ] = None
		cached_thoughts: Optional[ Tensor ] = None
		comparison_mode: bool = False

		def as_kwargs( self ):
			return { f.name: getattr( self, f.name ) for f in dataclasses.fields( self ) }

	def prepare_inputs_for_generation(
			self,
			input_ids: t.LongTensor,
			past_key_values: Optional[ Cache ] = None,
			attention_mask: Optional[ LongTensor ] = None,
			inputs_embeds: Optional[ FloatTensor ] = None,
			cache_position: Optional[ LongTensor ] = None,
			num_logits_to_keep: int = 1,
			confidence_parameter: Optional[ float ] = None,
			comparison_mode: bool = False,
			**kwargs,
	):
		return super().prepare_inputs_for_generation(
			input_ids,
			past_key_values,
			attention_mask,
			inputs_embeds,
			cache_position,
			num_logits_to_keep = num_logits_to_keep,
			confidence_parameter = confidence_parameter,
			comparison_mode = comparison_mode,
			**kwargs )

	def _update_model_kwargs_for_generation(
			self,
			outputs: ThoughtfulLMOutputWithPast,
			model_kwargs: Dict[ str, Any ],
			is_encoder_decoder: bool = False,
			num_new_tokens: int = 1,
	) -> Dict[ str, Any ]:
		ret = super()._update_model_kwargs_for_generation(
			outputs, model_kwargs, is_encoder_decoder, num_new_tokens )
		ret[ "cached_thoughts" ] = outputs.thoughts
		return ret

	@t.inference_mode()
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
				cache_pos = t.arange(
					past_seen_tokens, past_seen_tokens + input_ids.shape[ -1 ]
				)
			if position_ids is None:
				position_ids = cache_pos.unsqueeze( 0 )

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
			context[ ..., offset + self.thought_depth + 1 ] = self.end_thought_token

			init_cache_len = params.past_key_values.get_seq_length() if params.past_key_values is not None else None

			# Get the logits and the hidden state so we can let the outputs deallocate
			pl, ph = outputs.logits, outputs.hidden_states[ -1 ][ ..., -1:, : ].clone()

			def update_cache_pos( p ):
				if p is not None:
					return p[ ..., -1: ] + 1

			cache_pos = update_cache_pos( cache_pos )

			if params.confidence_parameter is None:
				params.confidence_parameter = self.confidence_parameter

			confidence = None
			if params.cached_thoughts is not None and params.confidence_parameter > 0:
				context[ ..., offset + 1: offset + self.thought_depth + 1 ] = params.cached_thoughts

				test_pass_cache_pos = cache_pos[ ..., -1: ] + t.arange( 0, self.thought_depth + 2 )

				# Do a pass through the model to get the logits and hidden states with the cached thought
				outputs = self.lm_model(
					input_ids = context[ ..., 1:offset + self.thought_depth + 2 ],
					position_ids = context_position_ids[ ..., 1:offset + self.thought_depth + 2 ],
					attention_mask = context_mask[ ..., :mask_offset + self.thought_depth + 2 ],
					past_key_values = params.past_key_values,
					use_cache = params.use_cache,
					output_attentions = False,
					output_hidden_states = True,
					return_dict = True,
					cache_position = test_pass_cache_pos,
					num_logits_to_keep = 1,
				)
				qcl, qch = outputs.logits, outputs.hidden_states[ -1 ][ ..., -1:, : ]

				pre_mix = self.mixer_head( ph, qch )
				confidence = self.confidence_head( ph, qch, pre_mix ).ge( params.confidence_parameter ).to( self.dtype )
				assert confidence.shape[ -1 ] == 1

				# if confidence.all() and not params.comparison_mode:
				# 	# We're done (really only relevant for unbatched inference, but who knows)
				# 	# Crop the cache to initial length, and return outputs
				# 	crop_cache( params.past_key_values, init_cache_len )
				# 	return ThoughtfulLMOutputWithPast(
				# 		logits = pl,
				# 		past_key_values = outputs.past_key_values,
				# 		thoughts = params.cached_thoughts
				# 	)

				if not params.comparison_mode:
					raise NotImplementedError( "Only comparison mode is currently supported" )

				qcl *= pre_mix
				confident_logits = qcl + pl * (1 - pre_mix)

				# Crop the cache for regeneration
				if isinstance( params.past_key_values, DynamicCache ) or isinstance(
						params.past_key_values, StaticCache ):
					crop_cache( params.past_key_values, init_cache_len )
				else:
					warnings.warn(
						f"Cache is of unsupported type {type( params.past_key_values )} and can't be cropped, falling back to invalidating it instead" )
					params.past_key_values = None

			def sample( logits ):
				logits[ ..., self.start_thought_token ] = -t.inf
				logits[ ..., self.end_thought_token ] = -t.inf
				if self.thought_temperature == 0.0:
					tok = logits[ ..., 0, : ].argmax( dim = -1 )
				else:
					tok = gumbel_softmax(
						logits[ ..., 0, : ],
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
					attention_mask = context_mask[ ..., :mask_offset + i ],
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

			i += 2

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
				attention_mask = context_mask[ ..., :mask_offset + i ],
				past_key_values = cache,
				use_cache = params.use_cache,
				output_attentions = False,
				output_hidden_states = True,
				return_dict = True,
				cache_position = cache_pos,
				num_logits_to_keep = 1,
			)

			ql, qh = outputs.logits, outputs.hidden_states[ -1 ][ ..., -1:, : ]

			alpha = self.mixer_head( ph, qh )

			# Extremely memory intensive if we do this out of place,
			# python isn't smart enough to optimize this on its own
			# logits = pl * (1 - alpha) + ql * alpha
			pl *= (1 - alpha)
			ql *= alpha
			pl += ql
			logits = pl
			logits_without_confidence = logits

			if isinstance( params.past_key_values, DynamicCache ) or isinstance( params.past_key_values, StaticCache ):
				crop_cache( params.past_key_values, init_cache_len )
			else:
				warnings.warn(
					f"Cache is of unsupported type {type( params.past_key_values )} and can't be cropped, falling back to invalidating it instead" )
				params.past_key_values = None

			if confidence is not None:
				# Reincorporate the confident logits
				# Remember that we discretized confidence to 0 or 1
				logits *= 1 - confidence
				confident_logits *= confidence
				logits += confident_logits

			if params.return_dict:
				return ThoughtfulLMOutputWithPast(
					logits = logits,
					logits_without_confidence = logits_without_confidence,
					past_key_values = outputs.past_key_values,
					thoughts = context[ ..., -self.thought_depth - 1:-1 ]
				)
			else:
				return (logits,) + ((pkv,) if (pkv := outputs.past_key_values) is not None else tuple())

	class ThoughtMask:
		def __init__( self, l: int, d: int, k: int, padding_mask: Tensor ):
			self._k = k
			self._d = d
			self._l = l
			self._thought_mask = self.prepare_thought_mask( d, l, padding_mask )

		def let( self, *, l: Optional[ int ] = None, d: Optional[ int ] = None, k: Optional[ int ] = None, padding_mask: Tensor):
			return self.__class__( l or self._l, d or self._d, k or self._k, padding_mask )

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
			# k = tuple() if k is None else (k,)
			return rearrange( "b... d D l L -> (b...) k (d l) (D L)", thought_mask, k = k )

		@staticmethod
		def invert_mask( mask: Tensor ):
			return t.zeros_like( mask ).masked_fill( mask == 0, t.finfo( mask.dtype ).min )

	Loss = Tuple[ str, Tensor, Tensor ]
	Metric = Tuple[ str, Tensor ]
	Losses = Sequence[ Loss ]
	Metrics = Sequence[ Metric ]

	def _loss_callback(
			self, *, thought_state, targets_loss, targets_thought, pl, ph, ql, qh, alpha, mask, **kwargs
	) -> None | Tuple[ Losses, Metrics ]:
		pass

	@staticmethod
	def compute_cross_entropy_loss( logits, targets, *, mask = None, temperature = 0.0, loss_fn = cross_entropy ):
		if temperature != 0.0:
			logits = logits / temperature

		# loss = vmap( "... [l v], ... [l] -> ... [l]", logits, targets, op = cross_entropy, kwargs = { "reduction": "none" } )
		l, tg = rearrange( "... l v, ... l -> (... l) v, (... l)", logits, targets )
		ls = loss_fn( l, tg, reduction = "none" )
		loss = rearrange( "(B... l) -> B... l", ls, **solve( "B... l", targets ) )

		if mask is not None:
			loss = loss.masked_fill( ~(mask.to( t.bool )), t.nan )

		return reduce( "... d l -> ... l", loss, op = t.nanmean )

	@classmethod
	def compute_policy_loss( cls, base_loss, logits_thought, targets_thought, *, mask = None, temperature = 0.0 ):
		with t.no_grad():
			r_mean = -reduce( "b... n l -> b... 1 l", base_loss, op = t.nanmean )
			reward = relu( -base_loss - r_mean )

		policy_loss = reward * cls.compute_cross_entropy_loss(
			logits_thought, targets_thought,
			mask = mask,
			temperature = temperature )

		return policy_loss, (("r_mean", r_mean), ("reward", reward))

	# @compile_on_first_call
	# def training_forward( self, params: ForwardParams ):
	# 	ts = self.ThoughtState(
	# 		self,
	# 		self.thought_depth + 3 + self.look_ahead,
	# 		self.n_thoughts,
	# 		params.input_ids.dtype,
	# 		self.dtype,
	# 		self.dtype,
	# 		lambda *args: self.ForgetfulDict(
	# 			{ 1 } | { i for i in range( self.thought_depth + 3, self.thought_depth + 3 + self.look_ahead ) },
	# 		),
	# 		params )
	# 	ts[ 0 ] = rearrange( "b... l -> b... n l", params.input_ids, n = self.n_thoughts )
	# 	ts[ 1 ] = self.start_thought_token
	# 	ts[ self.thought_depth + 2 ] = self.end_thought_token
	# 	lab_len = ts.l - self.look_ahead
	# 	labels = params.labels[ ..., 1: ].unfold( -1, lab_len, 1 )
	# 	pl = ts.logits[ 1 ]
	# 	ph = ts.hidden_states[ 1 ]
	# 	ql, qh = [ ], [ ]
	# 	padded_labels = pad( labels, (0, self.look_ahead) )
	# 	for i in range( self.look_ahead ):
	# 		j = i + self.thought_depth + 3
	# 		ql.append( ts.logits[ j ] )
	# 		qh.append( ts.hidden_states[ j ] )
	# 		ts[ j ] = padded_labels[ ..., i, : ]
	#
	# 	ql = rearrange( "d b... l v-> b... d l v", t.stack( ql ) )[ ..., :-self.look_ahead, : ]
	# 	qh = rearrange( "d b... l e -> b... d l e", t.stack( qh ) )[ ..., :-self.look_ahead, : ]
	#
	# 	pl = rearrange( "b... d v l -> b... d l v", pl[ ..., 1:, : ].unfold( -2, lab_len, 1 ) )
	# 	ph = rearrange( "b... d e l -> b... d l e", ph[ ..., 1:, : ].unfold( -2, lab_len, 1 ) )
	#
	# 	alpha = self.mixer_head( ph, qh )
	#
	# 	logits_loss = ql * alpha + pl * (1 - alpha)
	#
	# 	sliding_mask = rearrange(
	# 		"b... d l -> b... n d l", params.attention_mask[ ..., 1: ].unfold( -1, lab_len, 1 ), n = self.n_thoughts )
	#
	# 	targets_loss = rearrange( "b... d l -> b... n d l", labels, n = self.n_thoughts )
	# 	base_loss = self.compute_cross_entropy_loss( logits_loss, targets_loss, mask = sliding_mask )
	# 	logits_thought = ts.logits[ 2:self.thought_depth + 2 ]
	# 	targets_thought = ts[ 2:self.thought_depth + 2 ]
	# 	policy_loss, policy_metrics = self.compute_policy_loss(
	# 		base_loss, logits_thought[ ..., :-self.look_ahead, : ], targets_thought[ ..., :-self.look_ahead ],
	# 		temperature = self.reinforce_temperature )
	#
	# 	del ql, qh
	# 	ts.mutate_for_confidence( self.thought_depth + 2 )
	# 	qch = [ ], [ ]
	# 	for i in range( self.look_ahead ):
	# 		j = i + self.thought_depth + 3
	# 		ts.logits[ j ]
	# 		qch.append( ts.hidden_states[ j ] )
	# 		ts[ j ] = padded_labels[ ..., i, : ]
	#
	# 	false_alpha = self.mixer_head( ph, qch )
	# 	confidence = self.confidence_head( ph, qch, false_alpha )
	# 	sliding_mask[ ..., 0 ] = 0  # Mask out the thought from the first token
	# 	with t.no_grad():
	# 		# Intentionally break the gradients to enforce causality
	# 		targets_confidence = false_alpha.ge( alpha ).to( self.dtype )
	#
	# 	confidence_loss = binary_cross_entropy( confidence, targets_confidence, reduction = "none" )
	# 	confidence_loss.masked_fill_( ~(sliding_mask.to( t.bool )), t.nan )
	# 	confidence_loss.nanmean( dim = -1 )
	#
	# 	losses = (
	# 		("base_loss", base_loss, self.base_loss_beta),
	# 		("policy_loss", policy_loss, self.policy_loss_beta),
	# 		("confidence_loss", confidence_loss, self.confidence_loss_beta))
	#
	# 	# callback_result = self._loss_callback(
	# 	# 	thought_state = ts,
	# 	# 	targets_loss = targets_loss,
	# 	# 	targets_thought = targets_thought,
	# 	# 	pl = pl,
	# 	# 	ph = ph,
	# 	# 	ql = ql,
	# 	# 	qh = qh,
	# 	# 	alpha = alpha,
	# 	# 	mask = sliding_mask )
	# 	#
	# 	# if callback_result is not None:
	# 	# 	additional_losses, additional_metrics = callback_result
	# 	# else:
	# 	# 	additional_losses, additional_metrics = (), ()
	# 	#
	# 	# if callback_result is not None:
	# 	# 	losses = tuple( losses ) + tuple( additional_losses )
	#
	# 	aggregate_loss = None
	# 	for _, loss, beta in losses:
	# 		if aggregate_loss is None:
	# 			aggregate_loss = beta * loss
	# 		else:
	# 			aggregate_loss += beta * loss
	#
	# 	aggregate_loss = aggregate_loss.mean()
	#
	# 	if self.logger is not None:
	# 		with t.no_grad():
	# 			thoughtful_loss = self.compute_cross_entropy_loss(
	# 				ql[ ..., -self.look_ahead:, :, : ], targets_loss, mask = sliding_mask )
	#
	# 			metrics_to_log = (
	# 				("thoughtful_loss", thoughtful_loss),
	# 				("alpha", alpha),
	# 			)
	#
	# 			if policy_metrics is not None:
	# 				metrics_to_log += policy_metrics
	#
	# 			# metrics_to_log += additional_metrics
	#
	# 			metrics_to_log += tuple( (name, loss) for name, loss, _ in losses )
	#
	# 			log_dict = OrderedDict()
	# 			for name, metric in metrics_to_log:
	# 				log_dict[ f"{name}_avg" ] = metric.mean()
	# 			log_dict[ f"{name}_max" ] = metric.max()
	# 			log_dict[ f"{name}_min" ] = metric.min()
	# 			log_dict[ "aggregate_loss" ] = aggregate_loss
	#
	# 			self.logger( log_dict, prog_bar = True )
	#
	# 	return CausalLMOutputWithPast(
	# 		loss = aggregate_loss,
	# 		logits = logits_loss[ ..., -1, :, : ],
	# 		past_key_values = params.past_key_values
	# 	)

	@staticmethod
	def append_layer( acc, new_layer, dim ):
		if acc is None:
			return new_layer

		if not t.is_tensor( new_layer ):
			new_layer = t.tensor( new_layer ).expand_as( acc.select( dim, 0 ) )

		if new_layer.ndim < acc.ndim:
			new_layer = new_layer.unsqueeze( dim )
		return t.cat( (acc, new_layer), dim = dim )

	@staticmethod
	def cat_not_null( tensors, dim ):
		return t.cat( [ T for T in tensors if T is not None ], dim = dim )

	@staticmethod
	def prepare_cache_positions( seq_len, unseen_layers ):
		return t.arange( seq_len ) + seq_len * t.arange( unseen_layers[ 0 ], unseen_layers[ -1 ] + 1 ).unsqueeze( -1 )

	@staticmethod
	def prepare_position_ids( seq_len, unseen_layers ):
		pos_per_layer = t.arange( unseen_layers[ 0 ], unseen_layers[ -1 ] + seq_len )
		sliding_pos = pos_per_layer.unfold( -1, seq_len, 1 )
		return sliding_pos

	def broadcast_logits( self, ts, mask, cache, unseen_layers, layers_to_keep: Optional[ int ] = 1 ):
		B, d, l = solve( "b... d l", ts ).values()

		if layers_to_keep is None:
			layers_to_keep = d

		if cache is not None:
			cache_pos = rearrange( "d l -> (d l)", self.prepare_cache_positions( ts.shape[ -1 ], unseen_layers ) )
			a_mask = mask[ unseen_layers[ 0 ]:unseen_layers[ -1 ] + 1, : ]
		else:
			cache_pos = None
			a_mask = mask[ unseen_layers[ 0 ]:unseen_layers[ -1 ] + 1, unseen_layers[ 0 ]:unseen_layers[ -1 ] + 1 ]

		out = self.lm_model(
			input_ids = rearrange( "b... d l -> (b...) (d l)", ts[ ..., -len( unseen_layers ):, : ] ),
			position_ids = rearrange(
				"d l -> (b...) (d l)", self.prepare_position_ids( ts.shape[ -1 ], unseen_layers ), b = B ),
			attention_mask = a_mask,
			past_key_values = cache,
			use_cache = cache is not None,
			output_attentions = False,
			output_hidden_states = True,
			return_dict = True,
			cache_position = cache_pos,
			num_logits_to_keep = l * layers_to_keep,
		)

		out_log = rearrange( "(b...) (d l) v -> b... d l v", out.logits, b = B, d = layers_to_keep, l = l )
		out_hidden = rearrange(
			"(b...) (d l) e -> b... d l e", out.hidden_states[ -1 ][ ..., -l * layers_to_keep:, : ], b = B,
			d = layers_to_keep, l = l )

		return out_log, out_hidden

	def advance_thoughts( self, ts, logs, mask, cache, unseen_layers ):
		new_logs, _ = self.broadcast_logits( ts, mask, cache, unseen_layers )

		with t.no_grad():
			if self.thought_temperature == 0.0:
				tok = new_logs.argmax( dim = -1 )
			else:
				tok = gumbel_softmax(
					new_logs, tau = self.thought_temperature, hard = True ).argmax( dim = -1 )

		return self.cat_not_null( (ts, tok), dim = -2 ), self.cat_not_null( (logs, new_logs), dim = -3 )

	def calculate_loss( self, ts, cache, unseen_layers, labels, mask, padding_mask, logits_thought ):
		losses = OrderedDict()
		metrics = OrderedDict()
		misc_state = { }

		sliding_labels = labels.unfold( -1, ts.shape[ -1 ], 1 )
		sliding_mask = padding_mask.unfold( -1, ts.shape[ -1 ], 1 )

		naive_state = sliding_labels[ ..., :-1, : ]

		assert naive_state.shape[ -2 ] == self.look_ahead

		ts = t.cat( (ts, naive_state[ ..., 1:, : ]), dim = -2 )

		for fn in self.loss_fns:
			new_losses, new_metrics, new_state = fn(
				cache, mask, naive_state, sliding_labels, sliding_mask, ts, logits_thought, unseen_layers, losses,
				metrics, misc_state )

			losses.update( new_losses )
			metrics.update( new_metrics )
			misc_state.update( new_state )

		aggregate_loss = None
		for loss, beta in losses.values():
			if aggregate_loss is None:
				aggregate_loss = beta * loss
			else:
				aggregate_loss += beta * loss

		aggregate_loss = aggregate_loss.mean()

		if self.logger is not None:
			with t.no_grad():
				# thoughtful_loss = self.compute_cross_entropy_loss( ql[ ..., -self.look_ahead:, :, : ], targets_loss, mask = sliding_mask )

				for name, (loss, _) in losses.items():
					metrics[ name ] = loss

				log_dict = OrderedDict()
				for name, metric in metrics.items():
					log_dict[ f"{name}_avg" ] = metric.mean()
					log_dict[ f"{name}_max" ] = metric.max()
					log_dict[ f"{name}_min" ] = metric.min()

				log_dict[ "aggregate_loss" ] = aggregate_loss

				self.logger( log_dict, prog_bar = True )

		return aggregate_loss

	def calculate_confidence_loss(
			self, cache, mask, naive_state, sliding_labels, sliding_mask, ts, logits_thought, unseen_layers, losses,
			metrics, misc_state ):
		# Shift the thoughts one step forward
		cts = t.cat(
			(
				ts[ ..., :2, 1: ],
				ts[ ..., 2:2 + self.thought_depth, :-1 ],
				ts[ ..., 2 + self.thought_depth:, 1: ],
			), dim = -2 )

		# Keep cts the same shape as ts, to avoid recompiling the model
		cts = t.cat((t.zeros((*cts.shape[:-1], 1), dtype = cts.dtype), cts), dim = -1)

		# Mask out the inserted padding tokens
		new_mask = t.ones(cts.shape[-1], dtype= sliding_mask.dtype)
		new_mask[0] = 0
		padding_mask = sliding_mask[..., 0, :] * new_mask
		mask = mask.let( padding_mask = padding_mask )

		cql, cqh = self.broadcast_logits( cts, mask, None, t.arange( self.look_ahead ), self.look_ahead )
		# Confidence loss should not influence the mixer head
		with t.no_grad():
			calpha = self.mixer_head( misc_state[ "ph" ], cqh )
			confidence_target = calpha.ge( misc_state[ "alpha" ] )
		confidence = self.confidence_head( misc_state[ "ph" ], cqh, calpha )
		confidence_loss = binary_cross_entropy( confidence, confidence_target.to( confidence.dtype ), reduction = "none" )
		confidence_loss = reduce("b... [d] l 1 -> b... l", confidence_loss, op=t.nanmean)
		return {
			"confidence loss": (confidence_loss, self.confidence_loss_beta),
		}, {
			"confidence": confidence,
		}, { }

	def calculate_policy_loss(
			self, cache, mask, naive_state, sliding_labels, sliding_mask, ts, logits_thought, unseen_layers, losses,
			metrics, misc_state ):
		thoughts = ts[ ..., 2:2 + self.thought_depth, : ]

		policy_loss, policy_metrics = self.compute_policy_loss(
			losses[ "base loss" ][ 0 ], logits_thought, thoughts, mask = sliding_mask[ ..., :1, : ],
			temperature = self.reinforce_temperature )

		return {
			"policy loss": (policy_loss, self.policy_loss_beta),
		}, {
			k: v for k, v in policy_metrics
		}, { }

	def calculate_base_loss(
			self, cache, mask, naive_state, sliding_labels, sliding_mask, ts, logits_thought, unseen_layers, losses,
			metrics, misc_state ):
		targets_base, mask_base = sliding_labels[ ..., 1:, : ], sliding_mask[ ..., :-1, : ]

		pl, ph = self.broadcast_logits( naive_state, mask, None, t.arange( self.look_ahead ), self.look_ahead )

		unseen_layers = t.cat(
			(
				unseen_layers,
				unseen_layers[ -1 ] + 1 + t.arange( self.look_ahead - 1 )
			), dim = -1 )

		ql, qh = self.broadcast_logits( ts, mask, cache, unseen_layers, self.look_ahead )
		alpha = self.mixer_head( ph, qh )
		logits_loss = ql * alpha + pl * (1 - alpha)

		base_loss = self.compute_cross_entropy_loss(
			logits_loss, targets_base, mask = mask_base, temperature = self.reinforce_temperature )

		return {
			"base loss": (base_loss, self.base_loss_beta),
		}, {
			"alpha": alpha
		}, {
			"pl": pl,
			"ph": ph,
			"ql": ql,
			"qh": qh,
			"alpha": alpha
		}

	def training_forward( self, params ):
		if not isinstance( params.past_key_values, StaticCache ):
			raise ValueError( "Static cache required for training" )

		# Add an additional batch dimension for the number of thoughts
		# and truncate the input ids by look_ahead
		ts = rearrange( "b l -> b n 1 l", params.input_ids[ ..., :-self.look_ahead ], n = self.n_thoughts )
		padding_mask = rearrange( "b l -> b n l", params.attention_mask, n = self.n_thoughts )

		# Prepare the thought mask, truncating the padding mask by look_ahead
		mask = self.ThoughtMask(
			ts.shape[ -1 ],
			self.thought_depth + 2 + self.look_ahead,
			self.lm_model.config.num_attention_heads,
			padding_mask[ ..., :-self.look_ahead ] )

		# Append the start token
		ts = self.append_layer( ts, self.start_thought_token, dim = -2 )

		# Generate the thoughts
		logs = None
		unseen_layers = t.arange( 2 )
		cache = params.past_key_values
		for i in range( self.thought_depth ):
			ts, logs = self.advance_thoughts( ts, logs, mask, cache, unseen_layers )
			unseen_layers = unseen_layers[ -1: ] + 1

		# Append the end token
		ts = self.append_layer( ts, self.end_thought_token, dim = -2 )
		unseen_layers = t.cat( (unseen_layers, unseen_layers[ -1: ] + 1), dim = -1 )

		labels = rearrange( "b l -> b n l", params.labels, n = self.n_thoughts )

		return CausalLMOutputWithPast(
			loss = self.calculate_loss( ts, cache, unseen_layers, labels, mask, padding_mask, logs ),
		)

	def get_cache_size( self, batch_size, sequence_length ):
		return (sequence_length - self.look_ahead) * (
				self.thought_depth + 2 + self.look_ahead), batch_size * self.n_thoughts

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
