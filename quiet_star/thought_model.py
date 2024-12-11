from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import copy
from dataclasses import dataclass
from math import prod as lprod
from typing import List, Sequence

import torch as t
import wrapt
from einx.op import *
from torch import Tensor, LongTensor, FloatTensor
from torch.nn.functional import gumbel_softmax, relu, cross_entropy
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast


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
		return self.mlp( t.cat( (pre_thought_hidden_state, post_thought_hidden_state), dim = -1 ) )


class ThoughtModel( PreTrainedModel, GenerationMixin, ABC ):
	base_model_prefix = "_lm_model"

	def __init__( self, config ):
		super().__init__( config )
		self.thought_depth = config.thought_depth
		self.start_thought_token = config.start_thought_token
		self.end_thought_token = config.end_thought_token
		self.thought_temperature = config.thought_temperature
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
		input_ids: LongTensor
		attention_mask: Optional[ Tensor ]
		position_ids: Optional[ LongTensor ]
		past_key_values: Optional[ List[ FloatTensor ] ]
		inputs_embeds: Optional[ FloatTensor ]
		labels: Optional[ LongTensor ]
		use_cache: Optional[ bool ]
		output_attentions: Optional[ bool ]
		output_hidden_states: Optional[ bool ]
		return_dict: Optional[ bool ]
		cache_position: Optional[ LongTensor ]
		num_logits_to_keep: int = None

	@t.no_grad()
	def inference_forward( self, params: ForwardParams ):
		context = t.empty(
			params.input_ids.shape[ ..., :-1 ] + (self.thought_depth + 2,), dtype = params.input_ids.dtype )
		context[ ..., :params.input_ids.shape[ -1 ] ] = params.input_ids

		if params.attention_mask is None:
			padding_mask = t.ones_like( context )
		else:
			padding_mask = t.empty_like( context, dtype = params.attention_mask.dtype )
			padding_mask[ ..., :params.attention_mask.shape[ -1 ] ] = params.attention_mask
			padding_mask[ ..., params.attention_mask.shape[ -1 ]: ] = 1

		offset = params.input_ids.shape[ -1 ]
		context[ ..., offset ] = self.start_thought_token

		first_iter = True
		for i in range( self.thought_depth ):
			outputs = self.lm_model(
				context[ ..., :offset + i ],
				attention_mask = padding_mask[ ..., :offset + i ],

				past_key_values = params.past_key_values,
				use_cache = params.use_cache,
				cache_position = params.cache_position,

				return_dict = True,
				num_logits_to_keep = 1
			)

			if params.cache_position is not None:
				params.cache_position += 1
				if first_iter:
					params.cache_position += 1

			params.past_key_values = outputs.past_key_values
			logits = outputs.logits
			set_at( "... [v], [2], 1", logits, [ self.start_thought_token, self.end_thought_token ], -t.inf )

			if self.thought_temperature == 0.0:
				next_tokens = logits[ ..., 0 ].argmax( dim = -1 )
			else:
				next_tokens = gumbel_softmax(
					logits[ ..., 0 ], tau = self.thought_temperature, hard = True ).argmax( dim = -1 )

			context[ ..., offset + i + 1 ] = next_tokens

		context[ ..., offset + self.thought_depth + 1 ] = self.end_thought_token

		return self.lm_model( context, **params._asdict() )

	@t.no_grad()
	def prepare_thought_mask( self, d: int, l: int, padding_mask: Tensor ):
		causal_padding_mask = rearrange( "b... l -> b... L l", padding_mask, L = l ).tril()

		thought_thought = multiply( "d..., l... -> d... l...", t.full( (d - 1, d), 1 ).tril( 1 ), t.eye( l ) )
		ret = rearrange(
			"d D l L, b... l L -> b... (d + 1) D l L", causal_padding_mask, thought_thought, D = d, l = l, L = l )

		return ret

	def broadcast_logits(
			self, thought_state: LongTensor, mask: Tensor, params: ForwardParams, num_layers_to_keep: int = 1 ):
		d, l = thought_state.shape[ -2 ], thought_state.shape[ -1 ]
		if mask.shape == thought_state.shape:
			thought_mask = mask
		else:
			thought_mask = self.prepare_thought_mask( d, l, mask )
		thought_mask = t.zeros( thought_mask.shape ).masked_fill( ~(thought_mask.to( t.bool )), -t.inf )

		causal_mask = rearrange(
			"b... d... l... -> (b...) k (d l)...", thought_mask, d = (d, d), l = (l, l),
			k = self.lm_model.config.num_attention_heads )

		layer_shape = thought_state.shape[ :-2 ] + thought_state.shape[ -1: ]

		outputs: CausalLMOutputWithPast = self.lm_model(
			rearrange( "b... d l -> (b...) (d l)", thought_state, d = d, l = l ),
			attention_mask = causal_mask,

			past_key_values = params.past_key_values,
			use_cache = params.use_cache,
			cache_position = params.cache_position,

			return_dict = True,
			num_logits_to_keep = lprod( layer_shape ),
			output_hidden_states = True )

		params.past_key_values = outputs.past_key_values

		logits_out = rearrange(
			"(b...) (d l) v -> b... d l v", outputs.logits, d = num_layers_to_keep,
			**solve( "b... _ l", thought_mask ) )
		hidden_out = rearrange(
			"(b...) (d l) e -> b... d l e", outputs.hidden_states, d = num_layers_to_keep,
			**solve( "b... _ l", thought_mask ) )

		return logits_out, hidden_out

	@t.no_grad()
	def broadcast_tokens( self, thought_state: LongTensor, mask: Tensor, params: ForwardParams ):
		logits, _ = self.broadcast_logits( thought_state, mask, params )

		set_at( "... [v], [2], 1", logits, [ self.start_thought_token, self.end_thought_token ], -t.inf )

		if self.thought_temperature == 0.0:
			return logits.argmax( dim = -1 )
		else:
			return gumbel_softmax( logits, tau = self.thought_temperature, dim = -1, hard = True ).argmax( dim = -1 )

	@t.no_grad()
	def generate_thoughts( self, inputs: LongTensor, mask: Tensor, params: ForwardParams ):
		input_layer = rearrange( "b... l -> b... n 1 l", inputs, n = self.n_thoughts )
		mask = rearrange( "b... l -> b... n l", mask, n = self.n_thoughts )

		layer_shape = list( input_layer.shape )
		thought_state_shape = copy( layer_shape )
		thought_state_shape[ -2 ] = self.thought_depth + 3 + self.look_ahead

		thought_state = t.empty( thought_state_shape, dtype = t.long )
		set_at( "b... [d] l, [1], b... l", thought_state, 0, input_layer )
		set_at( "b... [d] l, [1], (b... l)", thought_state, 1, self.start_thought_token )
		for i in range( 2, self.thought_depth + 2 ):
			set_at(
				"b... [d] l, [1], b... l", thought_state, i,
				self.broadcast_tokens( thought_state[ ..., :i, : ], mask, params ) )
		set_at( "b... [d] l, [1], (b... l)", thought_state, self.thought_depth + 2, self.end_thought_token )

		return thought_state

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
		sliding_labels = rearrange(
			"b... l s -> b... n s l", labels[ ..., 1: ].unfold( dim = -1, size = l, step = 1 ),
			n = self.n_thoughts )
		sliding_mask = rearrange(
			"b... l s -> b... n s l", mask[ ..., 1: ].unfold( dim = -1, size = l, step = 1 ),
			n = self.n_thoughts )
		pl = rearrange(
			"b... l v s -> b... n s l v", pl[ ..., 1: ].unfold( dim = -2, size = l, step = 1 ),
			n = self.n_thoughts )
		pl = rearrange(
			"b... l e s -> b... n s l e", ph[ ..., 1: ].unfold( dim = -2, size = l, step = 1 ),
			n = self.n_thoughts )

		set_at( "b... n [d] l, s [1], b... n s l", thought_state, t.arange( self.look_ahead ), sliding_labels )
		# Repoint at the slice to allow garbage collection of redundant tensors
		del sliding_labels
		targets_loss = thought_state[ ..., -self.look_ahead:, : ]  # b n d l

		thought_mask = self.prepare_thought_mask( d, l, mask )
		thought_mask = thought_mask.to( t.bool ) & rearrange(
			"b... n d D l L, b... n s l -> b... n d (D + s) l L", lambda s: t.ones( s ), sliding_mask, d = d, D = d,
			l = l,
			L = l ).to( t.bool )

		ql, qh = self.broadcast_logits(
			thought_state, thought_mask[ ..., :-1, : ], params, num_layers_to_keep = self.look_ahead )

		alpha = self.mixer_head( ph, qh )

		final_logits = alpha * ql + (1 - alpha) * pl

		logits_loss = final_logits[ ..., -self.look_ahead:, :, : ]  # b n d l v
		logits_thought = final_logits[ ..., 2:self.thought_depth + 2, :, : ]  # b n d l v
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
				log_dict[ "aggergate_loss" ] = aggregate_loss

				self.logger.log_dict( log_dict, prog_bar = True )

		return aggregate_loss

	@staticmethod
	def compute_cross_entropy_loss( logits, targets, *, mask = None, temperature = 0.0 ):
		if temperature != 0.0:
			logits = logits / temperature

		loss = vmap( "... [l v], ... [l] -> ... l", logits, targets, cross_entropy, kwargs = { "reduction": "none" } )

		if mask is not None:
			loss.masked_fill_( ~(mask.to( t.bool )), t.nan )

		return reduce( "... l -> ...", loss, t.nanmean )

	@classmethod
	def compute_policy_loss( cls, base_loss, logits_thought, targets_thought, *, temperature = 0.0 ):
		with t.no_grad():
			r_mean = -reduce( "b... n l -> b... 1 l", base_loss, t.nanmean )
			reward = relu( -base_loss - r_mean )

		policy_loss = reward * cls.compute_cross_entropy_loss(
			logits_thought, targets_thought,
			temperature = temperature )

		return policy_loss, (("r_mean", r_mean), ("reward", reward))

	def training_forward( self, params: ForwardParams ) -> Union[ Tuple, CausalLMOutputWithPast ]:
		if params.num_logits_to_keep is None:
			thought_state = self.generate_thoughts(
				params.input_ids[ ..., :-self.look_ahead ], params.attention_mask[ ..., :-self.look_ahead ],
				params )
		else:
			thought_state = self.generate_thoughts( params.input_ids, params.attention_mask, params )

		pl, ph = self.broadcast_logits( thought_state[ ..., 0:1, : ], params.attention_mask, params )

		if params.num_logits_to_keep is None:
			base_loss = self.calculate_loss( pl, ph, thought_state, params.labels, params.attention_mask, params )
			logits = None
		else:
			base_loss = self.calculate_loss(
				pl, ph, thought_state[ ..., :-self.look_ahead ], params.labels, params.attention_mask, params )

			ql, qh = self.broadcast_logits(
				thought_state[ ..., :self.thought_depth, : ], params.attention_mask, params )

			alpha = self.mixer_head( ph, qh )
			logits = alpha * ql + (1 - alpha) * pl

		return CausalLMOutputWithPast(
			loss = base_loss,
			logits = logits,
			past_key_values = params.past_key_values
		)

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
