import logging
from abc import ABCMeta, abstractmethod
from collections.abc import MutableSequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from operator import methodcaller
from queue import SimpleQueue

import torch as t
from lightning_utilities import apply_to_collection

from . import futures as ft
from .training.data import uncurry

logger = logging.getLogger( __name__ )


class Broker( metaclass = ABCMeta ):
	@abstractmethod
	def __call__( self, obj, pin_memory = False ):
		pass


class TensorBroker( Broker ):
	def __init__( self, size = None ):
		if size is None:
			size = 16
		self.pool = ThreadPoolExecutor( max_workers = size )

	@t.inference_mode()
	def __send_to_cpu( self, tensor, pin_memory = False ):
		stream = t.cuda.Stream()
		with t.cuda.stream( stream ):
			receive = t.empty( tensor.shape, dtype = tensor.dtype, device = "cpu", pin_memory = True )
			receive.copy_( tensor )
			# We have to copy again, as the receiving tensor *must* be pinned
			# for the transfer to not block the GPU
			# however, holding pinned memory can be expensive,
			# so we only keep the data in pinned memory if instructed
			if not pin_memory:
				unpinned = t.empty( tensor.shape, dtype = tensor.dtype, device = "cpu" )
				unpinned.copy_( receive )
				return unpinned
			return receive

	def __call__( self, tensor, pin_memory = False ):
		ret = self.pool.submit( self.__send_to_cpu, tensor.contiguous(), pin_memory = pin_memory )
		return ret

	def __del__( self ):
		self.pool.shutdown()


class CollectionBroker( Broker ):
	def __init__( self, ):
		self._tensor_broker = TensorBroker()

	def __call__( self, tensors, pin_memory = False ):
		return ft.collect( apply_to_collection( tensors, t.Tensor, partial(self._tensor_broker, pin_memory = pin_memory) ) )


class BrokeredList( MutableSequence ):
	def __init__( self, size ):
		self._broker = CollectionBroker( size )
		self.futures = [ ]

	def __getitem__( self, index ):
		if isinstance( index, int ):
			return self.futures[ index ].result()
		else:
			return map( methodcaller( "result" ), self.futures[ index ] )

	def __setitem__( self, index, value ):
		if isinstance( index, int ):
			self.futures[ index ] = self._broker( value )
		else:
			self.futures[ index ] = map( lambda v: self._broker( v ), value )

	def __delitem__( self, index ):
		del self.futures[ index ]

	def insert( self, index, value ):
		self.futures.insert( index, self._broker( value ) )

	def __len__( self ):
		return len( self.futures )

def brokered_saved_tensor_hook(broker, pin_memory = False):
	def pack_to_cpu(tensor: t.Tensor):
		return tensor.device, broker(tensor, pin_memory)

	@uncurry
	def unpack_from_cpu(device, f_tensor):
		return f_tensor.result().to(device)

	return t.autograd.graph.saved_tensors_hooks(pack_to_cpu, unpack_from_cpu)
