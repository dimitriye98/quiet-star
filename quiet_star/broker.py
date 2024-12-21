import logging
from abc import ABCMeta, abstractmethod
from collections.abc import MutableSequence
from concurrent.futures import ThreadPoolExecutor
from operator import methodcaller
from queue import SimpleQueue

import torch as t
from lightning_utilities import apply_to_collection

from . import futures as ft

logger = logging.getLogger( __name__ )


class Broker( metaclass = ABCMeta ):
	@abstractmethod
	def __call__( self, obj ):
		pass


class _ChoppingTensorBroker( Broker ):
	def __init__( self, length, dtype, size ):
		self.pool = ThreadPoolExecutor( max_workers = size )
		self._idle_tensors = SimpleQueue()
		self.dtype = dtype
		bytes_per_elem = t.tensor( [ ], dtype = dtype ).element_size()
		self._length = length // bytes_per_elem

		self.pool.submit( self.__create_target_tensors, size )

	def __create_target_tensors( self, n ):
		with t.inference_mode():
			for i in range( n ):
				self._idle_tensors.put(
					t.empty(
						self._length, dtype = self.dtype, device = t.device( "cpu" ),
						requires_grad = False ).pin_memory() )

	@t.inference_mode()
	def __send_to_cpu( self, tensor ):
		target = self._idle_tensors.get()
		try:
			stream = t.cuda.Stream()
			with t.cuda.stream( stream ):
				flat = tensor.view( -1 )
				shape = tensor.shape
				# del tensor
				n_full = flat.shape[ 0 ] // self._length
				rem = flat.shape[ 0 ] % self._length
				receive = t.empty( flat.shape[ 0 ], dtype = self.dtype, device = t.device( "cpu" ) )
				for i in range( n_full ):
					target.copy_( flat[ i * self._length: (i + 1) * self._length ] )
					receive[ i * self._length: (i + 1) * self._length ].copy_( target )
				if rem > 0:
					target[ :rem ].copy_( flat[ -rem: ] )
					receive[ -rem: ].copy_( target[ :rem ] )
				return receive.view( shape )
		finally:
			self._idle_tensors.put( target )

	def __call__( self, tensor ):
		assert self.dtype == tensor.dtype

		ret = self.pool.submit( self.__send_to_cpu, tensor )
		return ret

	def __del__( self ):
		self.pool.shutdown()


class _MonotypeTensorBroker( Broker ):
	@property
	def dtype( self ):
		return self._dtype

	def __init__( self, numel, dtype, size ):
		self.pool = ThreadPoolExecutor( max_workers = size )
		self._idle_tensors = SimpleQueue()
		self._dtype = dtype
		self._numel = numel

		self.pool.submit( self.__create_target_tensors, size )

	def __create_target_tensors( self, n ):
		with t.inference_mode():
			for i in range( n ):
				self._idle_tensors.put(
					t.empty(
						self._numel, dtype = self._dtype, device = t.device( "cpu" ),
						requires_grad = False ).pin_memory() )

	@t.inference_mode()
	def __send_to_cpu( self, tensor ):
		target = self._idle_tensors.get()
		try:
			stream = t.cuda.Stream()
			with t.cuda.stream( stream ):
				target.copy_( tensor.reshape_as( target ) )
			return target.clone().reshape( tensor.shape )
		finally:
			self._idle_tensors.put( target )

	def __call__( self, tensor ):
		assert self.dtype == tensor.dtype
		assert self._numel == tensor.numel()

		ret = self.pool.submit( self.__send_to_cpu, tensor )
		return ret

	def __del__( self ):
		self.pool.shutdown()


# TODO: Implement this more efficiently, e.g. using an LRU cache of target tensors
class TensorBroker( Broker ):
	def __init__( self, chop = 104857600 ):
		self._size = 2
		self._chop = chop
		self._subbrokers = { }

	def __call__( self, tensor ):
		if not (subbroker := self._subbrokers.get( tensor.dtype, None )):
			logger.log( logging.INFO, f"Creating subbroker for tensors of dtype {tensor.dtype}" )
			subbroker = _ChoppingTensorBroker( self._chop, tensor.dtype, self._size )
			self._subbrokers[ tensor.dtype ] = subbroker
		return subbroker( tensor )


class CollectionBroker( Broker ):
	def __init__( self, chop ):
		self._tensor_broker = TensorBroker( chop )

	def __call__( self, tensors ):
		return ft.collect( apply_to_collection( tensors, t.Tensor, self._tensor_broker ) )


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
