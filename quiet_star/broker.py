from abc import ABCMeta, abstractmethod
from collections.abc import MutableSequence
from concurrent.futures import ThreadPoolExecutor, Future
from queue import SimpleQueue

import torch as t
from lightning_utilities import apply_to_collection


class Broker( metaclass = ABCMeta ):
	@abstractmethod
	def __call__( self, obj ):
		pass


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
	def __init__( self, size ):
		self._size = size
		self._subbrokers = { }

	def __call__( self, tensor ):
		if not (subbroker := self._subbrokers.get( (tensor.numel(), tensor.dtype), None )):
			subbroker = _MonotypeTensorBroker( tensor.numel(), tensor.dtype, self._size )
			self._subbrokers[ (tensor.numel(), tensor.dtype) ] = subbroker
		return subbroker( tensor )


class CollectionBroker( Broker ):
	def __init__( self, size ):
		self._tensor_broker = TensorBroker( size )

	def __call__( self, tensors ):
		return apply_to_collection( tensors, t.Tensor, self._tensor_broker )


class BrokeredList( MutableSequence ):
	def __init__( self, size ):
		self._broker = CollectionBroker( size )
		self._list = [ ]

	@staticmethod
	def _unwrap( c ):
		return apply_to_collection(c, Future, Future.result)

	def __getitem__( self, index ):
		if isinstance( index, int ):
			return self._unwrap( self._list[ index ] )
		else:
			return map( self._unwrap, self._list[ index ] )

	def __setitem__( self, index, value ):
		if isinstance( index, int ):
			self._list[ index ] = self._broker( value )
		else:
			self._list[ index ] = map( self._broker, value )

	def __delitem__( self, index ):
		del self._list[ index ]

	def insert( self, index, value ):
		self._list.insert( index, self._broker( value ) )

	def __len__( self ):
		return len( self._list )
