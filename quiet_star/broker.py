import os
from abc import ABCMeta, abstractmethod
from collections.abc import MutableSequence
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor, Future, CancelledError
from operator import methodcaller
from queue import SimpleQueue

import torch as t
from lightning_utilities import apply_to_collection


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
				flat = tensor.reshape( -1 )
				n_full = flat.shape[ 0 ] // self._length
				rem = flat.shape[ 0 ] % self._length
				receive = t.empty( flat.shape[ 0 ], dtype = self.dtype, device = t.device( "cpu" ) )
				for i in range( n_full ):
					target.copy_( flat[ i * self._length: (i + 1) * self._length ] )
					receive[ i * self._length: (i + 1) * self._length ].copy_( target )
				if rem > 0:
					target[ :rem ].copy_( flat[ -rem: ] )
					receive[ -rem: ].copy_( target[ :rem ] )
				return receive.reshape_as( tensor )
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
		self._size = 1
		self._chop = chop
		self._subbrokers = { }

	def __call__( self, tensor ):
		if not (subbroker := self._subbrokers.get( tensor.dtype, None )):
			print( f"Creating broker for tensors of dtype {tensor.dtype}", flush = True )
			subbroker = _ChoppingTensorBroker( self._chop, tensor.dtype, self._size )
			self._subbrokers[ tensor.dtype ] = subbroker
		return subbroker( tensor )


class CollectionBroker( Broker ):
	def __init__( self, chop ):
		self._tensor_broker = TensorBroker( chop )

	def __call__( self, tensors ):
		return CollectedFuture( apply_to_collection( tensors, t.Tensor, self._tensor_broker ) )


class Present:
	__slots__ = ( "_result", "_exception", "_cancelled" )

	def __init__( self, future ):
		self._result = None
		self._exception = None
		self._cancelled = False
		try:
			self._result = future.result()
		except Exception as e:
			if isinstance( e, CancelledError ):
				self._cancelled = True
			self._exception = e

	def cancel( self ):
		return self._cancelled

	def cancelled( self ):
		return self._cancelled

	def running( self ):
		return False

	def done( self ):
		return self._result is not None

	def result( self, timeout = None ):
		if self._exception is not None:
			raise self._exception
		return self._result

	def exception( self, timeout = None ):
		if isinstance( self._exception, CancelledError ):
			raise self._exception
		return self._exception

	def add_done_callback( self, callback ):
		try:
			callback( self )
		except Exception:
			futures._base.LOGGER.exception( 'exception calling callback for %r', self )


class CollectedFuture:
	def __init__( self, collection ):
		self._collection = collection
		self._done_callbacks = None
		self._n_done = 0
		self._n_futures = 0
		apply_to_collection( self._collection, Future, self._inc )

	def _inc( self, f ):
		self._n_futures += 1

	def _collect_op_all( self, op ):
		retval = False

		def impl( f ):
			nonlocal retval
			retval &= op( f )

		apply_to_collection( self._collection, Future, impl )
		return retval

	def _collect_op_any( self, op ):
		retval = False

		def impl( f ):
			nonlocal retval
			retval |= op( f )

		apply_to_collection( self._collection, Future, impl )
		return retval

	def cancel( self ):
		return self._collect_op_all( methodcaller( "cancel" ) )

	def cancelled( self ):
		return self._collect_op_all( methodcaller( "cancelled" ) )

	def running( self ):
		return self._collect_op_any( methodcaller( "running" ) )

	def done( self ):
		return self._collect_op_all( methodcaller( "done" ) )

	def result( self, timeout = None ):
		return apply_to_collection( self._collection, Future, methodcaller( "result", timeout ) )

	def exception( self, timeout = None ):
		try:
			apply_to_collection( self._collection, Future, methodcaller( "result", timeout ) )
			return None
		except (TimeoutError, CancelledError) as e:
			raise
		except Exception as e:
			return e

	def _mark_done( self, _ ):
		self._n_done += 1
		if self._n_done == self._n_futures:
			for cb in self._done_callbacks:
				cb( self )

	def add_done_callback( self, callback ):
		first_callback = self._done_callbacks is None
		if first_callback:
			self._done_callbacks = [ ]
		self._done_callbacks.append( callback )
		if first_callback:
			apply_to_collection( self._collection, Future, methodcaller( "add_done_callback", self._mark_done ) )


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


def be_nice():
	os.nice( 10 )
