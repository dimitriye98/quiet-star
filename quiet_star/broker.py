import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from math import prod
from queue import SimpleQueue
from tempfile import TemporaryDirectory, TemporaryFile

import torch as t


class LazyFuture( Future ):
	def __init__( self, delegate ):
		self.delegate = delegate
		self.__result = None
		self.__lazy_done = False
		self.__lazy_callbacks = [ ]
		self.__eager_callbacks = [ ]
		self.__done_callbacks = [ ]
		if self.delegate.done():
			self.__done_callback( delegate )
		else:
			self.__done = False
			self.delegate.add_done_callback( self.__done_callback )

	def __repr__( self ):
		return f"LazyFuture({self.delegate})"

	def __done_callback( self, future ):
		result = future.result()

		for f in self.__eager_callbacks:
			result = f( result )

		self.__result = result
		self.__done = True

		for f in self.__done_callbacks:
			f( self )

	def map( self, fn ):
		self.__eager_callbacks.append( fn )

	def lazy_map( self, fn ):
		self.__lazy_callbacks.append( fn )

	def cancel( self ):
		return self.delegate.cancel()

	def cancelled( self ):
		return self.delegate.cancelled()

	def running( self ):
		return self.delegate.running()

	def done( self ):
		return self.__done

	def add_done_callback( self, fn ):
		self.__done_callbacks.append( fn )

	def result( self, timeout = None ):
		if not self.done():
			self.delegate.result( timeout )
			while not self.done():
				time.sleep( 0 )
		if not self.__lazy_done:
			for fn in self.__lazy_callbacks:
				self.__result = fn( self.__result )
		return self.__result

	def set_running_or_notify_cancel( self ):
		return self.delegate.set_running_or_notify_cancel()

	def set_result( self, result ):
		self.delegate.set_result( result )

	def exception( self, timeout = None ):
		return self.delegate.exception( timeout )

	def set_exception( self, exception ):
		self.delegate.set_exception( exception )

	def __class_getitem__( cls, item, / ):
		return super().__class_getitem__( item )


class PinnedTensorBroker:
	@property
	def dtype( self ):
		return self.__dtype

	@property
	def size( self ):
		return prod( self.__shape )

	def __init__(
			self, shape, dtype, size, max_workers = None, serialize = False, dir = None, blocking_init = True, cm = None ):
		if max_workers is None:
			max_workers = size
		self.pool = ThreadPoolExecutor( max_workers = max_workers )
		self.__idle_tensors = SimpleQueue()
		self.__dtype = dtype
		self.__shape = shape
		self.serialize = serialize
		self.cm = cm if cm is not None else nullcontext()

		if dir is None:
			self.__temp_dir = TemporaryDirectory()
			self.dir = self.__temp_dir.name

		handle = self.pool.submit( self.__create_target_tensors, size )
		if blocking_init:
			handle.result()

	def __create_target_tensors( self, n ):
		with self.cm:
			for i in range( n ):
				self.__idle_tensors.put(
					t.empty( self.__shape, dtype = self.__dtype, device = t.device("cpu") ).detach().pin_memory() )

	@t.inference_mode()
	def __send_to_cpu( self, tensor ):
		with self.cm:
			target = self.__idle_tensors.get()
			stream = t.cuda.Stream()
			with t.cuda.stream( stream ):
				target.copy_( tensor.reshape( self.__shape ) )
			with t.device( "cpu" ):
				ret = target.clone().reshape( tensor.shape )
			self.__idle_tensors.put( target )
			return ret

	def __serialize( self, tensor ):
		tmp = TemporaryFile( dir = self.dir )
		t.save( tensor, tmp )
		return tmp

	@staticmethod
	def __deserialize( file ):
		return t.load( file )

	def send_to_cpu( self, tensor ):
		assert self.dtype == tensor.dtype
		assert self.size == prod( tensor.shape )

		ret = LazyFuture( self.pool.submit( self.__send_to_cpu, tensor ) )
		if self.serialize:
			ret.map( self.__serialize )
			ret.lazy_map( self.__deserialize )
		return ret
