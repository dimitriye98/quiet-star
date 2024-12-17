from concurrent.futures import Future, ThreadPoolExecutor
from math import prod
from queue import SimpleQueue

import torch as t

class TensorBroker:
	@property
	def dtype( self ):
		return self.__dtype

	@property
	def size( self ):
		return prod( self.__shape )

	def __init__( self, shape, dtype, size, max_workers = None ):
		if max_workers is None:
			max_workers = size
		self.pool = ThreadPoolExecutor( max_workers = max_workers )
		self.__idle_tensors = SimpleQueue()
		self.__dtype = dtype
		self.__shape = shape

		self.pool.submit( self.__create_target_tensors, size )

	def __create_target_tensors( self, n ):
		with t.inference_mode():
			for i in range( n ):
				self.__idle_tensors.put(
					t.empty( self.__shape, dtype = self.__dtype, device = t.device( "cpu" ) ).detach().pin_memory() )

	@t.inference_mode()
	def __send_to_cpu( self, tensor ):
		with t.inference_mode():
			target = self.__idle_tensors.get()
			stream = t.cuda.Stream()
			with t.cuda.stream( stream ):
				target.copy_( tensor.reshape( self.__shape ) )
			with t.device( "cpu" ):
				ret = target.clone().reshape( tensor.shape )
			self.__idle_tensors.put( target )
			return ret

	def __call__( self, tensor ):
		assert self.dtype == tensor.dtype
		assert self.size == prod( tensor.shape )

		ret = self.pool.submit( self.__send_to_cpu, tensor )
		return ret

	def __del__( self ):
		self.pool.shutdown()
