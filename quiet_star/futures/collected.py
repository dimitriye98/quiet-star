from _operator import methodcaller
from concurrent.futures import Future, CancelledError

from lightning_utilities import apply_to_collection

from .abc import StdFuture


class CollectedFuture( StdFuture ):
	def __init__( self, collection ):
		self._collection = collection
		self._done_callbacks = None
		self._n_done = 0
		self._n_futures = 0

		def inc( f ):
			self._n_futures += 1

		apply_to_collection( self._collection, Future, inc )

	def _all( self, op ):
		retval = False

		def impl( f ):
			nonlocal retval
			retval &= op( f )

		apply_to_collection( self._collection, Future, impl )
		return retval

	def _any( self, op ):
		retval = False

		def impl( f ):
			nonlocal retval
			retval |= op( f )

		apply_to_collection( self._collection, Future, impl )
		return retval

	def cancel( self ):
		return self._all( methodcaller( "cancel" ) )

	def cancelled( self ):
		return self._all( methodcaller( "cancelled" ) )

	def running( self ):
		return self._any( methodcaller( "running" ) )

	def done( self ):
		return self._all( methodcaller( "done" ) )

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

	def set_exception( self, exception ):
		raise NotImplementedError

	def set_result( self, result ):
		raise NotImplementedError

	def set_running_or_notify_cancel( self ):
		return self._all( methodcaller( "set_running_or_notify_cancel" ) )
