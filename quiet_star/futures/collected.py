from _operator import methodcaller
from concurrent.futures import Future, CancelledError

from lightning_utilities import apply_to_collection

from .abc import RichFuture

# FIXME: CollectedFuture should work for anything which implements the required methods, not just the stdlib future and subclasses of RichFuture
class CollectedFuture( RichFuture ):
	def __init__( self, collection ):
		self._collection = collection
		self._done_callbacks = None
		self._pending_futures = set()

		def inc( f ):
			self._pending_futures.add( f )

		apply_to_collection( self._collection, Future, inc )

	def _all( self, op ):
		retval = False

		def impl( f ):
			nonlocal retval
			retval &= op( f )

		apply_to_collection( self._collection, (Future, RichFuture), impl )
		return retval

	def _any( self, op ):
		retval = False

		def impl( f ):
			nonlocal retval
			retval |= op( f )

		apply_to_collection( self._collection, (Future, RichFuture), impl )
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
		return apply_to_collection( self._collection, (Future, RichFuture), methodcaller( "result", timeout ) )

	def exception( self, timeout = None ):
		try:
			apply_to_collection( self._collection, (Future, RichFuture), methodcaller( "result", timeout ) )
			return None
		except (TimeoutError, CancelledError) as e:
			raise
		except Exception as e:
			return e

	def _mark_done( self, f ):
		self._pending_futures.discard( f )
		if not self._pending_futures:
			self._do_callbacks()

	def _do_callbacks( self ):
		# Pop is atomic, so this is thread-safe
		# and guarantees each callback is executed only once
		# We can't however guarantee the order
		# of the callback executions
		try:
			while cb := self._done_callbacks.pop():
				cb( self )
		except IndexError:
			pass

	def add_done_callback( self, callback ):
		first_callback = self._done_callbacks is None
		if first_callback:
			self._done_callbacks = [ callback ]
			def impl(f):
				if hasattr(f, "add_done_callback") and callable(f.add_done_callback):
					f.add_done_callback(self._mark_done)
			apply_to_collection(self._collection, (Future, RichFuture), impl )
		else:
			self._done_callbacks.append( callback )
			if not self._pending_futures:
				self._do_callbacks()

	def set_exception( self, exception ):
		raise NotImplementedError

	def set_result( self, result ):
		raise NotImplementedError

	def set_running_or_notify_cancel( self ):
		return self._all( methodcaller( "set_running_or_notify_cancel" ) )
