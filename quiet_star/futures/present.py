from concurrent import futures
from concurrent.futures import CancelledError, InvalidStateError

from .abc import RichFuture


class Present( RichFuture ):
	__slots__ = ( "_result", "_exception", "_cancelled" )

	def __init__( self, future ):
		self._result = None
		self._exception = None
		self._cancelled = False
		self._callbacks = []
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
			self._callbacks.append( callback )
			callback( self )
		except Exception:
			futures._base.LOGGER.exception( 'exception calling callback for %r', self )

	def set_exception( self, exception ):
		raise InvalidStateError

	def set_result( self, result ):
		raise InvalidStateError

	def set_running_or_notify_cancel( self ):
		raise InvalidStateError
