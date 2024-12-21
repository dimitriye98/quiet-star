from concurrent.futures import Future
from operator import attrgetter

from toolz.curried import do, compose_left

__all__ = [ ]

from .collected import CollectedFuture

_export = do( compose_left( attrgetter( "__name__" ), __all__.append ) )


@_export
def collect( col ):
	return CollectedFuture( col )


@_export
def zip( *futures ):
	return CollectedFuture( futures )


class _MapCb( object ):
	def __init__( self, target, op ):
		self.target = target
		self.op = op

	def __call__( self, future ):
		if future.cancelled():
			self.target.cancel()
		elif e := future.exception():
			self.target.set_exception( e )
		elif future.done():
			self.target.set_result( self.op( future.result() ) )


@_export
def map( fn, future ):
	ret = Future()
	future.add_done_callback( _MapCb( ret, fn ) )
	return ret
