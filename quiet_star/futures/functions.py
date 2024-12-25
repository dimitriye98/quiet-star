from concurrent.futures import Future

from toolz import identity

from .collected import CollectedFuture

__all__ = [ "collect", "zip", "map", "flatmap" ]


def collect( col ):
	return CollectedFuture( col )


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
		else:
			self.target.set_result( self.op( future.result() ) )


def map( fn, future ):
	ret = Future()
	future.add_done_callback( _MapCb( ret, fn ) )
	return ret

class _FlatMapCb( object ):
	def __init__( self, target, op ):
		self.target = target
		self.op = op

	def __call__( self, future ):
		if future.cancelled():
			self.target.cancel()
		elif e := future.exception():
			self.target.set_exception( e )
		else:
			f2 = self.op( future.result() )
			f2.add_done_callback( _MapCb( self.target, identity ) )

def flatmap( fn, future ):
	ret = Future()
	future.add_done_callback( _FlatMapCb( ret, fn ) )
	return ret
