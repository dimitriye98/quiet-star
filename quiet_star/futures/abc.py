from abc import ABC, abstractmethod
from typing import Protocol

class RichFuture( ABC ):
	""" An abstract class for shadowing the stdlib concurrent.futures.Future class. """
	@abstractmethod
	def cancel( self ):
		...

	@abstractmethod
	def cancelled( self ):
		...

	@abstractmethod
	def running( self ):
		...

	@abstractmethod
	def done( self ):
		...

	@abstractmethod
	def result( self, timeout = None ):
		...

	@abstractmethod
	def exception( self, timeout = None ):
		...

	@abstractmethod
	def add_done_callback( self, callback ):
		...

	@abstractmethod
	def set_running_or_notify_cancel( self ):
		...

	@abstractmethod
	def set_result( self, result ):
		...

	@abstractmethod
	def set_exception( self, exception ):
		...

class ProtoStdFuture( Protocol ):
	def add_done_callback( self, callback ):
		...

	def cancel( self ):
		...

	def cancelled( self ):
		...

	def done( self ):
		...

	def exception( self, timeout = None ):
		...

	def result( self, timeout = None ):
		...

	def running( self ):
		...

	def set_exception( self, exception ):
		...

	def set_result( self, result ):
		...

	def set_running_or_notify_cancel( self ):
		...