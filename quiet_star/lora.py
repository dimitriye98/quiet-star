
def lora_mixin( cls ):
	class LoraConfig( cls.config_class ):
		is_composition = True

	class LoraMixin( cls ):
		def __init__( self, *args, **kwargs ):
			super().__init__( *args, **kwargs )

			self._lm_model