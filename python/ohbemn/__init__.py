from .ohbemn import *  # rust implementation
from . import ohpy
from .ohpy import wave


__doc__ = ohbemn.__doc__
if hasattr(ohbemn, "__all__"):
    __all__ = ohbemn.__all__
