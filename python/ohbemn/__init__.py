from .ohbemn import *


__doc__ = ohbemn.__doc__
if hasattr(ohbemn, "__all__"):
    __all__ = ohbemn.__all__

from . import geometry
from . import wave
from .boundary import Region
from .solver import HelmholtzSolver
