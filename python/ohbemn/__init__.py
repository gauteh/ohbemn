from .ohbemn import *  # rust implementation
from . import ohpy
from .ohpy import wave, source, rcoeff

Interior = Orientation.Interior
Exterior = Orientation.Exterior

__doc__ = ohbemn.__doc__
if hasattr(ohbemn, "__all__"):
    __all__ = ohbemn.__all__


def style_plots():
    import matplotlib as mpl
    import cmocean
    mpl.style.use('ggplot')
    mpl.rc('image', cmap='cmo.amp')

    font = {'size': 8}
    mpl.rc('font', **font)
