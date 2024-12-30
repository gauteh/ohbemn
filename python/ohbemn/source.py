import numpy as np
from numpy.linalg import norm
from scipy.special import hankel1

def point(k, x, y):
    """
    Point source in 2D.

    Params:

        k: wavenumber

        x,y: point to evaluate
    """
    return 1j * hankel1(0, k * r) / 4.0

def dpoint(k, x, y, p0, p1):
    """
    Normal derivative of point.
    """
    raise NotImplemented

def plane(kx, ky, x, y):
    raise NotImplemented

def dplane(kx, ky, x, y, p0, p1):
    raise NotImplemented
