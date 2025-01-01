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
    Normal derivative of point source at x, y of edge from p0 to p1.

    Params:

        k: Wavenumber

        x,y: Point on edge/boundary.

        p0: point 0, (p0_x, p0_y) of edge.
        p1: point 1, (p1_x, p1_y) of edge.
    """
    raise NotImplemented


def plane(kx, ky, x, y):
    """
    Plane wave in 2D.

    Params:

        kx, ky: Horizontal wavenumber-components: $k = \\sqrt{kx^2 + ky^2}$.

        x, y: point to evaluate.
    """
    return np.exp(1j * (kx * x + ky * y))


def dplane(kx, ky, x, y, n):
    """
    The normal derivative of a plane wave onto an edge with normal n.
    """

    f = plane(kx, ky, x, y)
    dfdx = -1j * kx * f
    dfdy = -1j * ky  * f
    gradf = np.array([dfdx, dfdy]).T

    return gradf.dot(n)
