import numpy as np

g = 9.80665  # [m/s^2] standard gravity


def eta(phi, k, c):
    """
    Calculate surface elevation (eta) from the velocity potential (phi):

    From the dynamic boundary condition:

    .. math::

        \frac{\partial phi}{\partial t} = -g \eta

    :math:`\phi` is the complex amplitude of the vector potential of a harmonic wave. The time component is:

    .. math::

        \exp(-i \omega t)

    which is straightforward to differentiate (since :math:`\phi` is otherwise constant in time).
    """
    w = k * c
    return (1j * -1 / g * w * phi)


def wavec_interm(T, h, g=g, N=200, prt=False):
    """
    Phase speed (iterative, N, from jan-victor)
    """
    T = np.atleast_1d(T)
    assert np.all(h > 0)
    w = 2 * np.pi / T
    kd = (2 * np.pi)**2 / g / T**2
    k = kd

    for n in range(N):
        if prt: print('.', end='', flush=True)
        k = w**2 / g / np.tanh(k * h)

    c = w / k
    kh = k * h
    kh[kh > 2 * np.pi] = 2 * np.pi
    n = 0.5 * (1 + 2 * kh / np.sinh(2 * kh))
    cg = c * n
    return c, cg, k


def wavec_deep(T, g=9.82):
    cp = g / (2 * np.pi) * T
    k = cp * T
    cg = cp / 2

    return cp, cg, k
