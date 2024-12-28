import numpy as np

def eta(phi, k, c):
    """
    Calculate surface elevation (eta) from the velocity potential (phi):

    From the dynamic boundary condition:

    .. math::

        \frac{\partial phi}{\partial t} = -g \eta

    :math:`\phi` is the complex amplitude of the vector potential of a harmonic wave. The time component is:

    .. math::

        \exp(i \omega t)

    which is straightforward to differentiate (since :math:`\phi` is otherwise constant in time).
    """
    raise NotImplemented
