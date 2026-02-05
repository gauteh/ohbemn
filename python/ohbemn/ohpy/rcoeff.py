import numpy as np


def berkoff_partial_reflection(R, phase_shift=None):
    """
    Calculate the real and imaginary reflection coefficient for a partially reflecting boundary, by using Berkhoff (1976)'s method. Equations 3.2.10.

    Args:

        R: Scalar or complex reflection coefficient.

        phase_shift: Phase shift of reflected wave (unless R is complex), in
                     radians between 0 and pi/2. Default is 0.

    Returns:
        A tuple of (a_1, a_2) in a = (a_1 + ia_2)

    The boundary condition is:

    .. math::

        \\frac{\\partial \\phi}{\\partial n} + \\alpha k \\phi = 0 \\
        \\alpha = \\alpha_1 + i \\alpha_2

    Example:

        Boundary condition with partially reflecting boundary:

        >>> a1, a2 = berkoff_partial_reflection(.7)
        >>> bc.f[i] = 0.
        >>> bc.alpha[i] = (a1 + 1j*a2)*k
        >>> bc.beta[i] = 1.

    """

    if np.iscomplex(R):
        phase_shift = np.angle(R)
        R = np.abs(R)
    else:
        if phase_shift is None:
            phase_shift = 0.

    if phase_shift < 0. or phase_shift > (np.pi / 2):
        raise ValueError(
            f"phase_shift ({phase_shift}) must be between 0 and pi/2")

    a_1 = (2. * R * np.sin(phase_shift)) / (1. + R**2 +
                                            2. * R * np.cos(phase_shift))

    a_2 = (1 - R**2) / (1 + R**2 + 2. * R * np.cos(phase_shift))

    return a_1, a_2
