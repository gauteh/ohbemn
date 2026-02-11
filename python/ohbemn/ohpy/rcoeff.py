import numpy as np


def berkoff_partial_reflection(R, phase_shift=None, incidence=0.0):
    """
    Calculate the real and imaginary reflection coefficient for a partially reflecting boundary, by using Berkhoff (1976)'s and Isaacson and Qu (1990)'s method. Equations 3.2.10 and Equation 15 respectively.

    Args:

        R: Scalar or complex reflection coefficient.

        phase_shift: Phase shift of reflected wave (unless R is complex), in
                     radians between 0 and pi/2. Default is 0.

        incidence: incidence angle (radians, default 0.)

    Returns:
        A complex transmission coefficient a = (a_1 + ia_2)

    The boundary condition is:

    .. math::

        \\frac{\\partial \\phi}{\\partial n} + \\alpha k \\phi = 0 \\
        \\alpha = \\alpha_1 + i \\alpha_2

    Example:

        Boundary condition with partially reflecting boundary:

        >>> a = berkoff_partial_reflection(.7)
        >>> bc.f[i] = 0.
        >>> bc.alpha[i] = a*k
        >>> bc.beta[i] = 1.

    """

    if np.iscomplex(R):
        phase_shift = np.angle(R)
        R = np.abs(R)
    else:
        if phase_shift is None:
            phase_shift = 0.

    assert R >= 0 and R <= 1., f"R must be between 0 and 1: {R}"

    if phase_shift < 0. or phase_shift > (np.pi / 2):
        raise ValueError(
            f"phase_shift ({phase_shift}) must be between 0 and pi/2")

    a_1 = (2. * R * np.sin(phase_shift)) / (1. + R**2 +
                                            2. * R * np.cos(phase_shift))

    a_2 = (1 - R**2) / (1 + R**2 + 2. * R * np.cos(phase_shift))

    return a_1 + 1j * a_2
