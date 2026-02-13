import numpy as np

from .kirby1983 import kirby_pwa_coeff

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


## Grazing and incidence angle
# Angle increase clock-wise in oceanographic and meteorology context. But
# counter-clock-wise in the unit-circle.
#
#
# (direction from) _     0/360 deg.
#                    \   |
#                     \ i| <---- incidence angle
#                      \ |
# grazing angle -->   g \|
# ------------------------------ interface -----  90 deg
#                        |\ g
#                        | \
#                        |i \
#                        |   \  __ (direction to)
#                       180 deg
#


def theta_normal(theta, n):
    """
    calculate incident theta in xy onto segment with normal, n.

    theta: incident angle onto a horizontal segment with vertical normal.

    n: [x, y]

    returns angle in -90 to 90 in radians (-pi/2, pi/2)
    """

    assert np.abs(theta) <= (
        np.pi /
        2), f"theta out of range [-pi/2, pi/2]: {theta}, {np.rad2deg(theta)}"

    # a = np.arctan(n[1] / n[0])
    # print("normal:", a, np.rad2deg(a))
    # return theta + (a - np.pi / 2)

    alpha = np.pi / 2 + theta  # angle from x axis, counter clock-wise
    nalpha = np.arctan2(n[1], n[0])
    return alpha - nalpha


def dir_from_to_incidence(theta, n):
    """
    `theta` is direction from onto a horizontal segment (angle positive clock-wise!).

    Calculate the relative incident angle onto segment with normal, n.

    Normal is pointing in the direction of the outside of the segment.

    theta (radian): incident angle onto a horizontal segment with vertical
    normal.

    n ([x, y]): normal of segment, postive outwards.

    Return:

        th: angle in +/-0 to +/-90 in radians (0, pi/2): positive sign is on the outside, negative on inside.

        outside: boolean, true if coming from same side as normal is pointing.
    """

    assert theta >= 0 and theta <= (
        2 *
        np.pi), f'theta must be in range 0 to 2*pi: {np.rad2deg(theta)} deg'

    # Angle of normal: counter-clock-wise! Starting from the top
    nalpha = np.arctan2(n[1], n[0]) - np.pi / 2
    cnalpha = 2 * np.pi - nalpha  # clock-wise.
    cnalpha = cnalpha % (2 * np.pi)
    fth = theta - cnalpha
    fth = fth % (2 * np.pi)

    assert fth >= 0 and fth <= 2 * np.pi, np.rad2deg(fth)

    if fth >= 3 / 2 * np.pi:
        # NW
        outside = True
        th = 2 * np.pi - fth
    elif fth <= np.pi / 2:
        # NE
        outside = True
        th = fth
    elif fth >= np.pi / 2 and fth <= np.pi:
        # SE
        outside = False
        th = np.pi - fth
    elif fth >= np.pi and fth <= 3 / 2 * np.pi:
        # SW
        outside = False
        th = fth - np.pi

    # outside = np.abs(fth) <= np.pi/2 # on same side as normal is pointing
    # th = fth % (np.pi/2) # incidence angle, symmetric around normal.

    # print(
    #     f'nalpha={np.rad2deg(nalpha)}, cnalpha={np.rad2deg(cnalpha)}, theta={np.rad2deg(theta)}, fth={np.rad2deg(fth)}, th={np.rad2deg(th)}, {outside=}'
    # )

    # kx, ky, _dto = dir_from_k(np.rad2deg(theta))

    return th, outside


def abs_incidence(theta):
    """
    return absolute incidence angle as long as its in the -90, 90 range.

    input and output in radians, otherwise return negative.
    """

    if np.abs(theta) <= np.pi / 2:
        return np.abs(theta)
    else:
        return None


def dir_to_k_grazing(dto, k=1.):
    """
    Calculate kx and ky from wave traveling 'direction to' in degrees.

    E.g.: Waves coming from the north-north-west towards 120 deg:

        120 - 90

    """

    gr = dto - 90.  # grazing angle

    a = np.deg2rad(gr)
    kx = np.cos(a)
    ky = np.sin(a)

    return kx * k, ky * k, gr


def dir_to_k_incidence(dto, k=1.):
    """
    Calculate kx and ky and incidence angle from wave traveling 'direction to' in degrees.
    """

    a = np.deg2rad(dto)
    kx = np.sin(a)
    ky = np.cos(a)

    return kx * k, ky * k, dto


def flipdir_deg(dt):
    """
    Flip direction from dir to to dir from or vice versa.
    """
    return (dt + 180) % 360


def flipdir_rad(dt):
    """
    Flip direction from dir to to dir from or vice versa.
    """
    return (dt + np.pi) % (2 * np.pi)


