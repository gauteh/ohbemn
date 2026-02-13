import numpy as np
from scipy import integrate as int

from .wave import wavec_interm as wavec


#%% Calculate reflection coefficients using Kirby - plane-wave approximation for a single step.
def m_l(k, theta):
    """
    k is wavenumber in medium.

    theta is the incident angle (radians)

    Returns:

        m: along-canyon (boundary) wave number, constant in Snells law
        l: cross-canyon wave number
    """
    assert np.all(k > 0)
    m = k * np.sin(theta)  # Kirby: below Eq. 2.3.
    # l = k * np.sin(theta) # Thomson 2007.
    l = np.sqrt(k**2 - m**2)

    # print("la=", l)
    # print("lb=", np.sqrt(k**2 - m**2))

    return m, l


def l(k, m):
    if k < m:
        return 1j * np.sqrt(m**2 - k**2)
    else:
        return np.sqrt(k**2 - m**2)


def Xi(z, ki, hi):
    # eq: 2.7 in Kirby
    return np.cosh(ki * (hi + z))


def I1(k1, h1, k2, h2):
    """
    h1 and h2 is depth (positive) of respectively left shelf and trench.
    """
    assert h1 > 0
    assert h2 > 0

    # eq: 4.3 in Kirby
    def integrand(z):
        return Xi(z, k1, h1) * Xi(z, k2, h2)

    return int.quad(integrand, -h1, 0)[0]


def I2(k2, h2, k3, h3):
    """
    h2 and h3 is depth (positive) of respectively trench and right side.
    """
    assert h2 > 0
    assert h3 > 0

    # eq: 4.3 in Kirby
    def integrand(z):
        return Xi(z, k3, h3) * Xi(z, k2, h2)

    return int.quad(integrand, -h3, 0)[0]


def I3(k2, h2):
    """
    h2 (positive) of trench.
    """
    assert h2 > 0

    # eq: 4.3 in Kirby
    def integrand(z):
        return Xi(z, k2, h2)**2

    return int.quad(integrand, -h2, 0)[0]


def I4(k1, h1):
    """
    h1 (positive) left side of trench.
    """
    assert h1 > 0

    # eq: 4.3 in Kirby
    def integrand(z):
        return Xi(z, k1, h1)**2

    return int.quad(integrand, -h1, 0)[0]


def alpha(l1, k1, h1, k2, h2):
    # eq: 4.3
    I = I1(k1, h1, k2, h2)
    return l1 * I**2, I


def beta(l2, k1, h1, k2, h2):
    # eq: 4.3
    i3 = I3(k2, h2)
    i4 = I4(k1, h1)
    return l2 * i3 * i4


def kirby_pwa_coeff(theta, h1, h2, h3, L, f=None, k1=None, k2=None, k3=None):
    """
    Kirby et al. (1983), Eqs 4.4 a and b.

    Returns the square of the reflection and transmission coefficients.
    """
    assert h2 > h1

    assert theta <= (np.pi / 2) and theta >= 0, "theta in range [0, pi/2]"

    if k1 is None:
        T = 1 / f
        _, _, k1 = wavec(T, h1, prt=False)

    if k2 is None:
        T = 1 / f
        _, _, k2 = wavec(T, h2, prt=False)

    if k3 is None:
        T = 1 / f
        _, _, k3 = wavec(T, h3, prt=False)

    m1, l1 = m_l(k1, theta)
    m2 = m1
    # m3 = m1
    l2 = l(k2, m2)
    # l3 = l(k3, m3)

    a, i1 = alpha(l1, k1, h1, k2, h2)
    b = beta(l2, k1, h1, k2, h2)

    A = (a**2 - b**2)**2 / (4 * a**2 * b**2) * np.sin(l2 * L)**2

    Kr2 = A / (1 + A)
    Kt2 = 1 / (1 + A)

    return Kr2, Kt2
