import numpy as np
from numpy.testing import assert_almost_equal

import ohbemn as oh
from ohbemn import ohpy


def test_phi_eta():
    c = 100.
    k = .5

    phi = 2. * np.exp(1j * .5 * 13)

    eta = ohpy.wave.eta(phi, k, c)
    print(eta)

    phi2 = ohpy.wave.phi(eta, k, c)
    assert_almost_equal(phi, phi2)
