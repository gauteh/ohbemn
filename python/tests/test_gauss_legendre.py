import numpy as np
from numpy.polynomial.legendre import leggauss

# Node/weight pairs hard-coded in `complex_quad_2d` (src/integrators.rs), used
# to integrate over a boundary element parametrized as x = a + x_val * (b -
# a), x_val in [0, 1]. Following Kirkup (1998), these should be the standard
# 8-point Gauss-Legendre rule, mapped from the reference interval [-1, 1]
# onto [0, 1].
NW = [
    [0.980144928249, 5.061426814519e-02],
    [0.898333238707, 0.111190517227],
    [0.762766204958, 0.156853322939],
    [0.591717321248, 0.181341891689],
    [0.408282678752, 0.181341891689],
    [0.237233795042, 0.156853322939],
    [0.101666761293, 0.111190517227],
    [1.985507175123e-02, 5.061426814519e-02],
]


def test_nw_matches_8point_gauss_legendre():
    x, w = leggauss(8)

    # map from [-1, 1] to [0, 1]
    x_mapped = (x + 1) / 2
    w_mapped = w / 2

    # sort descending in x to match the order of NW
    order = np.argsort(-x_mapped)
    x_mapped = x_mapped[order]
    w_mapped = w_mapped[order]

    x_nw = np.array([n for n, _ in NW])
    w_nw = np.array([w for _, w in NW])

    np.testing.assert_allclose(x_mapped, x_nw, atol=1e-12)
    np.testing.assert_allclose(w_mapped, w_nw, atol=1e-12)


def test_nw_weights_sum_to_one():
    # weights on [0, 1] should sum to the length of the interval (1).
    w_nw = np.array([w for _, w in NW])
    np.testing.assert_allclose(np.sum(w_nw), 1.0, atol=1e-12)
