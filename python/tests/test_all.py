import pytest
import numpy as np
import ohbemn


def test_sum_as_string():
    assert ohbemn.sum_as_string(1, 1) == "2"


@pytest.mark.parametrize("impl,fun", [("py", ohbemn.intop.l_2d), ("rust", ohbemn.l_2d)])
def test_l_2d(benchmark, impl, fun):
    p = [1, 1]
    qa = [0, 0]
    qb = [10, 10]

    p = np.array(p, dtype=np.float64)
    qa = np.array(qa, dtype=np.float64)
    qb = np.array(qb, dtype=np.float64)
    r = benchmark(fun, 3, p, qa, qb, False)
    print(r)

    np.testing.assert_almost_equal(r, 0.6585824828837403+0.5120036628488261j, decimal=6)

def test_complex_quad():
    from scipy.special import hankel1
    from numpy.linalg import norm

    qa = np.array([0, 0])
    qb = np.array([10, 10])
    p = np.array([1, 1])
    k = 10

    def l(q):
        print(q)
        print(p)
        h = hankel1(0, k * norm(p - q))
        print(h)
        return h

    print("h1 =", hankel1(0, 5 * 3))
    r = ohbemn.intop.complex_quad_2d(l, qa, qb)
    print(r)

def test_hankel1():
    from scipy.special import hankel1

    for j in np.arange(0, 10, 1):
        h = hankel1(0, (10.+0j) * j)
        print(f"{j} = {h}")

