"""
Coupled subregions (ohbemn.coupled) against published long-wave solutions.
"""
import numpy as np
from scipy.special import jv, jvp, hankel1, h1vp
import pytest

from ohbemn import coupled
from ohbemn.ohpy.wave import g


def ring(r, n=64):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return th, np.c_[r * np.cos(th), r * np.sin(th)]


# Longuet-Higgins (1967, Sects. 6-7): circular sill of radius a, depth h1 over the sill and h2
# outside, z = k1 a, eps = k2 / k1 = (h1 / h2)^(1/2). Over the sill phi = sum A_n J_n(k1 r) e^{in th},
# outside sum (B_n H_n(k2 r) + i^n J_n(k2 r)) e^{in th}.
a, h1, h2 = 1000., 10., 160.
eps = np.sqrt(h1 / h2)


def lh_F(n, z):
    """Eq. 6.16, G = z F (Eq. 6.17)."""
    return jv(n, z) * h1vp(n, eps * z) - eps * jvp(n, z) * hankel1(n, eps * z)


def lh_A(n, z):
    """Eq. 7.7."""
    return 2 * 1j**(n + 1) / (np.pi * eps * z * lh_F(n, z))


def lh_B(n, z):
    """Eq. 7.8 as printed. Solving Eq. 7.4 gives the opposite sign, which the BEM follows."""
    return 1j**n * (jv(n, z) * jvp(n, eps * z) - eps * jvp(n, z) * jv(n, eps * z)) / lh_F(n, z)


def sill(z, N=128):
    T = 2 * np.pi * a / (z * np.sqrt(g * h1))
    rim = coupled.Boundary.circle(a, N)
    return coupled.Problem(T, [coupled.Subregion(h1, outer=rim),
                               coupled.Exterior(h2, holes=[rim])]).solve(direction=0.)


@pytest.mark.parametrize('z, ns, tol', [(3.0, [0, 1, 2, 3, 4], 5e-3),
                                        (4.922409, [2], 2e-2)])  # first trapped mode, n = 2
def test_lh1967_sill(z, ns, tol):
    sol = sill(z)
    k1, k2 = z / a, eps * z / a
    th, p = ring(0.8 * a)
    fi = sol.field(p)
    th, p = ring(1.5 * a)
    fo = sol.field(p)
    for n in ns:
        An = np.mean(fi * np.exp(-1j * n * th)) / jv(n, k1 * 0.8 * a)
        Bn = (np.mean(fo * np.exp(-1j * n * th)) - 1j**n * jv(n, k2 * 1.5 * a)) / hankel1(n, k2 * 1.5 * a)
        assert abs(An / lh_A(n, z) - 1) < tol, (n, An, lh_A(n, z))
        assert abs(Bn / -lh_B(n, z) - 1) < tol, (n, Bn, -lh_B(n, z))


# Suh et al. (2005, Wave Motion 42): paraboloid pit h = h0 (1 - r^2 / a^2) for r < b in a shelf
# of depth d1 = h(b), k1 d1 = 0.167, b = L1 / 2. Long-wave Frobenius series (their Eqs. 11, 14-15),
# matched to the shelf by continuity of phi and dphi/dr at r = b (the depth is continuous).
d1, h0 = 3.2, 6.4
k1 = 0.167 / d1
T_suh = 2 * np.pi / (k1 * np.sqrt(g * d1))
b = 0.5 * 2 * np.pi / k1
a_suh = b / np.sqrt(1 - d1 / h0)


def bowl_R(n, rho, nu, mmax=400):
    """R_n(r) / a^n and its derivative with respect to rho = r / a, nu = omega a / (g h0)^(1/2)."""
    rho = np.asarray(rho, float)
    al, R, dR = 1., np.zeros_like(rho), np.zeros_like(rho)
    for m in range(0, mmax, 2):
        R += al * rho**(m + n)
        if m + n > 0:
            dR += al * (m + n) * rho**(m + n - 1)
        al *= ((m + n) * (m + n + 2) - nu**2 - n**2) / ((m + 2) * (m + 2 * n + 2))
    return R, dR


def suh_bowl(x, y, nmax=40):
    nu = 2 * np.pi / T_suh * a_suh / np.sqrt(g * h0)
    r, th = np.hypot(x, y), np.arctan2(y, x)
    out = np.zeros(r.shape, complex)
    o = r >= b
    for m in range(nmax + 1):
        Rb, dRb = bowl_R(m, b / a_suh, nu)
        Lb = dRb / Rb / a_suh
        M = np.array([[hankel1(m, k1 * b), -1.], [k1 * h1vp(m, k1 * b), -Lb]])
        A, B = np.linalg.solve(M, -np.array([jv(m, k1 * b), k1 * jvp(m, k1 * b)]))
        e = (2 if m else 1) * 1j**m * np.cos(m * th)
        out[o] += e[o] * A * hankel1(m, k1 * r[o])
        out[~o] += e[~o] * B * bowl_R(m, r[~o] / a_suh, nu)[0] / Rb
    out[o] += np.exp(1j * k1 * x[o])
    return out


def test_suh2005_bowl_terraced():
    """The bowl as M rings of constant depth (the mean depth of the bowl over each ring)."""
    M, N = 8, 64
    re = b * np.arange(1, M + 1) / M
    ri = np.r_[0, re[:-1]]
    dep = h0 * (1 - (ri**2 + re**2) / (2 * a_suh**2))
    rings = [coupled.Boundary.circle(r, N) for r in re]
    subs = [coupled.Subregion(dep[0], outer=rings[0])]
    subs += [coupled.Subregion(dep[j], outer=rings[j], holes=[rings[j - 1]]) for j in range(1, M)]
    subs += [coupled.Exterior(d1, holes=[rings[-1]])]
    sol = coupled.Problem(T_suh, subs).solve(direction=0.)

    # between the rings, away from the near-singular boundaries
    rm = np.r_[(ri + re) / 2, 1.5 * b, 3 * b]
    p = np.concatenate([ring(r, 24)[1] for r in rm])
    f, ref = sol.field(p), suh_bowl(p[:, 0], p[:, 1])
    r = np.hypot(*p.T)
    for m in (r < b, r > b):
        assert np.linalg.norm((f - ref)[m]) / np.linalg.norm(ref[m]) < 1e-2


def test_wall_cylinder():
    """Rigid (v = 0) circular cylinder of radius R, long waves (MacCamy & Fuchs 1954)."""
    h, R, T = 10., 100., 60.
    k = 2 * np.pi / T / np.sqrt(g * h)
    wall = coupled.Boundary.circle(R, 128)
    sol = coupled.Problem(T, [coupled.Exterior(h, holes=[wall])]).solve(direction=0.)
    th, p = ring(2 * R)
    ref = sum((2 if n else 1) * 1j**n * (jv(n, 2 * k * R) - jvp(n, k * R) / h1vp(n, k * R)
                                         * hankel1(n, 2 * k * R)) * np.cos(n * th)
              for n in range(30))
    assert np.max(np.abs(sol.field(p) - ref)) < 2e-3


@pytest.mark.parametrize('longwave', [True, False])
def test_transparent(longwave):
    """Subregions of equal depth leave the incident wave unchanged, for any boundary orientation."""
    s = np.linspace(0, 200, 40, endpoint=False)
    z = np.zeros_like(s)
    # counter-clockwise, reoriented by Boundary
    sq = coupled.Boundary.from_xy(np.r_[s, z + 200, 200 - s, z], np.r_[z, s, z + 200, 200 - s])
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    el = coupled.Boundary.from_xy(60 + 40 * np.cos(t), 100 + 70 * np.sin(t))
    h, T, d = 20., 10., 0.3
    subs = [coupled.Subregion(h, outer=sq, holes=[el]), coupled.Subregion(h, outer=el),
            coupled.Exterior(h, holes=[sq])]
    p = coupled.Problem(T, subs, longwave=longwave)
    sol = p.solve(direction=d)
    pts = np.array([[60., 100.], [150., 50.], [-100., 300.], [400., -50.]])
    k = subs[0].k
    ref = np.exp(1j * k * (pts[:, 0] * np.cos(d) + pts[:, 1] * np.sin(d)))
    assert np.max(np.abs(sol.field(pts) - ref)) < 1e-2  # converges as N^-2
