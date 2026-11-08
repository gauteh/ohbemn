"""
Coupled subregions of constant depth (subregion or multi-zone BEM, Brebbia & Dominguez 1992,
Sect. 2.14).

The domain is split into subregions of constant depth, each satisfying the Helmholtz equation
with its own wavenumber. Across a boundary shared by two subregions, the potential and the flux
F dphi/dn are continuous, with F = g h for long waves (Bartholomeusz 1958, Longuet-Higgins 1967).
The boundary equations of all the subregions (``Solver.boundary_matrices``) are assembled with
these conditions into one linear system for the potential and the flux on all the boundaries.

Every ``Boundary`` is a closed polygon with its normals pointing out of the area it encloses (it
is reoriented if needed). A ``Subregion`` is the area inside ``outer`` and outside its ``holes``,
and the ``Exterior`` is the unbounded area outside its ``holes``, where the incident plane wave is
given. A boundary used by only one subregion needs a condition, ``alpha phi + beta v = f`` with
``v = dphi/dn`` along the boundary normal (default: a fully reflecting wall, v = 0).

Example, a circular sill of depth 10 m and radius 1000 m in 160 m depth (Longuet-Higgins 1967)::

    rim = Boundary.circle(1000., 256)
    p = Problem(T=600., subregions=[Subregion(10., outer=rim), Exterior(160., holes=[rim])])
    sol = p.solve(direction=0.)
    phi = sol.field(points)

Long waves only by default. With ``longwave=False`` the wavenumber is from the linear dispersion
relation and F = c cg (as in the mild-slope equation, Berkhoff 1972). This ignores the evanescent
modes at a step in intermediate depth and has not been validated here.

Time dependence exp(-i omega t).
"""
import numpy as np

from ..ohbemn import Solver, Region, Orientation
from . import wave


def _outward(vertices, edges):
    """Integral of x . n over the polygon, 2 x area if the normals of Region point outwards."""
    reg = Region(vertices, edges)
    a, b = vertices[edges[:, 0]], vertices[edges[:, 1]]
    ds = np.linalg.norm(b - a, axis=1)
    return np.sum(np.sum(reg.centers() * reg.normals(), 1) * ds)


def _inside(vertices, edges, pts):
    """Crossing number (even-odd) test of pts against the polygon edges."""
    a, b = vertices[edges[:, 0]], vertices[edges[:, 1]]
    out = np.zeros(len(pts), dtype=bool)
    for i in range(0, len(pts), 4096):
        x, y = pts[i:i + 4096, 0:1], pts[i:i + 4096, 1:2]
        c = (a[:, 1] > y) != (b[:, 1] > y)
        with np.errstate(divide='ignore', invalid='ignore'):
            xc = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
        out[i:i + 4096] = np.sum(c & (x < xc), 1) % 2 == 1
    return out


class Boundary:
    """A closed polygon (vertices and edges as for ``Region``), with normals out of it.

    ``alpha``, ``beta`` and ``f`` are only used if the boundary belongs to one subregion.
    """

    def __init__(self, vertices, edges, alpha=0., beta=1., f=0.):
        vertices = np.asarray(vertices, dtype=float)
        edges = np.asarray(edges, dtype=np.uint64)
        if _outward(vertices, edges) < 0:
            edges = np.ascontiguousarray(edges[:, ::-1])
        self.vertices, self.edges = vertices, edges
        self.region = Region(vertices, edges)
        self.N = self.region.len()
        self.centers = np.asarray(self.region.centers())
        self.normals = np.asarray(self.region.normals())
        self.alpha, self.beta, self.f = (np.broadcast_to(np.asarray(u, complex), (self.N,))
                                         for u in (alpha, beta, f))

    @classmethod
    def from_xy(cls, x, y, **kw):
        """Polygon through the points (x, y), closed from the last to the first point."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        if x[0] == x[-1] and y[0] == y[-1]:
            x, y = x[:-1], y[:-1]
        n = len(x)
        return cls(np.c_[x, y], np.c_[np.arange(n), (np.arange(n) + 1) % n], **kw)

    @classmethod
    def circle(cls, r, N, center=(0., 0.), **kw):
        t = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return cls.from_xy(center[0] + r * np.cos(t), center[1] + r * np.sin(t), **kw)

    def inside(self, pts):
        return _inside(self.vertices, self.edges, np.asarray(pts, float))


class Subregion:
    """The area inside ``outer`` and outside ``holes``, with constant depth."""
    exterior = False

    def __init__(self, depth, outer, holes=()):
        self.depth = depth
        self.parts = [(outer, 1.)] + [(h, -1.) for h in holes]  # +1: normal out of the subregion

    def contains(self, pts):
        o, _ = self.parts[0]
        m = o.inside(pts)
        for h, _ in self.parts[1:]:
            m &= ~h.inside(pts)
        return m


class Exterior(Subregion):
    """The unbounded area outside ``holes``, with constant depth and the incident wave."""
    exterior = True

    def __init__(self, depth, holes):
        self.depth = depth
        # Orientation.Exterior uses the normals out of the polygons, into the exterior.
        self.parts = [(h, 1.) for h in holes]

    def contains(self, pts):
        m = np.ones(len(pts), dtype=bool)
        for h, _ in self.parts:
            m &= ~h.inside(pts)
        return m


class Problem:
    """Coupled subregions at period T. At most one of the subregions can be an ``Exterior``."""

    def __init__(self, T, subregions, longwave=True, g=wave.g, mu=None):
        self.T, self.g, self.mu = T, g, mu
        self.subregions = list(subregions)
        assert sum(s.exterior for s in self.subregions) <= 1, 'only one exterior subregion'
        w = 2 * np.pi / T
        for s in self.subregions:
            if longwave:
                s.k, s.F = w / np.sqrt(g * s.depth), g * s.depth
            else:
                c, cg, s.k = wave.wavec_interm(T, s.depth, g=g)
                s.F = c * cg

        self.boundaries = []
        for s in self.subregions:
            for b, _ in s.parts:
                if not any(b is c for c in self.boundaries):
                    self.boundaries.append(b)
        self.users = [[s for s in self.subregions for c, _ in s.parts if c is b]
                      for b in self.boundaries]
        assert all(len(u) <= 2 for u in self.users), 'a boundary can be shared by two subregions'

        off = np.cumsum([0] + [b.N for b in self.boundaries])
        self._off = {id(b): o for b, o in zip(self.boundaries, off)}
        self.n = off[-1]  # unknowns: phi on all boundaries, then the flux q = F dphi/dn

        for s in self.subregions:
            V = np.concatenate([b.vertices for b, _ in s.parts])
            vo = np.cumsum([0] + [len(b.vertices) for b, _ in s.parts])
            E = np.concatenate([(b.edges if sg > 0 else b.edges[:, ::-1]) + np.uint64(o)
                                for (b, sg), o in zip(s.parts, vo)]).astype(np.uint64)
            s.region = Region(V, E)
            s.solver = Solver(s.region)
            s.orientation = Orientation.Exterior if s.exterior else Orientation.Interior

    def _cols(self, b):
        o = self._off[id(b)]
        return np.arange(o, o + b.N)

    def solve(self, direction=0., amplitude=1.):
        """Plane wave amplitude exp(i k (x cos(direction) + y sin(direction))) in the exterior."""
        n = self.n
        S = np.zeros((2 * n, 2 * n), dtype=complex)
        rhs = np.zeros(2 * n, dtype=complex)
        ext = [s for s in self.subregions if s.exterior]
        kvec = None
        if ext:
            kvec = ext[0].k * np.array([np.cos(direction), np.sin(direction)])
        row = 0
        for s in self.subregions:
            A, B = (np.asarray(m) for m in s.solver.boundary_matrices(s.orientation, s.k, self.mu))
            col = 0
            m = A.shape[0]
            for b, sg in s.parts:
                cb = slice(col, col + b.N)
                S[row:row + m, self._cols(b)] += B[:, cb]
                S[row:row + m, n + self._cols(b)] -= sg / s.F * A[:, cb]
                if s.exterior:  # scattered field: B (phi - phi_i) = A (v - v_i)
                    pi, vi = self._incident(kvec, b, amplitude)
                    rhs[row:row + m] += B[:, cb] @ pi - A[:, cb] @ (sg * vi)
                col += b.N
            row += m
        for b, u in zip(self.boundaries, self.users):
            if len(u) == 1:  # alpha phi + beta v = f, v along the boundary normal
                c = self._cols(b)
                S[row + np.arange(b.N), c] = b.alpha
                S[row + np.arange(b.N), n + c] = b.beta / u[0].F
                rhs[row:row + b.N] = b.f
                row += b.N
        assert row == 2 * n
        x = np.linalg.solve(S, rhs)
        return Solution(self, x[:n], x[n:], kvec, amplitude)

    @staticmethod
    def _incident(kvec, b, amplitude):
        p = amplitude * np.exp(1j * b.centers @ kvec)
        return p, 1j * (b.normals @ kvec) * p


class Solution:
    """Potential ``phi`` and flux ``q = F dphi/dn`` (along the boundary normals) on the boundaries."""

    def __init__(self, problem, phi, q, kvec, amplitude):
        self.problem, self.kvec, self.amplitude = problem, kvec, amplitude
        self._phi, self._q = phi, q

    def phi(self, b):
        return self._phi[self.problem._cols(b)]

    def q(self, b):
        return self._q[self.problem._cols(b)]

    def field(self, points):
        """Total potential at points (nan outside all the subregions)."""
        p = self.problem
        pts = np.atleast_2d(np.asarray(points, float))
        out = np.full(len(pts), np.nan, dtype=complex)
        for s in p.subregions:
            m = s.contains(pts) & np.isnan(out)
            if not m.any():
                continue
            ph, v = [], []
            for b, sg in s.parts:
                ph.append(self.phi(b))
                v.append(sg * self.q(b) / s.F)
                if s.exterior:
                    pi, vi = p._incident(self.kvec, b, self.amplitude)
                    ph[-1] = ph[-1] - pi
                    v[-1] = v[-1] - sg * vi
            out[m] = np.asarray(s.solver.sample_field(s.orientation, s.k, np.concatenate(ph),
                                                      np.concatenate(v), pts[m]))
            if s.exterior:
                out[m] += self.amplitude * np.exp(1j * pts[m] @ self.kvec)
        return out
