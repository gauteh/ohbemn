import numpy as np

from . import geometry


class Region:
    vertices: np.ndarray
    _edges: np.ndarray
    named_partition: dict

    def __init__(self, vertices, edges):
        self.vertices = vertices
        self._edges = edges
        self.named_partition = {}

    def __repr__(self):
        result = "Region (\n"
        result += "aVertex({}) = {},\n ".format(self.vertices.shape[0],
                                                self.vertices)
        result += "aEdge({}) = {})".format(self._edges.shape[0], self._edges)
        # result += "namedPartition = {}\n)".format(self.named_partition)
        return result

    def plot(self, ax):
        for i in range(self.len()):
            p0 = self.vertices[self._edges[i, 0]]
            p1 = self.vertices[self._edges[i, 1]]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]])

    def len(self):
        return len(self._edges)

    def centers(self):
        if self._centers is None:
            self._centers = (self.vertices[self._edges[:, 0]] +
                             self.vertices[self._edges[:, 1]]) / 2.0
        return self._centers

    def edge(self, edge):
        return self.vertices[self._edges[edge, 0]], \
               self.vertices[self._edges[edge, 1]]

    def edges(self):
        result = np.empty((len(self._edges), 2, 2), dtype=np.float32)
        for i in range(len(self._edges)):
            result[i, :, :] = self.edge(i)
        return result

    def _compute_lengths_and_normals(self):
        # length of the boundary elements
        self._lengths = np.empty(self._edges.shape[0], dtype=np.float32)
        self._normals = np.empty((self._edges.shape[0], 2), dtype=np.float32)
        for i in range(self._lengths.size):
            a = self.vertices[self._edges[i, 0], :]
            b = self.vertices[self._edges[i, 1], :]
            ab = b - a
            normal = np.empty_like(ab)
            normal[0] = -ab[1]
            normal[1] = ab[0]
            length = np.linalg.norm(normal)
            self._normals[i] = normal / length
            self._lengths[i] = length

    def lengths(self):
        if self._lengths is None:
            self._compute_lengths_and_normals()
        return self._lengths

    def normals(self):
        if self._normals is None:
            self._compute_lengths_and_normals()
        return self._normals

    def areas(self, named_partition=None):
        """The areas of the surfaces created by rotating an edge around the x-axis."""
        if self._areas is None:
            self._areas = np.empty(self._edges.shape[0], dtype=np.float32)
            for i in range(self._areas.size):
                a, b = self.edge(i)
                self._areas[i] = np.pi * (a[0] +
                                          b[0]) * np.sqrt((a[0] - b[0])**2 +
                                                          (a[1] - b[1])**2)
        if named_partition is None:
            return self._areas
        else:
            partition = self.named_partition[named_partition]
            return self._areas[partition[0]:partition[1]]

    @classmethod
    def square(cls, w = 100.):
        v, e = geometry.square(w)
        return Region(v, e)


    def dirichlet_boundary_condition(self):
        """Returns a boundary condition with alpha the 1-function and f and beta 0-functions."""
        bc = BoundaryCondition(self.len())
        bc.alpha.fill(1.0)
        bc.beta.fill(0.0)
        bc.f.fill(1.0)
        return bc

    def neumann_boundary_condition(self):
        """Returns a boundary condition with f and alpha 0-functions and beta the 1-function."""
        bc = BoundaryCondition(self.len())
        bc.alpha.fill(0.0)
        bc.beta.fill(1.0)
        bc.f.fill(0.0)
        return bc


class BoundaryCondition:

    def __init__(self, size):
        self.alpha = np.empty(size, dtype=np.complex64)
        self.beta = np.empty(size, dtype=np.complex64)
        self.f = np.empty(size, dtype=np.complex64)

    def __repr__(self):
        result = self.__class__.__name__ + "("
        result += "alpha = {}, ".format(self.alpha)
        result += "beta = {}, ".format(self.beta)
        result += "f = {})".format(self.f)
        return result
