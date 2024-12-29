import numpy as np


class Region:
    vertices: np.ndarray
    edges: np.ndarray

    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges


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
