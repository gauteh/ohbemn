import numpy as np

from boundary import BoundaryCondition


class HelmholtzSolver:

    def __init__(self, region):
        self.region = region

    def len(self):
        return self.region.len()

    def compute_boundary_matrices(region, k, mu, orientation):
        A = np.empty((self.len(), self.len()), dtype=np.complex64)
        B = np.empty(A.shape, dtype=np.complex64)

        centers = self.geometry.centers()
        normals = self.geometry.normals()

        for i in range(self.len()):
            center = centers[i]
            normal = normals[i]
            for j in range(self.len()):
                qa, qb = self.geometry.edge(j)

                element_l = l_2d(k, center, qa, qb, i == j)
                element_m = m_2d(k, center, qa, qb, i == j)
                element_mt = mt_2d(k, center, normal, qa, qb, i == j)
                element_n = n_2d(k, center, normal, qa, qb, i == j)

                A[i, j] = element_l + mu * element_mt
                B[i, j] = element_m + mu * element_n

            if orientation == 'interior':
                # interior variant, signs are reversed for exterior
                A[i, i] -= 0.5 * mu
                B[i, i] += 0.5
            elif orientation == 'exterior':
                A[i, i] += 0.5 * mu
                B[i, i] -= 0.5
            else:
                assert False, 'Invalid orientation: {}'.format(orientation)

        return A, B

    def __repr__(self):
        result = self.__class__.__name__ + "("
        result += "  Region = " + repr(self.region) + ")"
        return result

    def len(self):
        return self.region.len()

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
