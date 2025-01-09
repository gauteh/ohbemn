import numpy as np

from .intop import l_2d, m_2d, mt_2d, n_2d
from . import wave


class Solver:

    def __init__(self, region):
        self.region = region

    def len(self):
        return self.region.len()

    def compute_boundary_matrices(self, k, mu, orientation):
        A = np.empty((self.len(), self.len()), dtype=np.complex64)
        B = np.empty(A.shape, dtype=np.complex64)

        centers = self.region.centers()
        normals = self.region.normals()

        for i in range(self.len()):
            center = centers[i]
            normal = normals[i]

            for j in range(self.len()):
                qa, qb = self.region.edge(j)

                element_l = np.atleast_1d(
                    l_2d(k, center.astype(np.float64), qa.astype(np.float64),
                         qb.astype(np.float64), i == j))[0]
                element_m = np.atleast_1d(m_2d(k, center, qa, qb, i == j))[0]
                element_mt = np.atleast_1d(
                    mt_2d(k, center, normal, qa, qb, i == j))[0]
                element_n = np.atleast_1d(
                    n_2d(k, center, normal, qa, qb, i == j))[0]

                mu = np.atleast_1d(mu)[0]

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

    def solve_boundary(self,
                       orientation,
                       k,
                       celerity,
                       boundary_condition,
                       boundary_incidence,
                       mu=None):
        mu = mu or (1j / (k + 1))
        mu = np.atleast_1d(mu)[0]
        assert boundary_condition.f.size == self.len()
        A, B = self.compute_boundary_matrices(k, mu, orientation)
        c = np.empty(self.len(), dtype=np.complex64)
        for i in range(self.len()):
            c[i] = boundary_incidence.phi[i] + mu * boundary_incidence.v[i]

        if 'exterior' == orientation:
            c = -1.0 * c
        else:
            assert 'interior' == orientation, "orientation must be either 'interior' or 'exterior'"

        phi, v = self.solve_linear_equation(B, A, c, boundary_condition.alpha,
                                            boundary_condition.beta,
                                            boundary_condition.f)

        return BoundarySolution(self, orientation, boundary_condition, k,
                                celerity, phi, v)

    def solve_linear_equation(self, Ai, Bi, ci, alpha, beta, f):
        A = np.copy(Ai)
        B = np.copy(Bi)
        c = np.copy(ci)

        x = np.empty(c.size, dtype=np.complex64)
        y = np.empty(c.size, dtype=np.complex64)

        gamma = np.linalg.norm(B, np.inf) / np.linalg.norm(A, np.inf)
        swapXY = np.empty(c.size, dtype=bool)
        for i in range(c.size):
            if np.abs(beta[i]) < gamma * np.abs(alpha[i]):
                swapXY[i] = False
            else:
                swapXY[i] = True

        for i in range(c.size):
            if swapXY[i]:
                for j in range(alpha.size):
                    c[j] += f[i] * B[j, i] / beta[i]
                    B[j, i] = -alpha[i] * B[j, i] / beta[i]
            else:
                for j in range(alpha.size):
                    c[j] -= f[i] * A[j, i] / alpha[i]
                    A[j, i] = -beta[i] * A[j, i] / alpha[i]

        A -= B
        y = np.linalg.solve(A, c)

        for i in range(c.size):
            if swapXY[i]:
                x[i] = (f[i] - alpha[i] * y[i]) / beta[i]
            else:
                x[i] = (f[i] - beta[i] * y[i]) / alpha[i]

        for i in range(c.size):
            if swapXY[i]:
                temp = x[i]
                x[i] = y[i]
                y[i] = temp

        return x, y

    def solve_samples(self, solution, incident_phis, samples, orientation):
        assert incident_phis.shape == samples.shape[:-1], \
            "Incident phi vector and sample points vector must match"

        result = np.empty(samples.shape[0], dtype=complex)

        for i in range(incident_phis.size):
            p = samples[i]
            sum = incident_phis[i]
            for j in range(solution.phis.size):
                qa, qb = self.region.edge(j)

                element_l = l_2d(solution.k, p.astype(np.float64),
                                 qa.astype(np.float64), qb.astype(np.float64),
                                 False)
                element_m = m_2d(solution.k, p, qa, qb, False)

                if orientation == 'interior':
                    sum += element_l * solution.velocities[
                        j] - element_m * solution.phis[j]
                elif orientation == 'exterior':
                    sum -= element_l * solution.velocities[
                        j] - element_m * solution.phis[j]
                else:
                    assert False, 'Invalid orientation: {}'.format(orientation)
            sum = np.atleast_1d(sum)
            assert len(sum) == 1
            result[i] = sum[0]

        return result


class BoundarySolution:

    def __init__(self, solver, orientation, boundary_condition, k, celerity,
                 phis, velocities):
        self.solver = solver
        self.boundary_condition = boundary_condition
        self.k = k
        self.c = celerity
        self.phis = phis
        self.velocities = velocities
        self.orientation = orientation

    def __repr__(self):
        result = self.__class__.__name__ + "("
        result += "solver = " + repr(self.solver) + ", "
        result += "boundary_condition = " + repr(
            self.boundary_condition) + ", "
        result += "k = " + repr(self.k) + ", "
        result += "aPhi = " + repr(self.phis) + ", "
        result += "aV = " + repr(self.velocities) + ")"
        return result

    def __str__(self):
        res = "k:      {} 1/m\n".format(self.k)
        res = "c:      {} m/s\n".format(self.c)
        res += "index   Potential               eta\n\n"
        for i in range(self.phis.size):
            eta = wave.eta(self.phis[i], self.k, self.c)
            res += f"{i+1} {self.phis[i]} {eta}\n"
            # print(eta)
            # print(self.phis[i])
            # res += "{:5d}  {:1.4e}{:+1.4e}  {:1.4e}{:+1.4e} \n".format(
            #     i + 1,
            #     self.phis[i].real,
            #     self.phis[i].imag,
            #     eta.real,
            #     eta.imag,
            # )
        return res

    def eta(self):
        return wave.eta(self.phis, self.k, self.c)

    def solve_samples(self, incident_phis, points):
        return SampleSolution(
            self,
            self.solver.solve_samples(self, incident_phis, points,
                                      self.orientation))


class SampleSolution:

    def __init__(self, boundary_solution, phis):
        self.boundarySolution = boundary_solution
        self.phis = phis

    def eta(self):
        return wave.eta(self.phis, self.boundarySolution.k,
                        self.boundarySolution.c)

    def __repr__(self):
        result = "SampleSolution("
        result += "boundarySolution = " + repr(self.boundarySolution) + ", "
        result += "aPhi = " + repr(self.phis) + ")"
        return result

    def __str__(self):
        result = "index   Potential                eta\n\n"
        for i in range(self.phis.size):
            eta = wave.eta(self.boundarySolution.k,
                           self.phis[i],
                           c=self.boundarySolution.c)

            result += f"{i+1} {self.phis[i]} {eta}\n"
            # result += "{:5d}  {: 1.4e}{:+1.4e}i  {: 1.4e}{:+1.4e}i\n".format( \
            #     i+1, self.phis[i].real, self.phis[i].imag, eta.real, eta.imag, )

        return result
