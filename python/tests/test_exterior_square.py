import numpy as np
from numpy.linalg import norm
from scipy.special import hankel1
import matplotlib.pyplot as plt

from ohbemn import Region, wave, Solver


def test_exterior_abem_dirichlet():
    # emulate the abem exterior 2d example
    # https://github.com/fjargsto/abem/blob/master/notebooks/exterior_helmholtz_solver_2d.ipynb
    region = Region.square(0.1)

    f = 400.0  # Hz
    c = 344.0  # speed of sound in air
    k = 2.0 * np.pi * f / c

    center_square = np.array([0.05, 0.05], dtype=np.float32)

    solver = Solver(region)

    boundary_condition = region.dirichlet_boundary_condition()
    centers = solver.region.centers()
    for i in range(centers.shape[0]):
        r = norm(centers[i, :] - center_square)
        boundary_condition.f[i] = 1j * hankel1(0, k * r) / 4.0

    boundary_incidence = region.boundary_incidence()
    boundary_incidence.phi.fill(0.0)
    boundary_incidence.v.fill(0.0)

    exterior_points = np.array([[0.0000, 0.1500], [0.0500, 0.1500],
                                [0.1000, 0.1500], [0.0500, -0.1000]],
                               dtype=np.float32)

    exterior_incident_phi = np.zeros(exterior_points.shape[0],
                                     dtype=np.complex64)

    boundary_solution = solver.solve_boundary('exterior', k, c,
                                              boundary_condition,
                                              boundary_incidence)
    sample_solution = boundary_solution.solve_samples(exterior_incident_phi,
                                                      exterior_points)
    print("\n\nTest Problem 1")
    print("==============\n")
    print(boundary_solution)
    print(sample_solution)

    abem_potential = np.array([
        1.9757e-02 + 2.0788e-01j, 4.1453e-02 + 2.1538e-01j,
        1.9757e-02 + 2.0788e-01j, -3.7472e-02 + 1.7919e-01j
    ])

    np.testing.assert_allclose(sample_solution.phis,
                               abem_potential,
                               atol=0.00001)
