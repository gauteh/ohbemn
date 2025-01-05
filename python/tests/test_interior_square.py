import numpy as np
import matplotlib.pyplot as plt

import ohbemn as ohrs
from ohbemn import ohpy


def test_ohpy_solve_boundary_rectangle_neumann(benchmark):
    # https://github.com/lzhw1991/AcousticBEM/blob/master/Jupyter/Rectangular%20Interior%20Helmholtz%20Problems.ipynb
    f = 1 / 5.  # [Hz]
    T = 1 / f
    d = 40.  # [m]
    c, cg, k = ohpy.wave.wavec_interm(T, d)
    print("wave length:", c / f)

    region = ohpy.Region.rectangle(100, 100, 10, 10)
    print("elements:", region.len())

    # Specifying boundary conditions

    bc = region.boundary_condition()
    bc.alpha.fill(1.0)
    bc.beta.fill(1.0)
    bc.f.fill(0.5j)
    bc.f[0:(2 * 32)].fill(1.0)
    # self.boundaryCondition.f[ 0] = 1.0
    # self.boundaryCondition.f[-1] = 1.0

    # definition of incident fields on boundary

    bi = region.boundary_incidence()
    bi.phi.fill(0.0)  # no incoming velocity potential on boundary
    bi.v.fill(0.0)  # no incoming velocity on boundary

    # Ready to solve!
    solver = ohpy.Solver(region)
    boundary_solution = benchmark(solver.solve_boundary, 'interior', k, c, bc,
                                  bi)
    print(boundary_solution)

def test_ohpy_solve_boundary_rectangle_neumann(benchmark):
    # https://github.com/lzhw1991/AcousticBEM/blob/master/Jupyter/Rectangular%20Interior%20Helmholtz%20Problems.ipynb
    f = 1 / 5.  # [Hz]
    T = 1 / f
    d = 40.  # [m]
    c, cg, k = ohpy.wave.wavec_interm(T, d)
    print("wave length:", c / f)

    region = ohpy.Region.rectangle(100, 100, 10, 10)
    print("elements:", region.len())

    # Specifying boundary conditions

    bc = region.boundary_condition()
    bc.alpha.fill(1.0)
    bc.beta.fill(1.0)
    bc.f.fill(0.5j)
    bc.f[0:(2 * 32)].fill(1.0)
    # self.boundaryCondition.f[ 0] = 1.0
    # self.boundaryCondition.f[-1] = 1.0

    # definition of incident fields on boundary

    bi = region.boundary_incidence()
    bi.phi.fill(0.0)  # no incoming velocity potential on boundary
    bi.v.fill(0.0)  # no incoming velocity on boundary

    # Ready to solve!
    solver = ohpy.Solver(region)
    boundary_solution = benchmark(solver.solve_boundary, 'interior', k, c, bc,
                                  bi)
    print(boundary_solution)

def test_ohrs_acousticbem_interior_rectangle():
    # https://github.com/lzhw1991/AcousticBEM/blob/master/Jupyter/Rectangular%20Interior%20Helmholtz%20Problems.ipynb
    f = 1 / 5.  # [Hz]
    T = 1 / f
    d = 40.  # [m]
    c, cg, k = ohpy.wave.wavec_interm(T, d) # XXX
    print("wave length:", c / f)

    region = ohpy.Region.rectangle(100, 100, 10, 10) # XXX
    region = ohrs.Region(region.vertices.astype(np.float64), region._edges.astype(np.uint64))
    print("elements:", region.len())

    # Specifying boundary conditions

    bc = region.boundary_condition()
    bc.alpha.fill(1.0)
    bc.beta.fill(1.0)
    bc.f.fill(0.5j)
    bc.f[0:(2 * 32)].fill(1.0)
    # self.boundaryCondition.f[ 0] = 1.0
    # self.boundaryCondition.f[-1] = 1.0

    # definition of incident fields on boundary

    bi = region.boundary_incidence()
    bi.phi.fill(0.0)  # no incoming velocity potential on boundary
    bi.v.fill(0.0)  # no incoming velocity on boundary

    # Interior grid where we want to know the field
    xi = np.arange(0, 100, 30)
    yi = np.arange(0, 100, 30)
    xx, yy = np.meshgrid(xi, yi)
    xx, yy = xx.ravel(), yy.ravel()

    ip = np.vstack((xx, yy)).T
    print("Interior points:", ip.shape[0])
    i_incident = np.zeros(ip.shape[0])

    # Ready to solve!

    solver = ohrs.Solver(region)
    boundary_solution = solver.solve_boundary('interior', k, c, bc, bi)

    # Now we can solve the field at the interior points:
    interior = boundary_solution.solve_samples(i_incident, ip)
