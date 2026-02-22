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
    print(boundary_solution.phis)

    phis = [
        0.79417443 + 1.5951941e-02j, 0.43684256 + 3.8052818e-05j,
        0.40029442 - 1.3810139e-03j, 0.21927303 - 2.4244445e-03j,
        -0.12927234 - 3.7010214e-03j, -0.12927246 - 3.7010799e-03j,
        0.21927309 - 2.4245677e-03j, 0.40029448 - 1.3811738e-03j,
        0.43684262 + 3.7951326e-05j, 0.79417443 + 1.5951961e-02j,
        0.79417443 + 1.5951861e-02j, 0.43684256 + 3.8041930e-05j,
        0.4002943 - 1.3809397e-03j, 0.21927297 - 2.4243901e-03j,
        -0.12927246 - 3.7010701e-03j, -0.12927246 - 3.7011108e-03j,
        0.21927303 - 2.4245181e-03j, 0.40029436 - 1.3810910e-03j,
        0.43684262 + 3.7954895e-05j, 0.79417443 + 1.5951859e-02j,
        0.79417443 + 1.5951885e-02j, 0.43684262 + 3.7905029e-05j,
        0.40029448 - 1.3810853e-03j, 0.21927309 - 2.4245125e-03j,
        -0.12927246 - 3.7011446e-03j, -0.12927246 - 3.7010754e-03j,
        0.21927303 - 2.4243903e-03j, 0.40029442 - 1.3809482e-03j,
        0.43684262 + 3.8084621e-05j, 0.79417443 + 1.5951850e-02j,
        0.79417443 + 1.5951950e-02j, 0.43684268 + 3.7923543e-05j,
        0.40029442 - 1.3811762e-03j, 0.21927303 - 2.4245721e-03j,
        -0.12927246 - 3.7010605e-03j, -0.12927246 - 3.7010056e-03j,
        0.21927303 - 2.4244385e-03j, 0.40029436 - 1.3810548e-03j,
        0.43684256 + 3.8082184e-05j, 0.7941744 + 1.5951913e-02j
    ]

    np.testing.assert_allclose(boundary_solution.phis, phis, atol=1e-7)


def test_ohrs_solve_boundary_rectangle_neumann(benchmark):
    # https://github.com/lzhw1991/AcousticBEM/blob/master/Jupyter/Rectangular%20Interior%20Helmholtz%20Problems.ipynb
    f = 1 / 5.  # [Hz]
    T = 1 / f
    d = 40.  # [m]
    c, cg, k = ohpy.wave.wavec_interm(T, d)
    print("wave length:", c / f)

    regionp = ohpy.Region.rectangle(100, 100, 10, 10)  # XXX
    region = ohrs.Region(regionp.vertices.astype(np.float64),
                         regionp._edges.astype(np.uint64))
    print("elements:", region.len())

    np.testing.assert_allclose(regionp.vertices, region.vertices())
    np.testing.assert_allclose(regionp.edges(), region.edges())

    region = ohrs.Region.rectangle(100, 100, 10, 10)  # XXX
    np.testing.assert_allclose(regionp.vertices, region.vertices())
    np.testing.assert_allclose(regionp.edges(), region.edges())

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
    solver = ohrs.Solver(region)
    boundary_solution = benchmark(solver.solve_boundary,
                                  ohrs.Orientation.Interior, k, c, bc, bi)
    print(boundary_solution.phis)

    phis = [
        0.79417443 + 1.5951941e-02j, 0.43684256 + 3.8052818e-05j,
        0.40029442 - 1.3810139e-03j, 0.21927303 - 2.4244445e-03j,
        -0.12927234 - 3.7010214e-03j, -0.12927246 - 3.7010799e-03j,
        0.21927309 - 2.4245677e-03j, 0.40029448 - 1.3811738e-03j,
        0.43684262 + 3.7951326e-05j, 0.79417443 + 1.5951961e-02j,
        0.79417443 + 1.5951861e-02j, 0.43684256 + 3.8041930e-05j,
        0.4002943 - 1.3809397e-03j, 0.21927297 - 2.4243901e-03j,
        -0.12927246 - 3.7010701e-03j, -0.12927246 - 3.7011108e-03j,
        0.21927303 - 2.4245181e-03j, 0.40029436 - 1.3810910e-03j,
        0.43684262 + 3.7954895e-05j, 0.79417443 + 1.5951859e-02j,
        0.79417443 + 1.5951885e-02j, 0.43684262 + 3.7905029e-05j,
        0.40029448 - 1.3810853e-03j, 0.21927309 - 2.4245125e-03j,
        -0.12927246 - 3.7011446e-03j, -0.12927246 - 3.7010754e-03j,
        0.21927303 - 2.4243903e-03j, 0.40029442 - 1.3809482e-03j,
        0.43684262 + 3.8084621e-05j, 0.79417443 + 1.5951850e-02j,
        0.79417443 + 1.5951950e-02j, 0.43684268 + 3.7923543e-05j,
        0.40029442 - 1.3811762e-03j, 0.21927303 - 2.4245721e-03j,
        -0.12927246 - 3.7010605e-03j, -0.12927246 - 3.7010056e-03j,
        0.21927303 - 2.4244385e-03j, 0.40029436 - 1.3810548e-03j,
        0.43684256 + 3.8082184e-05j, 0.7941744 + 1.5951913e-02j
    ]

    np.testing.assert_allclose(boundary_solution.phis, phis, atol=1e-4)


def test_ohrs_acousticbem_interior_rectangle(benchmark):
    # https://github.com/lzhw1991/AcousticBEM/blob/master/Jupyter/Rectangular%20Interior%20Helmholtz%20Problems.ipynb
    f = 1 / 5.  # [Hz]
    T = 1 / f
    d = 40.  # [m]
    c, cg, k = ohpy.wave.wavec_interm(T, d)  # XXX
    print("wave length:", c / f)

    region = ohpy.Region.rectangle(100, 100, 10, 10)  # XXX
    region = ohrs.Region(region.vertices.astype(np.float64),
                         region._edges.astype(np.uint64))
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

    ip = np.vstack((xx, yy)).T.astype(np.float64)
    print("Interior points:", ip.shape[0])
    i_incident = np.zeros(ip.shape[0], dtype=np.complex128)

    # Ready to solve!

    solver = ohrs.Solver(region)
    boundary_solution = solver.solve_boundary(ohrs.Orientation.Interior, k, c,
                                              bc, bi)

    # Now we can solve the field at the interior points:
    print(i_incident, type(i_incident))
    print(i_incident.dtype, i_incident.shape)
    interior = benchmark(boundary_solution.solve_samples, i_incident, ip)
    print(interior)


def test_ohpy_acousticbem_interior_rectangle(benchmark):
    # https://github.com/lzhw1991/AcousticBEM/blob/master/Jupyter/Rectangular%20Interior%20Helmholtz%20Problems.ipynb
    f = 1 / 5.  # [Hz]
    T = 1 / f
    d = 40.  # [m]
    c, cg, k = ohpy.wave.wavec_interm(T, d)  # XXX
    print("wave length:", c / f)

    region = ohpy.Region.rectangle(100, 100, 10, 10)  # XXX
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

    ip = np.vstack((xx, yy)).T.astype(np.float64)
    print("Interior points:", ip.shape[0])
    i_incident = np.zeros(ip.shape[0], dtype=np.complex128)

    # Ready to solve!

    solver = ohpy.Solver(region)
    boundary_solution = solver.solve_boundary('interior', k, c, bc, bi)

    # Now we can solve the field at the interior points:
    print(i_incident, type(i_incident))
    print(i_incident.dtype, i_incident.shape)
    interior = benchmark(boundary_solution.solve_samples, i_incident, ip)
    print(interior)
