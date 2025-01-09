import numpy as np
import matplotlib.pyplot as plt

from ohbemn import wave, Solver, Region, Orientation

#%% Wave parameters
f = 1 / 5.  # [Hz]
T = 1 / f
d = 40.  # [m]
c, cg, k = wave.wavec_interm(T, d)
print("wave length:", c / f)

region = Region.rectangle(100, 100, 32, 32)

f, ax = plt.subplots()
region.plot(ax)

print("elements:", region.len())

#%% Specifying boundary conditions
bc = region.boundary_condition()
bc.alpha.fill(1.0)
bc.beta.fill(1.0)
bc.f.fill(0.5j)
bc.f[0:(2 * 32)].fill(1.0)
# self.boundaryCondition.f[ 0] = 1.0
# self.boundaryCondition.f[-1] = 1.0
print(bc.alpha)
print(bc.beta)
print(bc.f)

# definition of incident fields on boundary
bi = region.boundary_incidence()
bi.phi.fill(0.0)  # no incoming velocity potential on boundary
bi.v.fill(0.0)  # no incoming velocity on boundary

print(bi.phi)
print(bi.v)

#%% Interior grid where we want to know the field
xi = np.arange(0, 100, 3)
yi = np.arange(0, 100, 3)
xx, yy = np.meshgrid(xi, yi)
xx, yy = xx.ravel(), yy.ravel()

ip = np.vstack((xx, yy)).T
print("Interior points:", ip.shape[0])
i_incident = np.zeros(ip.shape[0])

#%% Ready to solve!
solver = Solver(region)
boundary_solution = solver.solve_boundary(Orientation.Interior, k[0], c[0], bc, bi)
print(boundary_solution.phis)

#%% Now we can solve the field at the interior points:
interior = boundary_solution.solve_samples(i_incident, ip)

#%% And plot:
f, ax = plt.subplots()
region.plot(ax)

ax.imshow(interior.phis.reshape((len(xi), len(yi))).real,
          extent=[0, 100, 0, 100],
          origin='lower')

ax.set_xlim([-20, 120])
ax.set_ylim([-20, 120])
ax.set_title('$\\Re\{\\phi\}$')

#%% Show
plt.show()
f.savefig('figures/rectangular_interior.png', dpi=250)
