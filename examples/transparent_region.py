import numpy as np
import matplotlib.pyplot as plt

from ohbemn import Solver, Region, ohpy, Orientation, style_plots
from ohbemn.ohpy import wave, source

style_plots()

T = 8.   # [s]
f = 1/T
d = 40.  # [m]
c, cg, k = wave.wavec_interm(T, d)
print("wave length:", c/f)

print(k)
kx = k/2
assert kx <= k, k

ky = np.sqrt(k**2 - kx**2)

print("k:", np.sqrt(kx**2 + ky**2), k, "kx=", kx, "ky=", ky)

region = Region.rectangle(200, 200, 32, 32)

f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(12,8))
# region.plot(ax)
# ax.set_xlim([-200, 400])
# ax.set_ylim([-200, 400])
# ax.set_aspect('equal')

print("elements:", region.len())

##% Dirichlet
bc = region.boundary_condition()

centers = region.centers()
normals = region.normals()
bi = region.boundary_incidence()

# Define incidence field and boundary conditions
for i in range(region.len()):
    x = centers[i, :]
    n = normals[i, :]

    phi = source.plane(kx, ky, *x)[0]
    v = source.dplane(kx, ky, *x, -n)[0]

    bc.alpha[i] = 1.
    bc.beta[i] = 0.
    bc.f[i] = phi

    # bi.phi[i] = phi
    # bi.v[i] = v

# Sample grid in the interior

x = np.arange(0.01, 200, 10)
y = np.arange(0.01, 200, 10)
xx, yy = np.meshgrid(x, y)


# Incident field without region
fI = source.plane(kx, ky, xx, yy)

region.plot(ax1)
ax1.set_xlim([-200, 400])
ax1.set_ylim([-200, 400])
ax1.set_aspect('equal')
ax1.pcolor(xx, yy, fI.real, vmin=-1, vmax=1)
ax1.set_title('Incident field on interior region')

# Solve boundary.
solver = Solver(region)
boundary_solution = solver.solve_boundary(Orientation.Interior, k, c, bc, bi)

# Solve samples in interior
ep = np.vstack((xx.ravel(), yy.ravel())).T

print('Solving for interior points:', ep.shape)
eF = boundary_solution.solve_samples(np.zeros(ep.shape[0]), ep)

eFp = eF.phis.reshape(xx.shape)
region.plot(ax2)
ax2.pcolor(xx, yy, eFp.real, vmin=-1, vmax=1)
ax2.set_xlim([-200, 400])
ax2.set_ylim([-200, 400])
ax2.set_aspect('equal')
ax2.set_title('Computed interior region (Dirichlet)')

##% Entire field with external and internal field.
region.plot(ax3)
ex = np.arange(-200, 400, 10)
ey = np.arange(-200, 400, 10)
exx, eyy = np.meshgrid(ex, ey)
efI = source.plane(kx, ky, exx, eyy)
ax3.pcolor(exx, eyy, efI.real, vmin=-1, vmax=1, zorder=1)
ax3.pcolor(xx, yy, eFp.real, vmin=-1, vmax=1, zorder=2)
ax3.set_xlim([-200, 400])
ax3.set_ylim([-200, 400])
ax3.set_aspect('equal')
ax3.set_title('Entire region + transparent region')

#%% Neumann
bc = region.boundary_condition()

centers = region.centers()
normals = region.normals()
bi = region.boundary_incidence()

# Define incidence field and boundary conditions
for i in range(region.len()):
    x = centers[i, :]
    n = normals[i, :]

    phi = source.plane(kx, ky, *x)[0]
    v = source.dplane(kx, ky, *x, -n)[0]

    bc.alpha[i] = 0.
    bc.beta[i] = 1.
    bc.f[i] = v

    # bi.phi[i] = phi
    # bi.v[i] = v

region.plot(ax4)
ax4.set_xlim([-200, 400])
ax4.set_ylim([-200, 400])
ax4.set_aspect('equal')
ax4.pcolor(xx, yy, fI.real, vmin=-1, vmax=1)
ax4.set_title('Incident field on interior region')

# Solve boundary.
solver = Solver(region)
boundary_solution = solver.solve_boundary(Orientation.Interior, k, c, bc, bi)

# Solve samples in interior
ep = np.vstack((xx.ravel(), yy.ravel())).T

print('Solving for interior points:', ep.shape)
eF = boundary_solution.solve_samples(np.zeros(ep.shape[0]), ep)

eFp = eF.phis.reshape(xx.shape)
region.plot(ax5)
ax5.pcolor(xx, yy, eFp.real, vmin=-1, vmax=1)
ax5.set_xlim([-200, 400])
ax5.set_ylim([-200, 400])
ax5.set_aspect('equal')
ax5.set_title('Computed interior region (Neumann)')

##% Entire field with external and internal field.
region.plot(ax6)
ex = np.arange(-200, 400, 10)
ey = np.arange(-200, 400, 10)
exx, eyy = np.meshgrid(ex, ey)
efI = source.plane(kx, ky, exx, eyy)
ax6.pcolor(exx, eyy, efI.real, vmin=-1, vmax=1, zorder=1)
ax6.pcolor(xx, yy, eFp.real, vmin=-1, vmax=1, zorder=2)
ax6.set_xlim([-200, 400])
ax6.set_ylim([-200, 400])
ax6.set_aspect('equal')
ax6.set_title('Entire region + transparent region')

#%% Plot
f.suptitle('Transparent region $\\Re\\{\\phi\\}$')
f.savefig('figures/transparent.png', dpi=250)
plt.show()
