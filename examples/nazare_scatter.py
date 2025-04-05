import numpy as np
import matplotlib.pyplot as plt

from ohbemn import Solver, Region, ohpy, Orientation, style_plots
from ohbemn.ohpy import wave, source

style_plots()

T = 8   # [s]
f = 1/T
d = 40.  # [m]
c, cg, k = wave.wavec_interm(T, d)
print("wave length:", c/f)

print(k)
kx = k/3
assert kx <= k, k

ky = np.sqrt(k**2 - kx**2)

print("k:", np.sqrt(kx**2 + ky**2), k, "kx=", kx, "ky=", ky)


region = Region.rectangle(400, 1000, 32, 32)

vertices = region.vertices()
vertices[(32-8):32,0] = np.linspace(0, -50, 8)

region = Region(vertices, region.edges2d())

f, (ax1, ax2,) = plt.subplots(1,2, figsize=(4,8))
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
    # bc.f[i] = phi

    bi.phi[i] = phi
    # bi.v[i] = v

# Sample grid in the exterior
x = np.arange(-400, 0, 10)
y = np.arange(-2000, 2000, 10)
xx, yy = np.meshgrid(x, y)


# Incident field without region
fI = source.plane(kx, ky, xx, yy)

region.plot(ax1)
ax1.set_xlim([-500, 800])
ax1.set_ylim([-2300, 2300])
ax1.set_aspect('equal')
ax1.pcolor(xx, yy, fI.real, vmin=-1, vmax=1)
ax1.set_title('Incident field')

# Solve boundary.
solver = Solver(region)
boundary_solution = solver.solve_boundary(Orientation.Exterior, k+0.01j, c, bc, bi)

# Solve samples in interior
ep = np.vstack((xx.ravel(), yy.ravel())).T

print('Solving for exterior points:', ep.shape)
eF = boundary_solution.solve_samples(np.zeros(ep.shape[0]), ep)

eFp = eF.phis.reshape(xx.shape)
region.plot(ax2)
ax2.pcolor(xx, yy, eFp.real, vmin=-1, vmax=1)
ax2.set_xlim([-500, 800])
ax2.set_ylim([-2300, 2300])
ax2.set_aspect('equal')
ax2.set_title('Computed exterior (Dirichlet)')

##% Entire field with external and internal field.
# region.plot(ax3)
# ex = np.arange(-500, 800, 10)
# ey = np.arange(-2300, 2300, 10)
# exx, eyy = np.meshgrid(ex, ey)
# efI = source.plane(kx, ky, exx, eyy)
# # ax3.pcolor(exx, eyy, efI.real, vmin=-1, vmax=1, zorder=1)
# ax3.pcolor(xx, yy, eFp.real, vmin=-1, vmax=1, zorder=2)
# ax3.set_xlim([-500, 800])
# ax3.set_ylim([-2300, 2300])
# ax3.set_aspect('equal')
# ax3.set_title('Incident with computed exterior region on top')

#%% Plot
f.suptitle('Deep region $\\Re\\{\\phi\\}$')
f.savefig('figures/nazare_scatter.png', dpi=250)
plt.show()


