"""
Coupled subregions of constant depth (ohbemn.coupled), long waves.

(a) A circular sill (Longuet-Higgins 1967) at its first trapped mode for n = 2.
(b) The paraboloid pit of Suh et al. (2005) as 8 rings of constant depth.
(c) An elliptic shoal with a rigid island in it, and a deeper pit.

Every Boundary is a closed polygon. A Subregion is the area inside `outer` and outside `holes`,
the Exterior is the area outside its `holes` and carries the incident wave. A boundary that
belongs to only one subregion is a wall (alpha phi + beta dphi/dn = f, by default dphi/dn = 0).
"""
import numpy as np
import matplotlib.pyplot as plt

from ohbemn import coupled
from ohbemn.ohpy.wave import g


def grid(L, n=201):
    x = np.linspace(-L, L, n)
    xx, yy = np.meshgrid(x, x)
    return x, np.c_[xx.ravel(), yy.ravel()]


def show(ax, x, phi, boundaries, title):
    im = ax.pcolormesh(x, x, np.abs(phi).reshape(len(x), len(x)), shading='auto', cmap='viridis')
    for b in boundaries:
        v = b.vertices[np.r_[b.edges[:, 0], b.edges[0, 0]].astype(int)]
        ax.plot(*v.T, color='w', lw=.6)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, shrink=.7, label='$|\\varphi|$')


fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

# (a) circular sill, depth 10 m in 160 m, radius 1000 m, z = k1 a = 4.922
a, h1, h2, z = 1000., 10., 160., 4.922
T = 2 * np.pi * a / (z * np.sqrt(g * h1))
rim = coupled.Boundary.circle(a, 256)
p = coupled.Problem(T, [coupled.Subregion(h1, outer=rim), coupled.Exterior(h2, holes=[rim])])
x, pts = grid(3 * a)
show(axs[0], x, p.solve(direction=0.).field(pts), [rim],
     f'(a) Sill, $h_1/h_2 = 1/16$, $z = {z}$, $T = {T:.0f}$ s')

# (b) paraboloid pit h = h0 (1 - r^2 / a^2) for r < b, as M rings of constant depth
d1, h0, M = 3.2, 6.4, 8
k1 = 0.167 / d1
T = 2 * np.pi / (k1 * np.sqrt(g * d1))
b = np.pi / k1
ab = b / np.sqrt(1 - d1 / h0)
re = b * np.arange(1, M + 1) / M
ri = np.r_[0, re[:-1]]
depth = h0 * (1 - (ri**2 + re**2) / (2 * ab**2))  # mean depth over each ring
rings = [coupled.Boundary.circle(r, 128) for r in re]
subs = [coupled.Subregion(depth[0], outer=rings[0])]
subs += [coupled.Subregion(depth[j], outer=rings[j], holes=[rings[j - 1]]) for j in range(1, M)]
subs += [coupled.Exterior(d1, holes=[rings[-1]])]
x, pts = grid(4 * b)
show(axs[1], x, coupled.Problem(T, subs).solve(direction=0.).field(pts), rings,
     f'(b) Pit as {M} rings (Suh et al. 2005), $T = {T:.0f}$ s')

# (c) elliptic shoal (5 m) with a rigid island, and a pit (40 m), in 20 m
t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
shoal = coupled.Boundary.from_xy(1500 * np.cos(t), 800 * np.sin(t))
island = coupled.Boundary.from_xy(300 + 250 * np.cos(t[::2]), 150 * np.sin(t[::2]))  # wall
pit = coupled.Boundary.from_xy(-2500 + 400 * np.cos(t[::2]), 1500 + 400 * np.sin(t[::2]))
T = 120.
p = coupled.Problem(T, [coupled.Subregion(5., outer=shoal, holes=[island]),
                        coupled.Subregion(40., outer=pit),
                        coupled.Exterior(20., holes=[shoal, pit])])
x, pts = grid(4000)
show(axs[2], x, p.solve(direction=np.deg2rad(-30)).field(pts), [shoal, island, pit],
     f'(c) Shoal with an island, and a pit, $T = {T:.0f}$ s')

for ax in axs:
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
fig.savefig('figures/coupled_regions.png', dpi=150)
plt.show()
