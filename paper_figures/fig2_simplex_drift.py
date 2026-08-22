"""Figure 2 — where the measured composition lands relative to the 1:1:1 truth line.

The stacked-bar version answers "what fraction", not "how far off". A composition is a
point in DQ/TBZ/THI space and the dispensed 1:1:1 truth is a LINE through it (the ray
x = y = z). Every normalised composition projects onto that ray at the centroid, so the
dotted drop-line from each point IS its perpendicular distance to the truth line, and
theta is the angle it subtends. 54.7 deg — one pure compound — is the geometric maximum.

House style: Arial 18 pt throughout, black text, vertex colours and the THI-top /
TBZ-left / DQ-right orientation matched to the ternary panels of the composite figure.

Edit DATA below to point at the real per-map numbers.
Run:  python paper_figures/fig2_simplex_drift.py
"""
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = Path(__file__).resolve().parent
DQ, TBZ, THI = "#1a73e8", "#27a679", "#e8467c"
INK, AXIS, PLANE, GUIDE = "#000000", "#8f9aa6", "#dde5ee", "#8c8c8c"
PT = 18                                                  # one type size, everywhere

# concentration (uM) -> composition percentages (DQ, TBZ, THI)
DATA = [(10, (30, 23, 47)), (30, (18, 34, 48)), (100, (0, 0, 100)), (300, (0, 0, 100))]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size": PT, "axes.linewidth": 1.6, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": PT, "ytick.labelsize": PT,
    "xtick.major.width": 1.6, "ytick.major.width": 1.6, "savefig.facecolor": "white",
})

class Arrow3D(FancyArrowPatch):
    """A real filled arrowhead in 3-D — mplot3d's quiver head vanishes at this scale."""

    def __init__(self, xyz0, xyz1, **kw):
        super().__init__((0, 0), (0, 0), **kw)
        self._ends = (np.asarray(xyz0, float), np.asarray(xyz1, float))

    def do_3d_projection(self, renderer=None):
        a, b = self._ends
        xs, ys, zs = proj_transform(*np.column_stack([a, b]), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return float(np.min(zs))


P = np.array([np.array(c, float) / sum(c) for _, c in DATA])
conc = np.array([c for c, _ in DATA], float)
T = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)          # the 1:1:1 direction
truth = np.array([1.0, 1.0, 1.0]) / 3.0                 # where it pierces the simplex
theta = np.degrees(np.arccos(np.clip(P @ T / np.linalg.norm(P, axis=1), -1, 1)))
dist = np.linalg.norm(P - truth, axis=1)

fig = plt.figure(figsize=(13.4, 6.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], left=0.0, right=0.965,
                      top=0.93, bottom=0.16, wspace=0.16)

# ------------------------------------------------------------------ a  3-D simplex
ax = fig.add_subplot(gs[0], projection="3d", computed_zorder=False)
ax.view_init(elev=13, azim=22)
ax.set_axis_off()
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
ax.set_box_aspect((1, 1, 1), zoom=1.16)
ax.set_position([0.01, 0.06, 0.55, 0.86])

# plotted axes are (TBZ, DQ, THI) so the triangle reads TBZ left / DQ right / THI up,
# matching the ternary panels; every quantity below is permutation-invariant.
Q = P[:, [1, 0, 2]]
VERT = (((1, 0, 0), "TBZ", TBZ, (0.10, 0.0, 0.02), "right"),
        ((0, 1, 0), "DQ", DQ, (0.0, 0.12, 0.02), "left"),
        ((0, 0, 1), "THI", THI, (0.0, 0.0, 0.11), "center"))
for v, lab, col, off, ha in VERT:
    v = np.array(v, float)
    ax.plot(*np.array([[0, 0, 0], v * 1.04]).T, color=AXIS, lw=2.0, zorder=1)
    ax.text(*(v * 1.04 + np.array(off)), lab, color=col, fontsize=PT, weight="bold",
            ha=ha, va="center", zorder=12)

ax.add_collection3d(Poly3DCollection([[(1, 0, 0), (0, 1, 0), (0, 0, 1)]],
                                     facecolor=PLANE, alpha=0.6, zorder=2,
                                     edgecolor="#7c8b9b", linewidths=2.0))

tt = np.linspace(0, 0.86, 2)                             # the 1:1:1 truth line
ax.plot(tt, tt, tt, ls=(0, (5, 3)), color=INK, lw=2.4, zorder=6)
ax.scatter(*truth, s=150, facecolor="white", edgecolor=INK, linewidths=2.2, zorder=7)
ax.text(0.98, 0.98, 0.94, "1:1:1 truth line", color=INK, fontsize=PT, weight="bold",
        ha="center", zorder=12)

for q in Q:                                              # perpendicular offsets
    ax.plot(*np.array([truth, q]).T, ls=":", color=INK, lw=2.0, zorder=8)
ax.plot(Q[:, 0], Q[:, 1], Q[:, 2], color=THI, lw=3.0, zorder=9)
seg = Q[2] - Q[1]                                        # where the drift is heading
ax.add_artist(Arrow3D(Q[1] + 0.10 * seg, Q[1] + 0.72 * seg, color=THI, lw=3.0,
                      arrowstyle="-|>,head_width=0.34,head_length=0.68",
                      mutation_scale=26, shrinkA=0, shrinkB=0, zorder=11))
ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], s=170, color=THI, depthshade=False, zorder=10)

LAB = {10: ("10 µM", (-0.06, 0.18, -0.03), "left"),
       30: ("30 µM", (0.0, 0.04, 0.11), "center"),
       100: ("100 & 300 µM", (0.30, 0.0, 0.0), "right")}
for (c, _), q in zip(DATA, Q):
    if c in LAB:
        lab, d, ha = LAB[c]
        ax.text(q[0] + d[0], q[1] + d[1], q[2] + d[2], lab, color=INK, fontsize=PT,
                weight="bold", ha=ha, zorder=12)

fig.text(0.012, 0.945, "(a)", fontsize=PT, weight="bold", color=INK)

# ------------------------------------------- b  the three shares against 33.3 %
ax2 = fig.add_subplot(gs[1])
ax2.set_position([0.665, 0.17, 0.333, 0.72])
ax2.set_box_aspect(1)      # square panel
ax2.axhline(100 / 3, ls=(0, (6, 4)), color=GUIDE, lw=2.0, zorder=1)
ax2.text(370, 37.5, "1:1:1  (33.3 %)", fontsize=PT, color=INK, ha="right")
for j, (lab, col, dy) in enumerate((("DQ", DQ, 16), ("TBZ", TBZ, -32), ("THI", THI, 14))):
    y = np.array([c[j] for _, c in DATA], float)
    ax2.plot(conc, y, color=col, lw=3.0, marker="o", ms=12, zorder=5, clip_on=False)
    ax2.annotate(lab, (conc[0], y[0]), textcoords="offset points", xytext=(14, dy),
                 ha="left", fontsize=PT, weight="bold", color=col)

ARR = dict(arrowstyle="-|>,head_width=0.32,head_length=0.6", lw=3.0, shrinkA=0, shrinkB=0)
ax2.annotate("", xy=(110, 94), xytext=(110, 62), arrowprops=dict(color=THI, **ARR), zorder=6)
ax2.annotate("", xy=(110, 5), xytext=(110, 29), arrowprops=dict(color=DQ, **ARR), zorder=6)
ax2.annotate("", xy=(142, 5), xytext=(142, 29), arrowprops=dict(color=TBZ, **ARR), zorder=6)

ax2.set_xscale("log")
ax2.set_xlim(8, 400); ax2.set_ylim(-3, 105)
ax2.set_xticks([10, 30, 100, 300]); ax2.set_xticklabels(["10", "30", "100", "300"])
ax2.xaxis.set_minor_locator(LogLocator(base=10, subs=tuple(np.arange(2, 10) / 10 * 10), numticks=20))
ax2.xaxis.set_minor_formatter(NullFormatter())
ax2.yaxis.set_major_locator(MultipleLocator(20))
ax2.yaxis.set_minor_locator(MultipleLocator(10))
ax2.set_xlabel("Concentration (µM)", fontsize=PT)
ax2.set_ylabel("Composition (%)", fontsize=PT, labelpad=8)
ax2.spines[["top", "right"]].set_visible(False)
for sp in ("left", "bottom"):
    ax2.spines[sp].set_linewidth(2.2)
ax2.tick_params(which="major", direction="out", length=9, width=2.2, pad=8)
ax2.tick_params(which="minor", direction="out", length=5, width=1.6)
fig.text(0.60, 0.945, "(b)", fontsize=PT, weight="bold", color=INK)

fig.savefig(OUT / "fig2_simplex_drift.png", dpi=400)
fig.savefig(OUT / "fig2_simplex_drift.pdf")
print("theta (deg):", np.round(theta, 1), "| distance to the truth line:", np.round(dist, 3))
