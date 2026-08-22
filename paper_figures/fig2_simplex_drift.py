"""Figure 2 — where the measured composition lands relative to the 1:1:1 truth line.

The stacked-bar version answers "what fraction", not "how far off". A composition is a
point in DQ/TBZ/THI space, and the dispensed 1:1:1 truth is a LINE through it (the ray
x = y = z). Every normalised composition projects onto that ray at the centroid, so the
dotted drop-line from each point IS its perpendicular distance to the truth line, and
theta is the angle it subtends. 54.7 deg — one pure compound — is the geometric maximum.

Edit DATA below to point at the real per-map numbers.
Run:  python paper_figures/fig2_simplex_drift.py
"""
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = Path(__file__).resolve().parent
DQ, TBZ, THI = "#1a73e8", "#27a679", "#e8467c"
INK, MUTE, AXIS = "#12202e", "#7b8a9b", "#b9c3cf"

# concentration (uM) -> composition percentages (DQ, TBZ, THI)
DATA = [(10, (30, 23, 47)), (30, (18, 34, 48)), (100, (0, 0, 100)), (300, (0, 0, 100))]

plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 9, "axes.linewidth": 0.9,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9, "savefig.facecolor": "white",
})

P = np.array([np.array(c, float) / sum(c) for _, c in DATA])
conc = np.array([c for c, _ in DATA], float)
T = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)          # the 1:1:1 direction
truth = np.array([1.0, 1.0, 1.0]) / 3.0                 # where it pierces the simplex
theta = np.degrees(np.arccos(np.clip(P @ T / np.linalg.norm(P, axis=1), -1, 1)))
dist = np.linalg.norm(P - truth, axis=1)

fig = plt.figure(figsize=(7.6, 3.7))
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.10,
                      left=0.00, right=0.96, top=0.88, bottom=0.19)

# ------------------------------------------------------------------ a  3-D simplex
ax = fig.add_subplot(gs[0], projection="3d", computed_zorder=False)
ax.view_init(elev=13, azim=22)
ax.set_axis_off()
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
ax.set_box_aspect((1, 1, 1), zoom=1.12)
ax.set_position([0.005, 0.05, 0.58, 0.82])

VERT = (((1, 0, 0), "DQ", DQ, (0.04, 0.0, 0.02), "left"),
        ((0, 1, 0), "TBZ", TBZ, (0.0, 0.06, 0.02), "center"),
        ((0, 0, 1), "THI", THI, (0.16, -0.16, 0.03), "left"))
for v, lab, col, off, ha in VERT:
    v = np.array(v, float)
    ax.plot(*np.array([[0, 0, 0], v * 1.04]).T, color=AXIS, lw=1.0, zorder=1)
    ax.text(*(v * 1.04 + np.array(off)), f"{lab}\n100%", color=col, fontsize=9,
            weight="bold", ha=ha, va="center", zorder=12)

ax.add_collection3d(Poly3DCollection([[(1, 0, 0), (0, 1, 0), (0, 0, 1)]],
                                     facecolor="#dde5ee", alpha=0.55, zorder=2,
                                     edgecolor="#9aa8b8", linewidths=1.0))
tt = np.linspace(0, 0.82, 2)                                     # the 1:1:1 truth line
ax.plot(tt, tt, tt, ls=(0, (5, 3)), color=INK, lw=1.5, zorder=6)
ax.scatter(*truth, s=58, facecolor="white", edgecolor=INK, linewidths=1.5, zorder=7)
ax.text(0.90, 0.90, 0.86, "1 : 1 : 1 truth line", color=INK, fontsize=8.6,
        weight="bold", ha="left", zorder=12)

for p in P:                                                      # perpendicular offsets
    ax.plot(*np.array([truth, p]).T, ls=":", color="#5d6b7b", lw=1.1, zorder=8)
ax.plot(P[:, 0], P[:, 1], P[:, 2], color=THI, lw=1.7, alpha=0.9, zorder=9)
ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=58, color=THI, edgecolor="white",
           linewidths=0.9, depthshade=False, zorder=10)

for (c, _), p in zip(DATA, P):
    if c == 10:
        ax.text(p[0] + 0.06, p[1] - 0.14, p[2] - 0.02, "10 µM", color=INK, fontsize=8.6,
                weight="bold", ha="right", zorder=12)
    elif c == 30:
        ax.text(p[0], p[1] + 0.02, p[2] + 0.08, "30 µM", color=INK, fontsize=8.6,
                weight="bold", ha="center", zorder=12)
    elif c == 100:
        ax.text(p[0] + 0.04, p[1] + 0.24, p[2] - 0.14, "100 & 300 µM", color=INK,
                fontsize=8.6, weight="bold", ha="left", zorder=12)

fig.text(0.012, 0.935, "a", fontsize=13, weight="bold", color=INK)
fig.text(0.035, 0.937, "composition space", fontsize=9.5, weight="bold", color=INK)
fig.text(0.035, 0.888, "the dotted line is the offset from the truth",
         fontsize=8, color=MUTE, style="italic")

# ------------------------------------------------- b  how far off the truth line
ax2 = fig.add_subplot(gs[1])
ax2.axhline(54.7356, ls=(0, (5, 3)), color=MUTE, lw=1.0)
ax2.text(8.4, 57.0, "pure THI — the geometric maximum, 54.7°", fontsize=7.6, color=MUTE)
ax2.axhline(0, color=INK, lw=1.0)
ax2.text(8.4, 3.0, "on the truth line", fontsize=7.6, color=INK)
ax2.plot(conc, theta, color=THI, lw=1.7, marker="o", ms=6.5, mec="white", mew=1.0, zorder=5)
for c, th in zip(conc, theta):
    dx, dy, ha = ((-10, -18, "right") if c == 100 else
                  (8, -15, "left") if c == 300 else (0, 10, "center"))
    ax2.annotate(f"{th:.1f}°", (c, th), textcoords="offset points", xytext=(dx, dy),
                 ha=ha, fontsize=8.6, weight="bold", color=INK)
ax2.set_xscale("log")
ax2.set_xlim(7.5, 460); ax2.set_ylim(-6, 70)
ax2.set_xticks([10, 30, 100, 300]); ax2.set_xticklabels(["10", "30", "100", "300"])
ax2.set_yticks([0, 20, 40, 60])
ax2.set_xlabel("Concentration (µM)")
ax2.set_ylabel("Angle from the 1:1:1 line (°)", labelpad=3)
ax2.spines[["top", "right"]].set_visible(False)
ax2.tick_params(direction="out", length=3.5)
fig.text(0.605, 0.935, "b", fontsize=13, weight="bold", color=INK)
fig.text(0.628, 0.937, "how far off", fontsize=9.5, weight="bold", color=INK)

fig.text(0.5, 0.035, "up to 30 µM the readout stays near 1:1:1; from 100 µM it collapses onto pure "
         "THI — the largest deviation the simplex allows",
         ha="center", fontsize=8.2, color=MUTE, style="italic")

fig.savefig(OUT / "fig2_simplex_drift.png", dpi=400)
fig.savefig(OUT / "fig2_simplex_drift.pdf")
print("theta (deg):", np.round(theta, 1), "| distance to the truth line:", np.round(dist, 3))
