"""Ternary composition recovery — the hard subset only: every binary condition
plus the high-concentration ternary conditions (total >= 99 uM).

Held-out predictions from the integrated 20260824 re-evaluation (ratio-grouped
5-fold, refit per fold). Open circle = prepared composition, filled circle =
held-out prediction (colour = accuracy), arrow joins them. Panels: NNLS vs MLP.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labfig
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

labfig.setup()
CO = labfig.CO
INK = "#20262e"
MUTE = "#8a919b"

FULL = (r"S:/Google Drive/내 드라이브/github/Mixture Classifier/documentation/"
        r"results/integrated_final_20260824/full")
ROWS = {}
for m in ("nnls", "mlp"):
    ROWS[m] = json.load(open(os.path.join(FULL, f"partial_{m}.json"),
                             encoding="utf-8"))["rows"]

# order in rows: [DQ, TBZ, THI]; triangle: THI top, TBZ left, DQ right
V = {"THI": np.array([0.5, np.sqrt(3) / 2]),
     "TBZ": np.array([0.0, 0.0]),
     "DQ": np.array([1.0, 0.0])}


def bary(c):
    c = np.clip(np.asarray(c, float), 0, None)
    c = c / (c.sum() + 1e-12)
    return c[0] * V["DQ"] + c[1] * V["TBZ"] + c[2] * V["THI"]


def subset(rows):
    keep = []
    for r in rows:
        t = np.asarray(r["true"], float)
        tot = float(np.sum(r["concentration_uM"]))
        k = int((t > 0).sum())
        if k == 2 or tot >= 99.0:
            keep.append(r)
    return keep


SUB_N = {m: subset(ROWS[m]) for m in ROWS}
n_b = sum(1 for r in SUB_N["mlp"] if (np.asarray(r["true"]) > 0).sum() == 2)
n_h = len(SUB_N["mlp"]) - n_b

norm = Normalize(vmin=0.4, vmax=1.0)
cmap = cm.RdYlGn

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9))
for ax, (m, title) in zip(axes, [("nnls", "NNLS unmixing"),
                                 ("mlp", "MLP unmixing")]):
    ax.set_axis_off(); ax.set_aspect("equal")
    tri = np.array([V["TBZ"], V["DQ"], V["THI"], V["TBZ"]])
    ax.plot(tri[:, 0], tri[:, 1], color=INK, lw=1.0, zorder=2)
    for f in (0.25, 0.5, 0.75):                     # light interior grid
        for a, b, c in ((V["TBZ"], V["DQ"], V["THI"]),
                        (V["DQ"], V["THI"], V["TBZ"]),
                        (V["THI"], V["TBZ"], V["DQ"])):
            p0 = a + f * (b - a); p1 = a + f * (c - a)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#e2e6ea", lw=0.5,
                    zorder=1)
    ax.text(*(V["THI"] + [0, 0.05]), "THI", color=CO["THI"], fontsize=10,
            weight="bold", ha="center")
    ax.text(*(V["TBZ"] + [-0.04, -0.05]), "TBZ", color=CO["TBZ"], fontsize=10,
            weight="bold", ha="right")
    ax.text(*(V["DQ"] + [0.04, -0.05]), "DQ", color=CO["DQ"], fontsize=10,
            weight="bold", ha="left")
    dev_list = []
    for r in SUB_N[m]:
        t = np.asarray(r["true"], float); p = np.asarray(r["pred"], float)
        acc = 1.0 - 0.5 * np.abs(p / (p.sum() + 1e-12) - t).sum()
        dev_list.append(1 - acc)
        pt, pp = bary(t), bary(p)
        ax.annotate("", pp, pt,
                    arrowprops=dict(arrowstyle="-", color="#9aa3ad", lw=0.7,
                                    shrinkA=2.5, shrinkB=2.5), zorder=3)
        ax.scatter(*pt, s=26, facecolors="white", edgecolors="#6a7178",
                   linewidths=0.8, zorder=4)
        ax.scatter(*pp, s=30, facecolors=[cmap(norm(acc))], edgecolors=INK,
                   linewidths=0.5, zorder=5)
    ax.set_xlim(-0.14, 1.14); ax.set_ylim(-0.12, 1.02)
    ax.set_title(title, fontsize=10, weight="bold", pad=6)
    ax.text(0.5, -0.105, f"mean deviation {np.mean(dev_list)*100:.1f} %p",
            fontsize=8, color=MUTE, ha="center")

cax = fig.add_axes([0.435, 0.90, 0.13, 0.025])
cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                  orientation="horizontal", ticks=[0.4, 1.0])
cb.ax.set_xticklabels(["0.4", "1"], fontsize=7)
cb.outline.set_linewidth(0.5)
cax.set_title("accuracy", fontsize=7, pad=2)
fig.text(0.5, 0.015,
         f"held-out · every binary condition (n={n_b}) + ternary with total "
         f"≥ 99 µM (n={n_h}) · open ○ = prepared, filled ● = predicted",
         fontsize=7.5, color=MUTE, ha="center")

fig.tight_layout(rect=(0, 0.03, 1, 0.97))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_ternary_highconc_binary.{ext}"),
                dpi=600, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
print(f"saved: binary {n_b} + high-conc ternary {n_h} maps")
