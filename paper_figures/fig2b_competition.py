"""Figure 2b — the competition, as the three shares against the 1:1:1 truth.

One square panel: DQ / TBZ / THI composition against the four dispensed concentrations,
with 33.3 % (the dispensed 1:1:1) as the reference. The arrows carry the claim — THI
rises to the whole surface response while DQ and TBZ fall to zero at the same step.

The four concentrations are plotted as four evenly spaced categories, not on a log axis.
Edit DATA to point at the real per-map numbers.
Run:  python paper_figures/fig2b_competition.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

OUT = Path(__file__).resolve().parent
DQ, TBZ, THI = "#1a73e8", "#27a679", "#e8467c"
INK, GUIDE = "#000000", "#8c8c8c"
PT = 18                                                  # one type size, everywhere

# concentration (uM) -> composition percentages (DQ, TBZ, THI)
DATA = [(10, (30, 23, 47)), (30, (18, 34, 48)), (100, (0, 0, 100)), (300, (0, 0, 100))]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size": PT, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": PT, "ytick.labelsize": PT,
    "savefig.facecolor": "white",
})

x = np.arange(len(DATA), dtype=float)                    # four evenly spaced points
labels = [f"{c:g}" for c, _ in DATA]

fig = plt.figure(figsize=(6.0, 6.0))
ax = fig.add_axes([0.20, 0.16, 0.76, 0.80])
ax.set_box_aspect(1)

ax.axhline(100 / 3, ls=(0, (6, 4)), color=GUIDE, lw=2.0, zorder=1)
ax.text(3.28, 37, "1:1:1  (33.3 %)", fontsize=PT, color=INK, ha="right")

for j, (lab, col, dy) in enumerate((("DQ", DQ, 16), ("TBZ", TBZ, -32), ("THI", THI, 14))):
    y = np.array([c[j] for _, c in DATA], float)
    ax.plot(x, y, color=col, lw=3.0, marker="o", ms=12, zorder=5, clip_on=False)
    ax.annotate(lab, (x[0], y[0]), textcoords="offset points", xytext=(14, dy),
                ha="left", fontsize=PT, weight="bold", color=col)

ARR = dict(arrowstyle="-|>,head_width=0.32,head_length=0.6", lw=3.0, shrinkA=0, shrinkB=0)
ax.annotate("", xy=(2.30, 94), xytext=(2.30, 62), arrowprops=dict(color=THI, **ARR), zorder=6)
ax.annotate("", xy=(2.30, 5), xytext=(2.30, 29), arrowprops=dict(color=DQ, **ARR), zorder=6)
ax.annotate("", xy=(2.62, 5), xytext=(2.62, 29), arrowprops=dict(color=TBZ, **ARR), zorder=6)

ax.set_xlim(-0.28, 3.28); ax.set_ylim(-3, 105)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.yaxis.set_major_locator(MultipleLocator(20))
ax.yaxis.set_minor_locator(MultipleLocator(10))
ax.set_xlabel("Concentration (µM)", fontsize=PT, labelpad=8)
ax.set_ylabel("Composition (%)", fontsize=PT, labelpad=8)
ax.spines[["top", "right"]].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_linewidth(2.2)
ax.spines["left"].set_bounds(0, 100)                     # the spine stops at the last tick
ax.tick_params(which="major", direction="out", length=9, width=2.2, pad=8)
ax.tick_params(axis="y", which="minor", direction="out", length=5, width=1.6)
ax.tick_params(axis="x", which="minor", length=0)

fig.savefig(OUT / "fig2b_competition.png", dpi=400)
fig.savefig(OUT / "fig2b_competition.pdf")

box = ax.get_window_extent(fig.canvas.get_renderer())    # prove it is square
print(f"axes box: {box.width:.1f} x {box.height:.1f} px")
