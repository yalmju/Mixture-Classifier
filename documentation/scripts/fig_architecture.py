"""Model-architecture schematic — classic fully-connected layer diagrams in this
paper's flat figure style. No data insets: the layers are the subject.

(a) shared per-pixel MLP 1290-256-64-4 drawn as node columns with full fan
connections; (b) the calibration-anchored residual quantifier with its
12-128-32-3 net. Every stage carries its actual quantity/equation.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labfig
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

labfig.setup()
CO = labfig.CO
INK = "black"
MUTE = "#6a7178"
FAN = "#d3d7dc"
GREEN_BG = "#eef6f0"       # learned blocks — tint of the benchmark's MLP green
GREEN_E = "#2e8b62"
SUBS = ["DQ", "TBZ", "THI"]
CLASS_COLORS = [CO[s] for s in SUBS] + ["#9aa3ad"]

fig = plt.figure(figsize=(7.2, 4.2))
ov = fig.add_axes([0, 0, 1, 1], zorder=0)
ov.set_xlim(0, 1); ov.set_ylim(0, 1); ov.set_axis_off()
AR = 7.2 / 4.2

BOX_T, EQ_FS, SUB_FS = 7.4, 6.8, 6.0


def box(x, y, w, h, lines, fill="white", edge=INK, lw=0.9, ls="-"):
    ov.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.002,rounding_size=0.008",
                                facecolor=fill, edgecolor=edge, linewidth=lw,
                                linestyle=ls, zorder=2))
    n = len(lines)
    step = h / (n + 0.55)
    for i, (t, kind) in enumerate(lines):
        fs = BOX_T if kind == "t" else EQ_FS if kind == "e" else SUB_FS
        col = INK if kind in ("t", "e") else MUTE
        ov.text(x + w / 2, y + h - (i + 0.82) * step, t, fontsize=fs,
                weight="bold" if kind == "t" else "normal", color=col,
                ha="center", va="center", zorder=3)


def arrow(x0, y0, x1, y1, ls="-", lw=0.85):
    ov.annotate("", (x1, y1), (x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=lw,
                                linestyle=ls, mutation_scale=7,
                                shrinkA=1, shrinkB=1), zorder=3)


def elbow(pts, ls="-"):
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        if i == len(pts) - 2:
            arrow(a[0], a[1], b[0], b[1], ls=ls)
        else:
            ov.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=0.85, ls=ls,
                    zorder=3)


def cap(x, y, t, fs=SUB_FS, col=MUTE, weight="normal", ha="center"):
    ov.text(x, y, t, fontsize=fs, color=col, ha=ha, weight=weight, zorder=4)


def fc_net(xs, heights, slots, ells, cy, dims, descs, label_y, desc_y,
           node_s=15, out_colors=None):
    """Classic fully-connected diagram: node columns + complete fan lines."""
    cols = []
    for x, h, n, ell in zip(xs, heights, slots, ells):
        ys = list(np.linspace(cy + h / 2, cy - h / 2, n))
        if ell:
            m = n // 2
            ys = ys[:m - 1] + ys[m + 1:]
            ov.scatter([x] * 3, [cy - 0.011, cy, cy + 0.011], s=1.6, c=[INK],
                       edgecolors="none", zorder=5)
        cols.append((x, ys))
    for (x0, ys0), (x1, ys1) in zip(cols, cols[1:]):
        for y0 in ys0:
            for y1 in ys1:
                ov.plot([x0, x1], [y0, y1], color=FAN, lw=0.28, zorder=1)
    for j, (x, ys) in enumerate(cols):
        last = j == len(cols) - 1 and out_colors
        fc = out_colors if last else ["white"] * len(ys)
        ov.scatter([x] * len(ys), ys, s=node_s, facecolors=fc, edgecolors=INK,
                   linewidths=0.55, zorder=4)
    for x, d, ds in zip(xs, dims, descs):
        ov.text(x, label_y, d, fontsize=6.6, weight="bold", color=INK,
                ha="center", zorder=4)
        ov.text(x, desc_y, ds, fontsize=5.0, color=MUTE, ha="center", zorder=4)
    return cols


# ===================== (a) pixel-wise composition network ========================
ov.text(0.008, 0.952, "(a) Pixel-wise composition network", fontsize=10,
        weight="bold", color=INK, ha="left")

# schematic input spectrum (drawn, not data)
axs = fig.add_axes([0.028, 0.700, 0.120, 0.150])
wx = np.linspace(500, 2500, 600)
spec = sum(a * np.exp(-0.5 * ((wx - c) / wdt) ** 2)
           for a, c, wdt in [(0.65, 800, 18), (0.5, 1010, 14), (0.8, 1270, 15),
                             (0.9, 1370, 14), (1.0, 1570, 16), (0.35, 1650, 25)])
axs.plot(wx, spec, color=INK, lw=0.7)
for c_, col_ in zip((1570, 1270, 1370), [CO[s] for s in SUBS]):
    axs.plot([c_], [1.12], marker="v", ms=2.4, color=col_, clip_on=False)
axs.set_xlim(500, 2500); axs.set_ylim(0, 1.2)
axs.set_xticks([]); axs.set_yticks([]); axs.patch.set_alpha(0)
for spn in axs.spines.values():
    spn.set_visible(False)
axs.spines["bottom"].set_visible(True); axs.spines["bottom"].set_linewidth(0.6)
cap(0.088, 0.885, "NNLS-gated hit pixels · i = 1…N")
cap(0.088, 0.660, r"x$_i$ = log(1+I$_i$)", fs=BOX_T, col=INK)
cap(0.088, 0.626, "1,290 channels")

arrow(0.155, 0.775, 0.262, 0.775)

# shared MLP as a fully-connected node diagram
ov.add_patch(FancyBboxPatch((0.258, 0.530), 0.345, 0.415,
                            boxstyle="round,pad=0.003,rounding_size=0.012",
                            facecolor=GREEN_BG, edgecolor=GREEN_E, lw=0.9,
                            zorder=0.5))
cap(0.4305, 0.915, r"shared MLP  f(x$_i$)", fs=BOX_T, col=INK, weight="bold")
cols_a = fc_net([0.300, 0.388, 0.476, 0.560],
                heights=[0.265, 0.210, 0.155, 0.105],
                slots=[12, 9, 7, 4], ells=[True, True, False, False],
                cy=0.752, dims=["1,290", "256", "64", "4"],
                descs=["input", "FC · ReLU · drop", "FC · ReLU", "softmax"],
                label_y=0.585, desc_y=0.557, out_colors=CLASS_COLORS)
for (yy, s_, c_) in zip(cols_a[-1][1], SUBS + ["BLK"], CLASS_COLORS):
    ov.text(0.574, yy, s_, fontsize=5.4, color=c_, ha="left", va="center",
            weight="bold", zorder=4)

arrow(0.608, 0.752, 0.648, 0.752)
ov.scatter([0.626], [0.752], s=6, c=[INK], zorder=5)      # junction to (b)

# schematic per-pixel probability maps
cap(0.712, 0.850, r"p$_i$ = softmax(f(x$_i$))", fs=EQ_FS, col=INK)
CH_W = 0.036
CH_H = CH_W * AR
for k, (s_, c_) in enumerate(zip(SUBS, [CO[s] for s in SUBS])):
    x0 = 0.655 + k * 0.044
    ov.add_patch(FancyBboxPatch((x0, 0.752 - CH_H / 2), CH_W, CH_H,
                                boxstyle="round,pad=0.001,rounding_size=0.004",
                                facecolor=c_, edgecolor="white", lw=0.5,
                                alpha=0.85, zorder=3))
    for f_ in (1 / 3, 2 / 3):                       # map-grid glyph lines
        ov.plot([x0, x0 + CH_W], [0.752 - CH_H / 2 + f_ * CH_H] * 2,
                color="white", lw=0.4, zorder=4)
        ov.plot([x0 + f_ * CH_W] * 2, [0.752 - CH_H / 2, 0.752 + CH_H / 2],
                color="white", lw=0.4, zorder=4)
    ov.text(x0 + CH_W / 2, 0.752 - CH_H / 2 - 0.026, s_, fontsize=5.0,
            color=c_, ha="center", weight="bold", zorder=4)
cap(0.712, 0.640, "per-pixel probability maps")

arrow(0.792, 0.752, 0.822, 0.752)
# pooled composition (schematic thirds)
bx, bw = 0.828, 0.150
for k, (f0, f1) in enumerate([(0, 0.45), (0.45, 0.75), (0.75, 1.0)]):
    ov.add_patch(plt.Rectangle((bx + f0 * bw, 0.727), (f1 - f0) * bw, 0.05,
                               facecolor=[CO[s] for s in SUBS][k],
                               edgecolor="white", lw=0.5, zorder=3))
cap(0.903, 0.850, "map composition", fs=BOX_T, col=INK, weight="bold")
cap(0.903, 0.800, r"$\bar{p}$ = mean of p$_i$ over hit pixels", fs=SUB_FS)
cap(0.903, 0.678, "BLK removed · renormalised")

# ===================== (b) calibration-anchored quantification ===================
ov.text(0.008, 0.478, "(b) Calibration-anchored residual quantification",
        fontsize=10, weight="bold", color=INK, ha="left")

# per-pixel probabilities and intensities feed the band read-out
elbow([(0.626, 0.748), (0.626, 0.443), (0.103, 0.443), (0.103, 0.358)])
cap(0.370, 0.455, r"p$_i$ · I$_i$  (hit pixels)")

box(0.025, 0.205, 0.157, 0.150,
    [("band read-out", "t"),
     (r"I$_{eq,k}$ = $\Sigma_i$p$_{ik}$I$_i$($\nu_k$) / $\Sigma_i$p$_{ik}$", "e"),
     (r"$\nu_k$: 1570 / 1270 / 1370 cm$^{-1}$", "s")])
arrow(0.184, 0.280, 0.206, 0.280)
box(0.210, 0.205, 0.140, 0.150,
    [("inversion", "t"),
     (r"C$_{cal}$ = 10$^{(I_{eq}-b)/a}$", "e"),
     ("clipped to calibrated range", "s")])
box(0.210, 0.048, 0.140, 0.108,
    [("calibration", "t"),
     (r"I = a·log$_{10}$C + b", "e"),
     ("per-substance (a, b)", "s")])
arrow(0.280, 0.158, 0.280, 0.202)
arrow(0.352, 0.280, 0.384, 0.280)

# residual MLP as a fully-connected node diagram
ov.add_patch(FancyBboxPatch((0.388, 0.112), 0.292, 0.313,
                            boxstyle="round,pad=0.003,rounding_size=0.012",
                            facecolor=GREEN_BG, edgecolor=GREEN_E, lw=0.9,
                            zorder=0.5))
cap(0.534, 0.398, "residual MLP", fs=BOX_T, col=INK, weight="bold")
fc_net([0.425, 0.497, 0.569, 0.638],
       heights=[0.135, 0.185, 0.125, 0.070],
       slots=[8, 9, 7, 3], ells=[True, True, False, False],
       cy=0.278, dims=["12", "128", "32", "3"],
       descs=["z features", "FC · ReLU · drop", "FC · ReLU",
              r"$\Delta$log$_{10}$C"],
       label_y=0.160, desc_y=0.133, out_colors=[CO[s] for s in SUBS])
cap(0.534, 0.080,
    r"z = [ log$_{10}$C$_{cal}$ · $\bar{p}$ · log I$_{eq}$ · "
    "intensity P10/50/90 ]", fs=5.6)

arrow(0.684, 0.280, 0.702, 0.280)
box(0.706, 0.205, 0.125, 0.150,
    [(r"$\hat{C}$ (µM)", "t"),
     (r"$\hat{C}$ = C$_{cal}$·10$^{\Delta}$", "e"),
     ("validated · OOD flags", "s")], lw=1.15)
box(0.843, 0.205, 0.143, 0.150,
    [("known total", "t"),
     (r"$\hat{C}_i$ = $\bar{p}_i$ · C$_{total}$", "e"),
     ("declared · constrained", "s")],
    edge=MUTE, lw=0.8, ls=(0, (4, 2)))
# composition feeds the known-total gate — right-margin dashed route
elbow([(0.978, 0.727), (0.993, 0.727), (0.993, 0.280), (0.988, 0.280)],
      ls=(0, (4, 2)))

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_architecture_dl.{ext}"), dpi=600,
                bbox_inches="tight", pad_inches=0.02, facecolor="white")
print("saved fig_architecture_dl.png/.pdf")
