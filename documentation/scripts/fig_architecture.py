"""Model-architecture schematic — reads for a non-specialist, in the paper's
flat figure style.

Numbered steps with plain-language captions; the NNLS gate shown as an actual
pixel filter; fully-connected layer diagrams with true dimensions; training
objectives stated under each network; ONE reported output per panel (the
known-total reconstruction is a demoted footnote, not a parallel box).
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
PLAIN = "#3a4046"
FAN = "#d3d7dc"
GREEN_BG = "#eef6f0"
GREEN_E = "#2e8b62"
SUBS = ["DQ", "TBZ", "THI"]
CLASS_COLORS = [CO[s] for s in SUBS] + ["#9aa3ad"]

fig = plt.figure(figsize=(7.2, 4.5))
ov = fig.add_axes([0, 0, 1, 1], zorder=0)
ov.set_xlim(0, 1); ov.set_ylim(0, 1); ov.set_axis_off()
AR = 7.2 / 4.5

BOX_T, EQ_FS, SUB_FS, PL_FS = 7.2, 6.4, 5.4, 5.9


def box(x, y, w, h, lines, fill="white", edge=INK, lw=0.9, ls="-"):
    """Flat box. Line kinds: t bold title · p plain-language · e equation · s note."""
    ov.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.002,rounding_size=0.008",
                                facecolor=fill, edgecolor=edge, linewidth=lw,
                                linestyle=ls, zorder=2))
    n = len(lines)
    step = h / (n + 0.5)
    for i, (t, kind) in enumerate(lines):
        fs = {"t": BOX_T, "p": PL_FS, "e": EQ_FS, "s": SUB_FS}[kind]
        col = {"t": INK, "p": PLAIN, "e": INK, "s": MUTE}[kind]
        ov.text(x + w / 2, y + h - (i + 0.78) * step, t, fontsize=fs,
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


def badge(x, y, num):
    ov.scatter([x], [y], s=58, facecolors="white", edgecolors=INK,
               linewidths=0.8, zorder=5)
    ov.text(x, y, str(num), fontsize=5.6, weight="bold", color=INK,
            ha="center", va="center", zorder=6)


def out_tag(x, y, label="output"):
    ov.add_patch(FancyBboxPatch((x - 0.032, y - 0.014), 0.064, 0.028,
                                boxstyle="round,pad=0.002,rounding_size=0.010",
                                facecolor=INK, edgecolor="none", zorder=5))
    ov.text(x, y, label, fontsize=5.2, weight="bold", color="white",
            ha="center", va="center", zorder=6)


def fc_net(xs, heights, slots, ells, cy, dims, descs, label_y, desc_y,
           node_s=14, out_colors=None):
    cols = []
    for x, h, n, ell in zip(xs, heights, slots, ells):
        ys = list(np.linspace(cy + h / 2, cy - h / 2, n))
        if ell:
            m = n // 2
            ys = ys[:m - 1] + ys[m + 1:]
            ov.scatter([x] * 3, [cy - 0.010, cy, cy + 0.010], s=1.5, c=[INK],
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
        ov.text(x, label_y, d, fontsize=6.4, weight="bold", color=INK,
                ha="center", zorder=4)
        ov.text(x, desc_y, ds, fontsize=4.9, color=MUTE, ha="center", zorder=4)
    return cols


def pixel_grid(x0, y0, w, gated=False, seed=5):
    """6x6 schematic map. gated=True greys out the background pixels."""
    rng = np.random.default_rng(seed)
    vals = rng.uniform(0.15, 1.0, (6, 6))
    hitm = vals > 0.45
    cw = w / 6
    ch = cw * AR
    for r in range(6):
        for c in range(6):
            if gated and not hitm[r, c]:
                fc_ = "#e9ebee"
            else:
                fc_ = plt.cm.viridis(vals[r, c] * 0.85)
            ov.add_patch(plt.Rectangle((x0 + c * cw, y0 + r * ch), cw * 0.92,
                                       ch * 0.92, facecolor=fc_,
                                       edgecolor="none", zorder=3))
    return w, ch * 6


# ===================== (a) composition ===========================================
ov.text(0.008, 0.958, "(a) Pixel-wise composition network", fontsize=10,
        weight="bold", color=INK, ha="left")
ROW_A = 0.760

badge(0.030, 0.905, 1)
pixel_grid(0.022, 0.705, 0.056)
cap(0.050, 0.668, "SERS map", fs=BOX_T, col=INK, weight="bold")
cap(0.050, 0.637, "one spectrum", fs=5.1, col=PLAIN)
cap(0.050, 0.612, "per pixel", fs=5.1, col=PLAIN)

arrow(0.084, ROW_A, 0.102, ROW_A)
badge(0.112, 0.905, 2)
pixel_grid(0.106, 0.705, 0.056, gated=True)
cap(0.134, 0.668, "NNLS gate", fs=BOX_T, col=INK, weight="bold")
cap(0.134, 0.637, "substance pixels kept", fs=5.1, col=PLAIN)
cap(0.134, 0.612, "background dropped", fs=5.1, col=PLAIN)

arrow(0.168, ROW_A, 0.186, ROW_A)
axs = fig.add_axes([0.190, 0.700, 0.098, 0.130])
wx = np.linspace(500, 2500, 600)
spec = sum(a * np.exp(-0.5 * ((wx - c) / wdt) ** 2)
           for a, c, wdt in [(0.65, 800, 18), (0.5, 1010, 14), (0.8, 1270, 15),
                             (0.9, 1370, 14), (1.0, 1570, 16), (0.35, 1650, 25)])
axs.plot(wx, spec, color=INK, lw=0.7)
axs.set_xlim(500, 2500); axs.set_ylim(0, 1.2)
axs.set_xticks([]); axs.set_yticks([]); axs.patch.set_alpha(0)
for spn in axs.spines.values():
    spn.set_visible(False)
axs.spines["bottom"].set_visible(True); axs.spines["bottom"].set_linewidth(0.6)
cap(0.239, 0.668, r"pixel spectra x$_i$", fs=BOX_T, col=INK, weight="bold")
cap(0.239, 0.637, "log(1+I) · 1,290 pts", fs=5.1, col=PLAIN)
cap(0.239, 0.610, r"500–2500 cm$^{-1}$")

arrow(0.292, ROW_A, 0.318, ROW_A)
badge(0.327, 0.928, 3)
ov.add_patch(FancyBboxPatch((0.314, 0.545), 0.318, 0.400,
                            boxstyle="round,pad=0.003,rounding_size=0.012",
                            facecolor=GREEN_BG, edgecolor=GREEN_E, lw=0.9,
                            zorder=0.5))
cap(0.473, 0.918, r"shared neural network f(x$_i$)", fs=BOX_T, col=INK,
    weight="bold")
fc_net([0.356, 0.434, 0.512, 0.588],
       heights=[0.235, 0.190, 0.140, 0.095],
       slots=[12, 9, 7, 4], ells=[True, True, False, False],
       cy=0.765, dims=["1,290", "256", "64", "4"],
       descs=["input", "FC · ReLU · drop", "FC · ReLU", "softmax"],
       label_y=0.612, desc_y=0.586, out_colors=CLASS_COLORS)
cap(0.473, 0.557, "trained on prepared mixtures · weighted L1 · held-out by condition",
    fs=4.9)
for (yy, s_, c_) in zip(np.linspace(0.765 + 0.0475, 0.765 - 0.0475, 4),
                        SUBS + ["BLK"], CLASS_COLORS):
    ov.text(0.600, yy, s_, fontsize=5.2, color=c_, ha="left", va="center",
            weight="bold", zorder=4)

arrow(0.636, ROW_A, 0.656, ROW_A)
ov.scatter([0.646], [ROW_A], s=6, c=[INK], zorder=5)       # junction to (b)

badge(0.664, 0.905, 4)
CH_W = 0.032
CH_H = CH_W * AR
for k, (s_, c_) in enumerate(zip(SUBS, [CO[s] for s in SUBS])):
    x0 = 0.662 + k * 0.040
    ov.add_patch(FancyBboxPatch((x0, ROW_A - CH_H / 2), CH_W, CH_H,
                                boxstyle="round,pad=0.001,rounding_size=0.004",
                                facecolor=c_, edgecolor="white", lw=0.5,
                                alpha=0.85, zorder=3))
    for f_ in (1 / 3, 2 / 3):
        ov.plot([x0, x0 + CH_W], [ROW_A - CH_H / 2 + f_ * CH_H] * 2,
                color="white", lw=0.4, zorder=4)
        ov.plot([x0 + f_ * CH_W] * 2, [ROW_A - CH_H / 2, ROW_A + CH_H / 2],
                color="white", lw=0.4, zorder=4)
    ov.text(x0 + CH_W / 2, ROW_A - CH_H / 2 - 0.024, s_, fontsize=4.8,
            color=c_, ha="center", weight="bold", zorder=4)
cap(0.718, 0.855, "pesticide maps", fs=BOX_T, col=INK, weight="bold")
cap(0.718, 0.660, "where each pesticide sits", fs=PL_FS, col=PLAIN)
cap(0.718, 0.632, r"p$_i$ = softmax(f(x$_i$))")

arrow(0.784, ROW_A, 0.806, ROW_A)
badge(0.814, 0.905, 5)
out_tag(0.945, 0.905)
bx, bw = 0.812, 0.150
for k, (f0, f1) in enumerate([(0, 0.45), (0.45, 0.75), (0.75, 1.0)]):
    ov.add_patch(plt.Rectangle((bx + f0 * bw, ROW_A - 0.022), (f1 - f0) * bw,
                               0.044, facecolor=[CO[s] for s in SUBS][k],
                               edgecolor="white", lw=0.5, zorder=3))
cap(0.887, 0.855, "map composition", fs=BOX_T, col=INK, weight="bold")
cap(0.887, 0.660, "% of each pesticide in the map", fs=PL_FS, col=PLAIN)
cap(0.887, 0.632, r"$\bar{p}$ = pixel mean · BLK removed")

# ===================== (b) quantification ========================================
ov.text(0.008, 0.478, "(b) Calibration-anchored residual quantification",
        fontsize=10, weight="bold", color=INK, ha="left")
ROW_B = 0.290

elbow([(0.646, ROW_A - 0.004), (0.646, 0.443), (0.105, 0.443), (0.105, 0.377)])
cap(0.375, 0.455, r"pixel scores p$_i$ and spectra I$_i$")

badge(0.033, 0.400, 6)
box(0.025, 0.200, 0.162, 0.175,
    [("marker-band brightness", "t"),
     ("how strong is each band?", "p"),
     (r"I$_{eq,k}$ = $\Sigma_i$p$_{ik}$I$_i$($\nu_k$) / $\Sigma_i$p$_{ik}$", "e"),
     (r"$\nu_k$: 1570 / 1270 / 1370 cm$^{-1}$", "s")])
arrow(0.189, ROW_B, 0.208, ROW_B)
badge(0.216, 0.400, 7)
box(0.210, 0.200, 0.148, 0.175,
    [("calibration inversion", "t"),
     ("brightness → µM", "p"),
     (r"C$_{cal}$ = 10$^{(I_{eq}-b)/a}$", "e"),
     ("clipped to calibrated range", "s")])
box(0.210, 0.045, 0.148, 0.105,
    [("measured calibration", "t"),
     (r"I = a·log$_{10}$C + b", "e")])
arrow(0.284, 0.152, 0.284, 0.197)
arrow(0.360, ROW_B, 0.382, ROW_B)

badge(0.398, 0.408, 8)
ov.add_patch(FancyBboxPatch((0.386, 0.130), 0.288, 0.300,
                            boxstyle="round,pad=0.003,rounding_size=0.012",
                            facecolor=GREEN_BG, edgecolor=GREEN_E, lw=0.9,
                            zorder=0.5))
cap(0.530, 0.405, "correction network", fs=BOX_T, col=INK, weight="bold")
cap(0.530, 0.378, "learns how mixtures distort the calibration", fs=4.9,
    col=PLAIN)
fc_net([0.422, 0.494, 0.566, 0.636],
       heights=[0.125, 0.170, 0.115, 0.065],
       slots=[8, 9, 7, 3], ells=[True, True, False, False],
       cy=0.272, dims=["12", "128", "32", "3"],
       descs=["z features", "FC · ReLU · drop", "FC · ReLU",
              r"$\Delta$log$_{10}$C"],
       label_y=0.168, desc_y=0.143, out_colors=[CO[s] for s in SUBS])
cap(0.530, 0.108,
    r"z = [ log$_{10}$C$_{cal}$ · $\bar{p}$ · log I$_{eq}$ · intensity P10/50/90 ]",
    fs=5.2)
cap(0.530, 0.082, r"trained on the same mixtures · Huber loss on $\Delta$log$_{10}$C",
    fs=4.9)

arrow(0.678, ROW_B, 0.700, ROW_B)
badge(0.712, 0.400, 9)
out_tag(0.905, 0.400)
box(0.704, 0.200, 0.280, 0.175,
    [("reported concentration", "t"),
     ("the final answer, per pesticide", "p"),
     (r"log$_{10}\hat{C}$ = log$_{10}$C$_{cal}$ + $\Delta$   →   $\hat{C}$ (µM)", "e"),
     ("validated window · out-of-range flagged", "s")], lw=1.4)

# demoted footnote — NOT part of the reported pipeline
cap(0.704, 0.128, "optional, only when the sample's total concentration is "
    "declared:", fs=5.0, ha="left")
cap(0.704, 0.100, r"$\hat{C}_i$ = $\bar{p}_i$ · C$_{total}$ — reported "
    "separately, marked as constrained", fs=5.0, ha="left")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_architecture_dl.{ext}"), dpi=600,
                bbox_inches="tight", pad_inches=0.02, facecolor="white")
print("saved fig_architecture_dl.png/.pdf")
