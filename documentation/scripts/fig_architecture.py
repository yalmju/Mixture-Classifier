"""Model-architecture strip — one row, three titled stages, per the author's
layout: Raw SERS data → Compositional unmixing → Concentration recovery.

Flat paper style; numbered steps 3-9 with plain-language captions; layer
diagrams with true dimensions; training objectives stated; one tagged output
per stage; known-total demoted to a footnote. Designed at 14.4 in wide so a
50 % reduction to a 183 mm double-column width keeps every label legible.
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
PANEL_BG = "#f2f3f4"
PANEL_E = "#9aa1a8"
SUBS = ["DQ", "TBZ", "THI"]
CLASS_COLORS = [CO[s] for s in SUBS] + ["#9aa3ad"]

fig = plt.figure(figsize=(14.4, 2.9))
ov = fig.add_axes([0, 0, 1, 1], zorder=0)
ov.set_xlim(0, 1); ov.set_ylim(0, 1); ov.set_axis_off()
AR = 14.4 / 2.9

TITLE_FS, BOX_T, EQ_FS, SUB_FS, PL_FS = 12.5, 8.4, 7.4, 6.4, 7.0
CY = 0.55                                    # main-row centreline


def box(x, y, w, h, lines, fill="white", edge=INK, lw=1.0, ls="-"):
    ov.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.0015,rounding_size=0.006",
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


def arrow(x0, y0, x1, y1, ls="-", lw=1.0):
    ov.annotate("", (x1, y1), (x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=lw,
                                linestyle=ls, mutation_scale=9,
                                shrinkA=1, shrinkB=1), zorder=3)


def elbow(pts, ls="-", lw=1.0):
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        if i == len(pts) - 2:
            arrow(a[0], a[1], b[0], b[1], ls=ls, lw=lw)
        else:
            ov.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=lw, ls=ls,
                    zorder=3)


def cap(x, y, t, fs=SUB_FS, col=MUTE, weight="normal", ha="center"):
    ov.text(x, y, t, fontsize=fs, color=col, ha=ha, weight=weight, zorder=4)


def badge(x, y, num):
    ov.scatter([x], [y], s=105, facecolors="white", edgecolors=INK,
               linewidths=1.0, zorder=5)
    ov.text(x, y, str(num), fontsize=7.2, weight="bold", color=INK,
            ha="center", va="center", zorder=6)


def out_tag(x, y):
    ov.add_patch(FancyBboxPatch((x - 0.020, y - 0.045), 0.040, 0.090,
                                boxstyle="round,pad=0.0015,rounding_size=0.008",
                                facecolor=INK, edgecolor="none", zorder=5))
    ov.text(x, y, "output", fontsize=6.6, weight="bold", color="white",
            ha="center", va="center", zorder=6)


def fc_net(xs, heights, slots, ells, cy, dims, descs, label_y, desc_y,
           node_s=18, out_colors=None):
    cols = []
    for x, h, n, ell in zip(xs, heights, slots, ells):
        ys = list(np.linspace(cy + h / 2, cy - h / 2, n))
        if ell:
            m = n // 2
            ys = ys[:m - 1] + ys[m + 1:]
            ov.scatter([x] * 3, [cy - 0.030, cy, cy + 0.030], s=2.2, c=[INK],
                       edgecolors="none", zorder=5)
        cols.append((x, ys))
    for (x0, ys0), (x1, ys1) in zip(cols, cols[1:]):
        for y0 in ys0:
            for y1 in ys1:
                ov.plot([x0, x1], [y0, y1], color=FAN, lw=0.35, zorder=1)
    for j, (x, ys) in enumerate(cols):
        last = j == len(cols) - 1 and out_colors
        fc = out_colors if last else ["white"] * len(ys)
        ov.scatter([x] * len(ys), ys, s=node_s, facecolors=fc, edgecolors=INK,
                   linewidths=0.65, zorder=4)
    for x, d, ds in zip(xs, dims, descs):
        ov.text(x, label_y, d, fontsize=7.8, weight="bold", color=INK,
                ha="center", zorder=4)
        ov.text(x, desc_y, ds, fontsize=6.0, color=MUTE, ha="center", zorder=4)
    return cols


def pixel_grid(x0, y0, w, gated=False, seed=5):
    rng = np.random.default_rng(seed)
    vals = rng.uniform(0.15, 1.0, (6, 6))
    hitm = vals > 0.45
    cw = w / 6
    ch = cw * AR
    for r in range(6):
        for c in range(6):
            fc_ = ("#e9ebee" if (gated and not hitm[r, c])
                   else plt.cm.viridis(vals[r, c] * 0.85))
            ov.add_patch(plt.Rectangle((x0 + c * cw, y0 + r * ch), cw * 0.9,
                                       ch * 0.9, facecolor=fc_,
                                       edgecolor="none", zorder=3))
    return ch * 6


# ============================ stage frames & titles ==============================
ov.add_patch(FancyBboxPatch((0.223, 0.02), 0.348, 0.955,
                            boxstyle="round,pad=0.002,rounding_size=0.008",
                            facecolor=PANEL_BG, edgecolor=PANEL_E, lw=1.0,
                            zorder=0.4))
ov.add_patch(FancyBboxPatch((0.583, 0.02), 0.412, 0.955,
                            boxstyle="round,pad=0.002,rounding_size=0.008",
                            facecolor=PANEL_BG, edgecolor=PANEL_E, lw=1.0,
                            zorder=0.4))
cap(0.108, 0.905, "Raw SERS data", fs=TITLE_FS, col=INK, weight="bold")
cap(0.397, 0.905, "Compositional unmixing", fs=TITLE_FS, col=INK, weight="bold")
cap(0.789, 0.905, "Concentration recovery", fs=TITLE_FS, col=INK, weight="bold")

# ============================ stage 1 — raw data =================================
gh = pixel_grid(0.012, CY - 0.20, 0.030)
cap(0.027, 0.235, "SERS map", fs=BOX_T, col=INK, weight="bold")
cap(0.027, 0.155, "one spectrum", fs=6.2, col=PLAIN)
cap(0.027, 0.085, "per pixel", fs=6.2, col=PLAIN)
arrow(0.048, CY, 0.062, CY)
pixel_grid(0.069, CY - 0.20, 0.030, gated=True)
cap(0.084, 0.235, "NNLS gate", fs=BOX_T, col=INK, weight="bold")
cap(0.084, 0.155, "substance pixels kept", fs=6.2, col=PLAIN)
cap(0.084, 0.085, "background dropped", fs=6.2, col=PLAIN)
arrow(0.100, CY, 0.114, CY)

axs = fig.add_axes([0.122, 0.40, 0.062, 0.34])
wx = np.linspace(500, 2500, 600)
spec = sum(a * np.exp(-0.5 * ((wx - c) / wdt) ** 2)
           for a, c, wdt in [(0.65, 800, 18), (0.5, 1010, 14), (0.8, 1270, 15),
                             (0.9, 1370, 14), (1.0, 1570, 16), (0.35, 1650, 25)])
axs.plot(wx, spec, color=INK, lw=0.9)
axs.set_xlim(500, 2500); axs.set_ylim(0, 1.2)
axs.set_xticks([]); axs.set_yticks([]); axs.patch.set_alpha(0)
for spn in axs.spines.values():
    spn.set_visible(False)
axs.spines["bottom"].set_visible(True); axs.spines["bottom"].set_linewidth(0.8)
cap(0.153, 0.235, r"pixel spectra x$_i$", fs=BOX_T, col=INK, weight="bold")
cap(0.153, 0.155, "log(1+I) · 1,290 pts", fs=6.2, col=PLAIN)
cap(0.153, 0.085, r"500–2500 cm$^{-1}$")
arrow(0.185, CY, 0.230, CY)

# ============================ stage 2 — unmixing =================================
badge(0.240, 0.80, 3)
ov.add_patch(FancyBboxPatch((0.232, 0.115), 0.160, 0.745,
                            boxstyle="round,pad=0.002,rounding_size=0.008",
                            facecolor=GREEN_BG, edgecolor=GREEN_E, lw=1.0,
                            zorder=0.6))
cap(0.312, 0.795, r"shared MLP f(x$_i$)", fs=BOX_T, col=INK,
    weight="bold")
fc_net([0.252, 0.290, 0.328, 0.366],
       heights=[0.44, 0.36, 0.27, 0.19],
       slots=[12, 9, 7, 4], ells=[True, True, False, False],
       cy=CY, dims=["1,290", "256", "64", "4"],
       descs=["input", "FC · ReLU · drop", "FC · ReLU", "softmax"],
       label_y=0.255, desc_y=0.195, out_colors=CLASS_COLORS)
cap(0.312, 0.140, "trained on prepared mixtures · weighted L1 · held-out by condition",
    fs=5.8)
cap(0.312, 0.082, "hidden widths selected by capacity sweep", fs=5.6)
for (yy, s_, c_) in zip(np.linspace(CY + 0.072, CY - 0.072, 4),
                        SUBS + ["BLK"], CLASS_COLORS):
    ov.text(0.373, yy, s_, fontsize=6.2, color=c_, ha="left", va="center",
            weight="bold", zorder=4)

arrow(0.394, CY, 0.406, CY)
ov.scatter([0.400], [CY], s=8, c=[INK], zorder=5)          # junction to stage 3

badge(0.407, 0.795, 4)
cap(0.448, 0.795, "pesticide maps", fs=BOX_T, col=INK, weight="bold")
CH_W = 0.0165
CH_H = CH_W * AR
for k, (s_, c_) in enumerate(zip(SUBS, [CO[s] for s in SUBS])):
    x0 = 0.412 + k * 0.021
    ov.add_patch(FancyBboxPatch((x0, CY - CH_H / 2), CH_W, CH_H,
                                boxstyle="round,pad=0.0008,rounding_size=0.003",
                                facecolor=c_, edgecolor="white", lw=0.6,
                                alpha=0.85, zorder=3))
    for f_ in (1 / 3, 2 / 3):
        ov.plot([x0, x0 + CH_W], [CY - CH_H / 2 + f_ * CH_H] * 2,
                color="white", lw=0.5, zorder=4)
        ov.plot([x0 + f_ * CH_W] * 2, [CY - CH_H / 2, CY + CH_H / 2],
                color="white", lw=0.5, zorder=4)
    ov.text(x0 + CH_W / 2, CY - CH_H / 2 - 0.055, s_, fontsize=5.8,
            color=c_, ha="center", weight="bold", zorder=4)
cap(0.443, 0.235, "where each pesticide sits", fs=PL_FS, col=PLAIN)
cap(0.443, 0.155, r"p$_i$ = softmax(f(x$_i$))")

arrow(0.478, CY, 0.492, CY)
badge(0.492, 0.795, 5)
out_tag(0.545, 0.915)
cap(0.536, 0.795, "map composition", fs=BOX_T, col=INK, weight="bold")
bx, bw = 0.496, 0.062
for k, (f0, f1) in enumerate([(0, 0.45), (0.45, 0.75), (0.75, 1.0)]):
    ov.add_patch(plt.Rectangle((bx + f0 * bw, CY - 0.055), (f1 - f0) * bw,
                               0.11, facecolor=[CO[s] for s in SUBS][k],
                               edgecolor="white", lw=0.6, zorder=3))
cap(0.527, 0.235, "% of each pesticide", fs=PL_FS, col=PLAIN)
cap(0.527, 0.155, r"$\bar{p}$ = pixel mean · BLK removed")
cap(0.508, 0.030, "absence gate: P(present) < 0.2 → reported 0 (ND)", fs=5.6)

# ============================ stage 3 — concentration ============================
# per-pixel scores and spectra feed the band read-out (routed under the panels)
elbow([(0.400, CY - 0.010), (0.400, 0.062), (0.635, 0.062), (0.635, 0.170)])
cap(0.518, 0.090, r"pixel scores p$_i$ and spectra I$_i$", fs=6.0)

badge(0.598, 0.80, 6)
box(0.590, 0.175, 0.092, 0.575,
    [("marker-band brightness", "t"),
     ("how strong is each band?", "p"),
     (r"I$_{eq,k}$ = $\Sigma_i$p$_{ik}$I$_i$($\nu_k$) / $\Sigma_i$p$_{ik}$", "e"),
     (r"$\nu_k$: 1570 / 1270 / 1367 cm$^{-1}$", "s")])
arrow(0.684, CY, 0.694, CY)
badge(0.702, 0.80, 7)
box(0.696, 0.385, 0.080, 0.365,
    [("inversion", "t"),
     ("brightness → µM", "p"),
     (r"C$_{cal}$ = 10$^{(I_{eq}-b)/a}$", "e")])
box(0.696, 0.045, 0.080, 0.24,
    [("measured calibration", "t"),
     (r"I = a·log$_{10}$C + b", "e")])
arrow(0.736, 0.29, 0.736, 0.38)
arrow(0.778, CY, 0.790, CY)

badge(0.800, 0.80, 8)
ov.add_patch(FancyBboxPatch((0.792, 0.115), 0.112, 0.745,
                            boxstyle="round,pad=0.002,rounding_size=0.008",
                            facecolor=GREEN_BG, edgecolor=GREEN_E, lw=1.0,
                            zorder=0.6))
cap(0.848, 0.795, "correction MLP", fs=BOX_T, col=INK, weight="bold")
fc_net([0.808, 0.834, 0.860, 0.886],
       heights=[0.30, 0.40, 0.27, 0.16],
       slots=[8, 9, 7, 3], ells=[True, True, False, False],
       cy=CY, dims=["12", "128", "32", "3"],
       descs=["z", "", "", r"$\Delta$"],
       label_y=0.255, desc_y=0.195, out_colors=[CO[s] for s in SUBS])
cap(0.847, 0.195, "FC · ReLU · dropout", fs=5.8)
cap(0.848, 0.140, "learns how mixtures distort the calibration", fs=5.8)

arrow(0.906, CY, 0.916, CY)
badge(0.924, 0.80, 9)
out_tag(0.985, 0.915)
box(0.918, 0.335, 0.076, 0.415,
    [("reported", "t"),
     ("concentration", "t"),
     (r"$\hat{C}_i$ = $\bar{p}_i$·C$_{declared}$", "e"),
     ("declared-total · validated", "s")], lw=1.4)
cap(0.918, 0.255, "no declared total — fallback:", fs=5.4, ha="left")
cap(0.918, 0.185, r"log$_{10}\hat{C}$ = log$_{10}$C$_{cal}$ + $\Delta$ (semi-quant)",
    fs=5.4, ha="left")
cap(0.918, 0.115, r"$\Delta$ in decades: +0.3 = ×2", fs=5.4, ha="left")
# 주 경로: ⑤ 조성 × 선언 총량 → ⑨ (패널 사이 거터로 내려가 바닥을 타고 간다)
elbow([(0.558, CY - 0.02), (0.577, CY - 0.02), (0.577, 0.032),
       (0.990, 0.032), (0.990, 0.330)])
cap(0.848, 0.048, r"main route: composition $\bar{p}_i$ × declared total", fs=5.6)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_architecture_dl.{ext}"), dpi=600,
                bbox_inches="tight", pad_inches=0.02, facecolor="white")
print("saved fig_architecture_dl.png/.pdf")
