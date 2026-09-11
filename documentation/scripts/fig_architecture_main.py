"""Manuscript architecture — the essentials only.

Main-text variant of fig_architecture: the calibration/residual machinery
(steps 6-8 of the SI version) collapses into one fallback box, the declared-
total route is the single highlighted output, and per-box equations are cut
to one line each. The full pipeline stays in fig_architecture_dl (SI).
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

fig = plt.figure(figsize=(10.8, 2.9))
ov = fig.add_axes([0, 0, 1, 1], zorder=0)
ov.set_xlim(0, 1); ov.set_ylim(0, 1); ov.set_axis_off()
AR = 10.8 / 2.9
TITLE_FS, BOX_T, EQ_FS, SUB_FS, PL_FS = 12.5, 8.6, 7.6, 6.4, 7.0
CY = 0.55


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


def cap(x, y, t, fs=SUB_FS, col=MUTE, weight="normal", ha="center"):
    ov.text(x, y, t, fontsize=fs, color=col, ha=ha, weight=weight, zorder=4)


def out_tag(x, y):
    ov.add_patch(FancyBboxPatch((x - 0.026, y - 0.045), 0.052, 0.090,
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


# ── panels ───────────────────────────────────────────────────────────────
ov.add_patch(FancyBboxPatch((0.278, 0.02), 0.432, 0.955,
                            boxstyle="round,pad=0.002,rounding_size=0.008",
                            facecolor=PANEL_BG, edgecolor=PANEL_E, lw=1.0,
                            zorder=0.4))
ov.add_patch(FancyBboxPatch((0.722, 0.02), 0.272, 0.955,
                            boxstyle="round,pad=0.002,rounding_size=0.008",
                            facecolor=PANEL_BG, edgecolor=PANEL_E, lw=1.0,
                            zorder=0.4))
cap(0.135, 0.905, "Raw SERS data", fs=TITLE_FS, col=INK, weight="bold")
cap(0.494, 0.905, "Compositional unmixing", fs=TITLE_FS, col=INK, weight="bold")
cap(0.858, 0.905, "Concentration", fs=TITLE_FS, col=INK, weight="bold")

# ── raw data ─────────────────────────────────────────────────────────────
pixel_grid(0.016, CY - 0.20, 0.040)
cap(0.036, 0.235, "SERS map", fs=BOX_T, col=INK, weight="bold")
cap(0.036, 0.155, "one spectrum", fs=6.2, col=PLAIN)
cap(0.036, 0.085, "per pixel", fs=6.2, col=PLAIN)
arrow(0.062, CY, 0.078, CY)
pixel_grid(0.086, CY - 0.20, 0.040, gated=True)
cap(0.106, 0.235, "NNLS gate", fs=BOX_T, col=INK, weight="bold")
cap(0.106, 0.155, "background", fs=6.2, col=PLAIN)
cap(0.106, 0.085, "dropped", fs=6.2, col=PLAIN)
arrow(0.132, CY, 0.148, CY)

axs = fig.add_axes([0.157, 0.40, 0.082, 0.34])
wx = np.linspace(500, 2500, 600)
spec = sum(a * np.exp(-0.5 * ((wx - c) / wdt) ** 2)
           for a, c, wdt in [(0.65, 800, 18), (0.5, 1010, 14), (0.8, 1270, 15),
                             (0.9, 1367, 14), (1.0, 1570, 16), (0.35, 1650, 25)])
axs.plot(wx, spec, color=INK, lw=0.9)
axs.set_xlim(500, 2500); axs.set_ylim(0, 1.2)
axs.set_xticks([]); axs.set_yticks([]); axs.patch.set_alpha(0)
for spn in axs.spines.values():
    spn.set_visible(False)
axs.spines["bottom"].set_visible(True); axs.spines["bottom"].set_linewidth(0.8)
cap(0.198, 0.235, r"pixel spectra x$_i$", fs=BOX_T, col=INK, weight="bold")
cap(0.198, 0.155, "1,290 pts · 500–2500 cm$^{-1}$", fs=6.2, col=PLAIN)
arrow(0.244, CY, 0.286, CY)

# ── unmixing ─────────────────────────────────────────────────────────────
ov.add_patch(FancyBboxPatch((0.290, 0.115), 0.200, 0.745,
                            boxstyle="round,pad=0.002,rounding_size=0.008",
                            facecolor=GREEN_BG, edgecolor=GREEN_E, lw=1.0,
                            zorder=0.6))
cap(0.390, 0.795, r"shared MLP f(x$_i$)", fs=BOX_T, col=INK, weight="bold")
fc_net([0.315, 0.362, 0.409, 0.456],
       heights=[0.44, 0.36, 0.27, 0.19],
       slots=[12, 9, 7, 4], ells=[True, True, False, False],
       cy=CY, dims=["1,290", "256", "64", "4"],
       descs=["input", "FC · BN · ReLU\nDropout 0.15", "FC · ReLU", "softmax"],
       label_y=0.255, desc_y=0.195, out_colors=CLASS_COLORS)
cap(0.390, 0.135, "trained on prepared mixtures · held-out by condition", fs=5.8)
for (yy, s_, c_) in zip(np.linspace(CY + 0.072, CY - 0.072, 4),
                        SUBS + ["BLK"], CLASS_COLORS):
    ov.text(0.464, yy, s_, fontsize=6.2, color=c_, ha="left", va="center",
            weight="bold", zorder=4)
arrow(0.492, CY, 0.508, CY)

out_tag(0.688, 0.915)
cap(0.605, 0.795, "map composition", fs=BOX_T, col=INK, weight="bold")
bx, bw = 0.520, 0.086
for k, (f0, f1) in enumerate([(0, 0.45), (0.45, 0.75), (0.75, 1.0)]):
    ov.add_patch(plt.Rectangle((bx + f0 * bw, CY - 0.07), (f1 - f0) * bw,
                               0.14, facecolor=[CO[s] for s in SUBS][k],
                               edgecolor="white", lw=0.6, zorder=3))
for k, (fx, s_) in enumerate(zip((0.22, 0.60, 0.87), SUBS)):
    ov.text(bx + fx * bw, CY, f"{s_}", fontsize=6.0, color="white",
            ha="center", va="center", weight="bold", zorder=4)
cap(0.563, 0.320, r"$\bar{p}_i$ = pixel mean · BLK removed", fs=PL_FS,
    col=PLAIN)
cap(0.563, 0.235, "absent component → reported 0", fs=PL_FS, col=PLAIN)
cap(0.563, 0.155, "(absence gate, P < 0.2)", fs=SUB_FS)
arrow(0.612, CY, 0.726, CY)
cap(0.668, CY + 0.06, "gated pixels · composition", fs=6.2, col=INK)

# ── concentration: 신경망(조성 MLP와 같은 문법으로 시각화) + 캐스케이드 ──
# Δ 표기·검량곡선 블록은 뺐다: Ccal이 상수로 고정임이 확인돼 넷이 12피처에서
# µM을 직접 낸다는 서술이 정확하고, 곡선은 라이브러리 데이터로 승격됐다.
out_tag(0.958, 0.915)
cap(0.858, 0.795, "concentration net g(map)", fs=BOX_T, col=INK,
    weight="bold")
fc_net([0.772, 0.815, 0.858, 0.901],
       heights=[0.34, 0.30, 0.22, 0.13],
       slots=[8, 7, 5, 3], ells=[True, True, True, False],
       cy=CY, dims=["12", "128", "32", "3"],
       descs=["map\nfeatures", "FC · BN · ReLU\nDropout 0.25", "FC\nReLU", "Δlog10\n→ µM"],
       label_y=0.32, desc_y=0.265, node_s=14,
       out_colors=[CO[s] for s in SUBS])
for (yy, s_, c_) in zip(np.linspace(CY + 0.045, CY - 0.045, 3), SUBS,
                        [CO[s] for s in SUBS]):
    ov.text(0.908, yy, s_, fontsize=6.2, color=c_, ha="left", va="center",
            weight="bold", zorder=4)
cap(0.858, 0.845, "12 features: log10 Ccal ·3 · composition ·3 · log1p band ·3 · log-total p10/50/90",
    fs=5.4)
cap(0.858, 0.200, "C = Ccal · 10^Δ ;  Ccal = log-linear band calibration (9–144 µM)",
    fs=5.6, col=PLAIN)
cap(0.858, 0.163, "reported only if the nearest training map is within z-distance 3 · else refused",
    fs=5.4, col=MUTE)
box(0.752, 0.030, 0.212, 0.095,
    [("declared total (optional)", "t"),
     (r"$\hat{C}_i$ = $\bar{p}_i$ · C$_{total}$ — overlay when known", "s")],
    ls="--", lw=0.9)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_architecture_main.{ext}"), dpi=600,
                bbox_inches="tight", pad_inches=0.02, facecolor="white")
print("saved fig_architecture_main.png/.pdf")
