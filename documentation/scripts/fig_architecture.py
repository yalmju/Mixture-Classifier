"""Model-architecture schematic in THIS paper's figure language.

Matches the integrated benchmark figure style: white ground, flat boxes, bold
"(a)"-prefixed panel titles, despined mini-plots, grey + green (learned blocks)
+ substance colors. Every stage carries its actual quantity/equation, so the
algorithm reads directly: x_i -> p_i = softmax(f(x_i)) -> p-bar; I_eq -> C_cal
-> z(12) -> dlog10C -> C. All insets are real data (DQ24-TB12-TH6 map, hit
spectrum, per-pixel maps, pooled composition, 260821 calibration).

Cache: figcache.npz from fig_cache.py (CACHE env var overrides).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labfig
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

labfig.setup()
CO = labfig.CO
INK = "black"
MUTE = "#6a7178"
GREEN = "#e1f0e8"          # learned blocks — tint of the benchmark's MLP green
GREEN_E = "#2e8b62"

CACHE = os.environ.get("CACHE") or (
    r"C:\Users\SL\AppData\Local\Temp\claude"
    r"\S--Google-Drive--------github-Mixture-Classifier--claude-worktrees-"
    r"unmixr-workflow-changes-d093c0\fb8e0483-0af9-407c-bd5b-211b8f9da43d"
    r"\scratchpad\figcache.npz")
z = np.load(CACHE, allow_pickle=True)
coords, ratio_nb, hit = z["coords"], z["ratio_nb"], z["hit"]
total, wn, hitspec = z["total"], z["wn"], z["hitspec"]
SUBS = [str(s) for s in z["comps"]]
CAL = {s: z[f"cal_{s}"] for s in SUBS}
BANDS_TXT = "1570 / 1270 / 1370"
ux = np.unique(coords[:, 0]); uy = np.unique(coords[:, 1])
nx, ny = len(ux), len(uy)
gx = {v: i for i, v in enumerate(ux)}; gy = {v: i for i, v in enumerate(uy)}


def grid(vals):
    g = np.full((ny, nx), np.nan)
    for (x, y), v in zip(coords, vals):
        g[gy[y], gx[x]] = v
    return g


fig = plt.figure(figsize=(7.3, 4.05))
ov = fig.add_axes([0, 0, 1, 1], zorder=0)
ov.set_xlim(0, 1); ov.set_ylim(0, 1); ov.set_axis_off()

BOX_T, EQ_FS, SUB_FS = 7.6, 7.0, 6.2


def box(x, y, w, h, lines, fill="white", edge=INK, lw=0.9, ls="-"):
    """Flat box: first line bold, equations in ink, notes in grey."""
    ov.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.002,rounding_size=0.008",
                                facecolor=fill, edgecolor=edge, linewidth=lw,
                                linestyle=ls, zorder=2))
    n = len(lines)
    step = h / (n + 0.6)
    for i, (t, kind) in enumerate(lines):
        fs = BOX_T if kind == "t" else EQ_FS if kind == "e" else SUB_FS
        col = INK if kind in ("t", "e") else MUTE
        ov.text(x + w / 2, y + h - (i + 0.85) * step, t, fontsize=fs,
                weight="bold" if kind == "t" else "normal", color=col,
                ha="center", va="center", zorder=3)


def arrow(x0, y0, x1, y1, label=None, ls="-", lab_dy=0.016, lab_fs=SUB_FS):
    ov.annotate("", (x1, y1), (x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8,
                                linestyle=ls, mutation_scale=7,
                                shrinkA=1, shrinkB=1), zorder=3)
    if label:
        ov.text((x0 + x1) / 2, max(y0, y1) + lab_dy, label, fontsize=lab_fs,
                color=MUTE, ha="center", zorder=3)


def elbow(pts, ls="-", label=None, lab_xy=None):
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        if i == len(pts) - 2:
            ov.annotate("", b, a, arrowprops=dict(arrowstyle="-|>", color=INK,
                                                  lw=0.8, linestyle=ls,
                                                  mutation_scale=7,
                                                  shrinkA=0, shrinkB=1),
                        zorder=3)
        else:
            ov.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=0.8, ls=ls,
                    zorder=3)
    if label:
        ov.text(*lab_xy, label, fontsize=SUB_FS, color=MUTE, ha="center",
                zorder=3)


def cap(x, y, t, fs=SUB_FS, col=MUTE, weight="normal"):
    ov.text(x, y, t, fontsize=fs, color=col, ha="center", weight=weight,
            zorder=3)


# ===================== (a) pixel-wise composition network ========================
ov.text(0.008, 0.952, "(a) Pixel-wise composition network", fontsize=10,
        weight="bold", color=INK, ha="left")
ROW_A = 0.755

axm = fig.add_axes([0.015, 0.660, 0.100, 0.205]); axm.set_axis_off()
axm.imshow(grid(total), cmap="viridis", interpolation="nearest", aspect="auto")
cap(0.065, 0.625, "SERS map", fs=BOX_T, col=INK)
cap(0.065, 0.588, "DQ24 : TBZ12 : THI6 µM")

arrow(0.118, ROW_A, 0.148, ROW_A)
cap(0.133, ROW_A + 0.035, "NNLS\nhit gate")

axs = fig.add_axes([0.152, 0.660, 0.120, 0.205])
axs.plot(wn, hitspec, color=INK, lw=0.6)
topv = float(np.nanmax(hitspec))
for s, b in zip(SUBS, (1570, 1270, 1370)):
    axs.plot([b], [topv * 1.07], marker="v", ms=2.4, color=CO[s], clip_on=False)
axs.set_xlim(500, 2500); axs.set_ylim(0, topv * 1.14)
axs.set_xticks([]); axs.set_yticks([]); axs.patch.set_alpha(0)
for spn in axs.spines.values():
    spn.set_visible(False)
axs.spines["bottom"].set_visible(True); axs.spines["bottom"].set_linewidth(0.6)
cap(0.212, 0.625, r"x$_i$ = log(1+I$_i$)", fs=BOX_T, col=INK)
cap(0.212, 0.588, "hit-pixel spectra · 1,290 ch")

arrow(0.278, ROW_A, 0.306, ROW_A)
box(0.310, 0.660, 0.150, 0.205,
    [("shared MLP", "t"),
     ("1290 – 256 – 64 – 4", "e"),
     ("ReLU · dropout · softmax", "s"),
     (r"p$_i$ = softmax(f(x$_i$))", "e")],
    fill=GREEN, edge=GREEN_E)
# class dots as the MLP box's caption row
for k, (s_, c_) in enumerate(list(zip(SUBS, [CO[s] for s in SUBS]))
                             + [("BLK", "#9aa3ad")]):
    xx = 0.343 + k * 0.028
    ov.scatter([xx], [0.628], s=10, c=[c_], zorder=4, edgecolors="white",
               linewidths=0.3)
    ov.text(xx, 0.598, s_, fontsize=5.2, color=MUTE, ha="center", zorder=4)

arrow(0.462, ROW_A, 0.492, ROW_A)
for i, s in enumerate(SUBS):
    ax = fig.add_axes([0.497 + i * 0.088, 0.665, 0.076, 0.155]); ax.set_axis_off()
    cm = LinearSegmentedColormap.from_list("m", ["#ffffff", CO[s]])
    ax.imshow(grid(ratio_nb[:, i]), cmap=cm, vmin=0, vmax=1,
              interpolation="nearest", aspect="auto")
    ov.text(0.535 + i * 0.088, 0.636, s, fontsize=SUB_FS, color=CO[s],
            ha="center", weight="bold")
cap(0.623, 0.845, r"per-pixel probability maps p$_i$", fs=BOX_T, col=INK)

arrow(0.762, ROW_A, 0.792, ROW_A)
comp = ratio_nb[hit].mean(0); comp = comp / comp.sum()
axb = fig.add_axes([0.798, 0.735, 0.180, 0.062]); axb.set_axis_off()
axb.set_xlim(0, 1); axb.set_ylim(0, 1); axb.patch.set_alpha(0)
x0 = 0.0
for i, s in enumerate(SUBS):
    axb.add_patch(plt.Rectangle((x0, 0.06), comp[i], 0.88, facecolor=CO[s],
                                edgecolor="white", lw=0.5))
    if comp[i] > 0.08:
        axb.text(x0 + comp[i] / 2, 0.5, f"{comp[i]*100:.0f}%", ha="center",
                 va="center", fontsize=6.0, color="white", weight="bold")
    x0 += comp[i]
cap(0.888, 0.828, "map composition", fs=BOX_T, col=INK)
cap(0.888, 0.688, r"$\bar{p}$ = mean of p$_i$ over hit pixels", fs=SUB_FS)
cap(0.888, 0.655, "BLK removed · renormalised", fs=SUB_FS)

# ===================== (b) calibration-anchored quantification ===================
ov.text(0.008, 0.520, "(b) Calibration-anchored residual quantification",
        fontsize=10, weight="bold", color=INK, ha="left")
ROW_B = 0.300

axc = fig.add_axes([0.020, 0.185, 0.105, 0.235])
for s in SUBS:
    c, sig = CAL[s]
    lv = np.unique(c)
    ms = np.array([sig[c == v].mean() for v in lv])
    a_, b_ = np.polyfit(np.log10(c), sig, 1)
    xx = np.linspace(np.log10(lv.min()), np.log10(lv.max()), 20)
    axc.plot(xx, a_ * xx + b_, color=CO[s], lw=0.8, alpha=0.9, zorder=2)
    axc.plot(np.log10(lv), ms, "o", ms=2.0, color=CO[s], zorder=3,
             markeredgecolor="white", markeredgewidth=0.3)
axc.patch.set_alpha(0)
axc.set_xticks([1, 2]); axc.set_xticklabels(["10", "100"], fontsize=SUB_FS)
axc.set_yticks([])
for spn in ("top", "right"):
    axc.spines[spn].set_visible(False)
for spn in ("left", "bottom"):
    axc.spines[spn].set_linewidth(0.6)
axc.tick_params(length=1.6, width=0.6, pad=1)
axc.set_xlabel("C (µM)", fontsize=SUB_FS, labelpad=1)
axc.set_ylabel(r"I$_{band}$", fontsize=SUB_FS, labelpad=1)
cap(0.072, 0.448, "measured calibration", fs=BOX_T, col=INK)
cap(0.072, 0.118, r"I = a·log$_{10}$C + b", fs=SUB_FS)

box(0.152, 0.225, 0.170, 0.155,
    [("band read-out", "t"),
     (r"I$_{eq,k}$ = $\Sigma_i$ p$_{ik}$I$_i$($\nu_k$) / $\Sigma_i$ p$_{ik}$", "e"),
     (r"$\nu_k$: %s cm$^{-1}$" % BANDS_TXT, "s")])
box(0.352, 0.225, 0.140, 0.155,
    [("inversion", "t"),
     (r"C$_{cal}$ = 10$^{(I_{eq}-b)/a}$", "e"),
     ("clipped to calibrated range", "s")])
box(0.522, 0.225, 0.160, 0.155,
    [("map features z (12)", "t"),
     (r"log$_{10}$C$_{cal}$ · $\bar{p}$ · log I$_{eq}$", "e"),
     ("intensity quantiles P10/50/90", "s")])
box(0.712, 0.225, 0.130, 0.155,
    [("residual MLP", "t"),
     ("12 – 128 – 32 – 3", "e"),
     (r"$\Delta$log$_{10}$C, clipped ±2", "s")],
    fill=GREEN, edge=GREEN_E)
box(0.872, 0.225, 0.115, 0.155,
    [(r"$\hat{C}$ (µM)", "t"),
     (r"$\hat{C}$ = C$_{cal}$·10$^{\Delta}$", "e"),
     ("validated window · OOD", "s")], lw=1.2)

arrow(0.324, ROW_B, 0.348, ROW_B)
arrow(0.494, ROW_B, 0.518, ROW_B)
arrow(0.684, ROW_B, 0.708, ROW_B)
arrow(0.844, ROW_B, 0.868, ROW_B)
elbow([(0.072, 0.175), (0.072, 0.198), (0.422, 0.198), (0.422, 0.222)],
      label="a, b", lab_xy=(0.250, 0.205))

# what feeds the band read-out: per-pixel probabilities and raw intensities
elbow([(0.664, 0.655), (0.664, 0.468), (0.280, 0.468), (0.280, 0.395)],
      label=r"p$_i$ · I$_i$  (hit pixels)", lab_xy=(0.492, 0.482))

# optional known-total gate (declared metadata; constrained)
box(0.542, 0.040, 0.330, 0.110,
    [(r"known total (optional):  $\hat{C}_i$ = $\bar{p}_i$ · C$_{total}$", "e"),
     ("declared sample-prep metadata — reported as constrained", "s")],
    edge=MUTE, lw=0.8, ls=(0, (4, 2)))
elbow([(0.888, 0.648), (0.995, 0.648), (0.995, 0.095), (0.876, 0.095)],
      ls=(0, (4, 2)))

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_architecture_dl.{ext}"), dpi=600,
                bbox_inches="tight", pad_inches=0.02, facecolor="white")
print("saved fig_architecture_dl.png/.pdf")
