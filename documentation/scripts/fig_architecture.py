"""Paper figure — deployed DL architecture, drawn Nature-style from real data.

Every visual element is real: the input map, its hit-pixel mean spectrum, the
per-pixel component maps and pooled composition (DQ24-TB12-TH6), and the measured
260821 calibration points. Network glyphs are layer columns with true dimensions.
Colors follow Pure/colors.json via labfig. Re-run after retraining to refresh.

Cache: build documentation/scripts/../..-scratch figcache.npz with fig_cache.py
(or pass CACHE env var); falls back to the scratchpad path used on 2026-08-26.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labfig
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.colors import LinearSegmentedColormap

labfig.setup()
CO = labfig.CO
INK = "#242b33"
MUTE = "#7a828c"
CONN = "#9aa1aa"

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
ux = np.unique(coords[:, 0]); uy = np.unique(coords[:, 1])
nx, ny = len(ux), len(uy)
gx = {v: i for i, v in enumerate(ux)}; gy = {v: i for i, v in enumerate(uy)}


def grid(vals):
    g = np.full((ny, nx), np.nan)
    for (x, y), v in zip(coords, vals):
        g[gy[y], gx[x]] = v
    return g


fig = plt.figure(figsize=(7.2, 3.6))
ov = fig.add_axes([0, 0, 1, 1], zorder=0)
ov.set_xlim(0, 1); ov.set_ylim(0, 1); ov.set_axis_off()

FS = 6.0          # base label size
FSS = 5.2         # small


def head(x, y, t):
    ov.text(x, y, t, fontsize=6.6, weight="bold", color=INK, ha="left")


def cap(x, y, t, fs=FS, col=MUTE, ha="center", **kw):
    ov.text(x, y, t, fontsize=fs, color=col, ha=ha, **kw)


def arrow(x0, y0, x1, y1, ls="-", con="arc3,rad=0"):
    ov.annotate("", (x1, y1), (x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=CONN, lw=0.8, linestyle=ls,
                                connectionstyle=con, shrinkA=1, shrinkB=1))


def layer_glyph(cx, cy, dims, labels, out_colors=None, w=0.016, gap=0.030,
                hmax=0.30, label_y=None):
    """Vertical layer columns, height ~ log(units), trapezoid connectors."""
    hs = [hmax * (0.30 + 0.70 * np.log(d) / np.log(max(dims))) for d in dims]
    xs = [cx + (i - (len(dims) - 1) / 2) * gap for i in range(len(dims))]
    for i in range(len(dims) - 1):
        ov.add_patch(Polygon([(xs[i] + w / 2, cy - hs[i] / 2),
                              (xs[i] + w / 2, cy + hs[i] / 2),
                              (xs[i + 1] - w / 2, cy + hs[i + 1] / 2),
                              (xs[i + 1] - w / 2, cy - hs[i + 1] / 2)],
                             closed=True, facecolor="#e4e8ee", edgecolor="none",
                             zorder=1))
    for i, (x, h, d) in enumerate(zip(xs, hs, dims)):
        last = i == len(dims) - 1 and out_colors
        ov.add_patch(FancyBboxPatch((x - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.002,rounding_size=0.006",
                                    facecolor="white" if not last else "none",
                                    edgecolor=INK, lw=0.7, zorder=2))
        if last:
            k = len(out_colors)
            yy = [cy + h / 2 - (j + 0.5) * h / k for j in range(k)]
            ov.scatter([x] * k, yy, s=7, c=out_colors, edgecolors="none", zorder=3)
        cap(x, label_y if label_y is not None else cy - hmax / 2 - 0.045,
            labels[i], fs=FSS, col=INK)
    return xs[0] - w, xs[-1] + w


# =================== top band — composition ======================================
head(0.035, 0.955, "Composition — one shared per-pixel model")

# input map (real total intensity)
axm = fig.add_axes([0.035, 0.615, 0.095, 0.19]); axm.set_axis_off()
axm.imshow(grid(total), cmap="viridis", interpolation="nearest", aspect="auto")
cap(0.0825, 0.575, "SERS map", col=INK)
cap(0.0825, 0.538, "DQ24 : TBZ12 : THI6 µM", fs=FSS)

# hit-pixel spectrum (real), marker bands ticked in substance colors
axs = fig.add_axes([0.175, 0.615, 0.135, 0.21])
axs.plot(wn, hitspec, color=INK, lw=0.6)
for s, b in zip(SUBS, (1570, 1270, 1367)):
    axs.axvline(b, color=CO[s], lw=0.7, alpha=0.85, ymax=0.22)
axs.set_xlim(500, 2500); axs.set_xticks([]); axs.set_yticks([])
for sp in axs.spines.values():
    sp.set_visible(False)
axs.spines["bottom"].set_visible(True); axs.spines["bottom"].set_linewidth(0.5)
cap(0.2425, 0.575, "hit-pixel spectra · log(1+x)", col=INK)
cap(0.2425, 0.538, "NNLS gate keeps hit pixels", fs=FSS)

# shared MLP glyph
layer_glyph(0.40, 0.725, [1290, 256, 64, 4], ["1,290", "256", "64", "4"],
            out_colors=[CO["DQ"], CO["TBZ"], CO["THI"], "#9aa3ad"], label_y=0.545)
cap(0.40, 0.90, "shared MLP → softmax", col=INK)
cap(0.40, 0.512, "DQ · TBZ · THI · BLK", fs=FSS)

# per-pixel component maps (real)
for i, s in enumerate(SUBS):
    ax = fig.add_axes([0.50 + i * 0.082, 0.635, 0.072, 0.144]); ax.set_axis_off()
    cm = LinearSegmentedColormap.from_list("m", ["#ffffff", CO[s]])
    ax.imshow(grid(ratio_nb[:, i]), cmap=cm, vmin=0, vmax=1,
              interpolation="nearest", aspect="auto")
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.4); sp.set_edgecolor("#c9ced4")
    cap(0.536 + i * 0.082, 0.575, s, fs=FSS, col=CO[s], weight="bold")
cap(0.618, 0.90, "per-pixel component maps", col=INK)

# pooled composition (real numbers) as one stacked bar
comp = ratio_nb[hit].mean(0); comp = comp / comp.sum()
axb = fig.add_axes([0.795, 0.66, 0.175, 0.075]); axb.set_axis_off()
axb.set_xlim(0, 1); axb.set_ylim(0, 1)
x0 = 0.0
for i, s in enumerate(SUBS):
    axb.add_patch(plt.Rectangle((x0, 0.15), comp[i], 0.7, facecolor=CO[s],
                                edgecolor="white", lw=0.4))
    if comp[i] > 0.08:
        axb.text(x0 + comp[i] / 2, 0.5, f"{comp[i]*100:.0f}%", ha="center",
                 va="center", fontsize=5.4, color="white", weight="bold")
    x0 += comp[i]
cap(0.8825, 0.90, "map composition", col=INK)
cap(0.8825, 0.575, "mean pool · BLK removed · renormalised", fs=FSS)

arrow(0.132, 0.71, 0.172, 0.71)
arrow(0.312, 0.71, 0.342, 0.71)
arrow(0.462, 0.71, 0.496, 0.71)
arrow(0.752, 0.71, 0.792, 0.71)

# =================== bottom band — quantification ================================
head(0.035, 0.44, "Quantification — calibration-anchored residual")

# measured calibration (real points + log-linear fits)
axc = fig.add_axes([0.175, 0.09, 0.135, 0.27])
for s in SUBS:
    c, sig = CAL[s]
    axc.plot(np.log10(c), sig, "o", ms=1.6, color=CO[s])
    a, b = np.polyfit(np.log10(c), sig, 1)
    xx = np.linspace(np.log10(c.min()), np.log10(c.max()), 20)
    axc.plot(xx, a * xx + b, color=CO[s], lw=0.7, alpha=0.8)
axc.set_xticks([]); axc.set_yticks([])
for sp in axc.spines.values():
    sp.set_linewidth(0.5)
axc.spines["top"].set_visible(False); axc.spines["right"].set_visible(False)
axc.set_xlabel(r"log$_{10}$C", fontsize=FSS, labelpad=1)
axc.set_ylabel(r"I$_{band}$", fontsize=FSS, labelpad=1)
cap(0.2425, 0.415, "measured calibration (embedded)", col=INK)
cap(0.2425, 0.032, r"I = a·log$_{10}$C + b  →  C$_{cal}$", fs=FSS)

# residual net glyph
layer_glyph(0.415, 0.245, [12, 128, 32, 3], ["12", "128", "32", "3"],
            out_colors=[CO["DQ"], CO["TBZ"], CO["THI"]], hmax=0.22, label_y=0.105)
cap(0.415, 0.415, "residual net", col=INK)
cap(0.415, 0.058, r"map features → $\Delta$log$_{10}$C", fs=FSS)

# concentration output chip
ov.add_patch(FancyBboxPatch((0.535, 0.155), 0.175, 0.165,
                            boxstyle="round,pad=0.006,rounding_size=0.012",
                            facecolor="#f2f6f2", edgecolor=INK, lw=0.8))
cap(0.6225, 0.272, "concentration (µM)", col=INK, weight="bold")
cap(0.6225, 0.222, r"C = C$_{cal}$ · 10$^{\Delta}$", fs=FS, col=INK)
cap(0.6225, 0.175, "validated window · OOD flags", fs=FSS)

# optional known-total gate (dashed, constrained)
ov.add_patch(FancyBboxPatch((0.795, 0.155), 0.175, 0.165,
                            boxstyle="round,pad=0.006,rounding_size=0.012",
                            facecolor="none", edgecolor=MUTE, lw=0.7,
                            linestyle=(0, (3, 2))))
cap(0.8825, 0.272, "known total (optional)", col=INK, weight="bold")
cap(0.8825, 0.222, r"C$_i$ = p$_i$ × C$_{total}$", fs=FS, col=INK)
cap(0.8825, 0.175, "declared metadata — constrained", fs=FSS)

# connectors between bands
arrow(0.445, 0.545, 0.295, 0.395, con="arc3,rad=-0.18")
cap(0.455, 0.482, r"probability-weighted band signal I$_{eq}$", fs=FSS, ha="left")
arrow(0.312, 0.235, 0.352, 0.235)
arrow(0.478, 0.235, 0.532, 0.235)
arrow(0.8825, 0.645, 0.8825, 0.335, ls=(0, (3, 2)))

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_architecture_dl.{ext}"), dpi=600,
                bbox_inches="tight", pad_inches=0.02, facecolor="white")
print("saved fig_architecture_dl.png/.pdf")
