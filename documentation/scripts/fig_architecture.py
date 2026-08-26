"""Paper figure — deployed DL architecture in the classic ML-paper idiom:
3-D layer slabs with FC bands, pastel module panels, input column on the left.

All data insets are real (DQ24-TB12-TH6 map, hit-pixel spectrum, per-pixel
component maps, pooled composition, measured 260821 calibration series).
Cache: figcache.npz from fig_cache.py (CACHE env var overrides).
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
INK = "#20262e"
MUTE = "#767f8a"
CONN = "#4d5560"

P_IN = "#edf0f5"          # input column
P_COMP = "#e9edfa"        # composition module
P_QUANT = "#e6f3ec"       # quantification module
P_OUT = "#fdf3dc"         # outputs
SLAB_C = ("#c7d2f0", "#aebbe6", "#93a3d8")     # face, top, side (periwinkle)
SLAB_G = ("#bfe3cd", "#a8d8bd", "#8cc7a6")     # quantification greens
FC_BAND = "#f4cf5a"

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


fig = plt.figure(figsize=(7.4, 3.9))
ov = fig.add_axes([0, 0, 1, 1], zorder=0)
ov.set_xlim(0, 1); ov.set_ylim(0, 1); ov.set_axis_off()

TITLE_FS, NAME_FS, SUB_FS, TINY = 6.8, 6.0, 5.1, 4.7
DX, DY = 0.0065, 0.011                     # 3-D extrusion offsets


def panel(x, y, w, h, fill, title=None):
    ov.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.014",
                                facecolor=fill, edgecolor="none", zorder=0.5))
    if title:
        ov.text(x + 0.010, y + h - 0.042, title, fontsize=TITLE_FS,
                weight="bold", color=INK, ha="left", zorder=3)


def slab(cx, cy, w, h, cols):
    face, topc, side = cols
    x0, y0 = cx - w / 2, cy - h / 2
    ov.add_patch(Polygon([(x0 + w, y0), (x0 + w + DX, y0 + DY),
                          (x0 + w + DX, y0 + h + DY), (x0 + w, y0 + h)],
                         closed=True, facecolor=side, edgecolor=INK, lw=0.4,
                         zorder=2))
    ov.add_patch(Polygon([(x0, y0 + h), (x0 + DX, y0 + h + DY),
                          (x0 + w + DX, y0 + h + DY), (x0 + w, y0 + h)],
                         closed=True, facecolor=topc, edgecolor=INK, lw=0.4,
                         zorder=2))
    ov.add_patch(plt.Rectangle((x0, y0), w, h, facecolor=face, edgecolor=INK,
                               lw=0.5, zorder=2.1))


def fc_band(x0, h0, x1, h1, cy, label="FC"):
    ov.add_patch(Polygon([(x0, cy - h0 / 2), (x0, cy + h0 / 2),
                          (x1, cy + h1 / 2), (x1, cy - h1 / 2)],
                         closed=True, facecolor=FC_BAND, alpha=0.40,
                         edgecolor="none", zorder=1.5))
    ov.text((x0 + x1) / 2, cy, label, fontsize=4.2, color="#8a6d1d",
            ha="center", va="center", zorder=2.5, rotation=90)


def slab_stack(cx, cy, dims, labels, cols, hmax, last_classes=None, w=0.016,
               gap=0.052):
    hs = [hmax * (0.32 + 0.68 * np.log(d) / np.log(max(dims))) for d in dims]
    xs = [cx + (i - (len(dims) - 1) / 2) * gap for i in range(len(dims))]
    for i in range(len(dims) - 1):
        fc_band(xs[i] + w / 2 + DX, hs[i], xs[i + 1] - w / 2, hs[i + 1], cy)
    for i, (x, h) in enumerate(zip(xs, hs)):
        if i == len(dims) - 1 and last_classes:
            k = len(last_classes)
            ch = h / k
            for j, c in enumerate(last_classes):
                slab(x, cy + h / 2 - (j + 0.5) * ch, w, ch * 0.86,
                     (c, c, c))
        else:
            slab(x, cy, w, h, cols)
        ov.text(x + DX / 2, cy - hmax / 2 - 0.038, labels[i], fontsize=TINY,
                color=INK, ha="center", zorder=3)
    return xs[0] - w, xs[-1] + w


def arrow(x0, y0, x1, y1, ls="-", lw=0.9):
    ov.annotate("", (x1, y1), (x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=CONN, lw=lw,
                                linestyle=ls, mutation_scale=7,
                                shrinkA=1, shrinkB=1), zorder=3)


# ============================ panels =============================================
panel(0.015, 0.050, 0.118, 0.905, P_IN)
panel(0.150, 0.525, 0.835, 0.430, P_COMP, "Composition module — shared per-pixel MLP")
panel(0.150, 0.045, 0.505, 0.415, P_QUANT, "Quantification module — calibration-anchored residual")
panel(0.672, 0.045, 0.313, 0.415, P_OUT, "Output")

# ---------------------------- input column --------------------------------------
ov.text(0.074, 0.918, "Input", fontsize=TITLE_FS, weight="bold", color=INK,
        ha="center")
axm = fig.add_axes([0.030, 0.665, 0.088, 0.175]); axm.set_axis_off()
axm.imshow(grid(total), cmap="viridis", interpolation="nearest", aspect="auto")
ov.text(0.074, 0.630, "SERS map", fontsize=SUB_FS, color=INK, ha="center")
ov.text(0.074, 0.598, "DQ24:TBZ12:THI6 µM", fontsize=TINY, color=MUTE, ha="center")
arrow(0.074, 0.575, 0.074, 0.525)
ov.text(0.078, 0.545, "NNLS hit gate", fontsize=TINY, color=MUTE, ha="left")
axs = fig.add_axes([0.026, 0.330, 0.096, 0.165])
axs.plot(wn, hitspec, color=INK, lw=0.5)
topv = float(np.nanmax(hitspec))
for s, b in zip(SUBS, (1570, 1270, 1367)):
    axs.plot([b], [topv * 1.07], marker="v", ms=1.9, color=CO[s], clip_on=False)
axs.set_xlim(500, 2500); axs.set_ylim(0, topv * 1.14)
axs.set_xticks([]); axs.set_yticks([]); axs.patch.set_alpha(0)
for spn in axs.spines.values():
    spn.set_visible(False)
axs.spines["bottom"].set_visible(True); axs.spines["bottom"].set_linewidth(0.5)
ov.text(0.074, 0.295, "hit-pixel spectra", fontsize=SUB_FS, color=INK, ha="center")
ov.text(0.074, 0.263, "log(1+x) · 1,290 ch", fontsize=TINY, color=MUTE, ha="center")
ov.text(0.074, 0.085, r"500–2500 cm$^{-1}$", fontsize=TINY, color=MUTE, ha="center")

# ---------------------------- composition module --------------------------------
ROW_A = 0.725
arrow(0.128, 0.41, 0.163, 0.60)            # input → composition slabs
slab_stack(0.255, ROW_A, [1290, 256, 64, 4], ["1,290", "256", "64", "softmax 4"],
           SLAB_C, hmax=0.27,
           last_classes=[CO["DQ"], CO["TBZ"], CO["THI"], "#9aa3ad"])

arrow(0.345, ROW_A, 0.392, ROW_A)
for i, s in enumerate(SUBS):
    ax = fig.add_axes([0.400 + i * 0.077, 0.655, 0.065, 0.130]); ax.set_axis_off()
    cm = LinearSegmentedColormap.from_list("m", ["#ffffff", CO[s]])
    ax.imshow(grid(ratio_nb[:, i]), cmap=cm, vmin=0, vmax=1,
              interpolation="nearest", aspect="auto")
    ov.text(0.4325 + i * 0.077, 0.622, s, fontsize=TINY, color=CO[s], ha="center",
            weight="bold")
ov.text(0.510, 0.585, "per-pixel component maps", fontsize=SUB_FS, color=INK,
        ha="center")

comp = ratio_nb[hit].mean(0); comp = comp / comp.sum()
arrow(0.638, ROW_A, 0.688, ROW_A)
axb = fig.add_axes([0.695, 0.695, 0.165, 0.058]); axb.set_axis_off()
axb.set_xlim(0, 1); axb.set_ylim(0, 1); axb.patch.set_alpha(0)
x0 = 0.0
for i, s in enumerate(SUBS):
    axb.add_patch(plt.Rectangle((x0, 0.08), comp[i], 0.84, facecolor=CO[s],
                                edgecolor="white", lw=0.4))
    if comp[i] > 0.08:
        axb.text(x0 + comp[i] / 2, 0.5, f"{comp[i]*100:.0f}%", ha="center",
                 va="center", fontsize=5.0, color="white", weight="bold")
    x0 += comp[i]
ov.text(0.7775, 0.780, "map composition", fontsize=SUB_FS, color=INK, ha="center")
ov.text(0.7775, 0.640, "mean pool over hit pixels", fontsize=TINY, color=MUTE,
        ha="center")
ov.text(0.7775, 0.608, "BLK removed · renormalised", fontsize=TINY, color=MUTE,
        ha="center")

# ---------------------------- quantification module -----------------------------
arrow(0.128, 0.38, 0.163, 0.24)            # input → calibration inversion
axc = fig.add_axes([0.170, 0.115, 0.112, 0.230])
for s in SUBS:
    c, sig = CAL[s]
    lv = np.unique(c)
    ms = np.array([sig[c == v].mean() for v in lv])
    a_, b_ = np.polyfit(np.log10(c), sig, 1)
    xx = np.linspace(np.log10(lv.min()), np.log10(lv.max()), 20)
    axc.plot(xx, a_ * xx + b_, color=CO[s], lw=0.7, alpha=0.9, zorder=2)
    axc.plot(np.log10(lv), ms, "o", ms=1.8, color=CO[s], zorder=3,
             markeredgecolor="white", markeredgewidth=0.3)
axc.patch.set_alpha(0)
axc.set_xticks([1, 2]); axc.set_xticklabels(["10", "100"], fontsize=TINY)
axc.set_yticks([])
for spn in ("top", "right"):
    axc.spines[spn].set_visible(False)
for spn in ("left", "bottom"):
    axc.spines[spn].set_linewidth(0.5)
axc.tick_params(length=1.5, width=0.5, pad=1)
axc.set_ylabel(r"I$_{band}$", fontsize=TINY, labelpad=0.5)
ov.text(0.226, 0.362, "measured calibration", fontsize=SUB_FS, color=INK,
        ha="center")
ov.text(0.226, 0.068, r"I = a·log$_{10}$C + b → C$_{cal}$ (µM)", fontsize=TINY,
        color=MUTE, ha="center")

arrow(0.292, 0.225, 0.328, 0.225)
slab_stack(0.415, 0.225, [12, 128, 32, 3], ["12", "128", "32", "3"],
           SLAB_G, hmax=0.20,
           last_classes=[CO["DQ"], CO["TBZ"], CO["THI"]])
ov.text(0.415, 0.362, "residual net", fontsize=SUB_FS, color=INK, ha="center")
ov.text(0.415, 0.068, r"map features → $\Delta$log$_{10}$C", fontsize=TINY,
        color=MUTE, ha="center")
arrow(0.505, 0.225, 0.545, 0.225)
ov.text(0.578, 0.240, r"C = C$_{cal}$·10$^{\Delta}$", fontsize=NAME_FS, color=INK,
        ha="center", weight="bold")
ov.text(0.578, 0.195, r"$\Delta$ clipped ±2 dec", fontsize=TINY, color=MUTE,
        ha="center")

# I_eq connector: composition probabilities weight the band read-out
arrow(0.300, 0.525, 0.300, 0.462)
ov.text(0.310, 0.487, r"probability-weighted band signal I$_{eq}$ "
        "(1570 / 1270 / 1367 cm$^{-1}$)", fontsize=TINY, color=MUTE, ha="left")

# ---------------------------- output panel --------------------------------------
arrow(0.612, 0.225, 0.678, 0.225)
ov.add_patch(FancyBboxPatch((0.688, 0.235), 0.282, 0.130,
                            boxstyle="round,pad=0.004,rounding_size=0.010",
                            facecolor="white", edgecolor=INK, lw=0.7, zorder=2))
ov.text(0.829, 0.322, "concentration (µM)", fontsize=NAME_FS, weight="bold",
        color=INK, ha="center", zorder=3)
ov.text(0.829, 0.276, "per component · validated window · OOD flags",
        fontsize=TINY, color=MUTE, ha="center", zorder=3)
ov.add_patch(FancyBboxPatch((0.688, 0.075), 0.282, 0.125,
                            boxstyle="round,pad=0.004,rounding_size=0.010",
                            facecolor="white", edgecolor=MUTE, lw=0.7,
                            linestyle=(0, (3, 2)), zorder=2))
ov.text(0.829, 0.158, r"known total (optional): C$_i$ = p$_i$ × C$_{total}$",
        fontsize=SUB_FS, color=INK, ha="center", zorder=3)
ov.text(0.829, 0.112, "declared metadata — constrained", fontsize=TINY,
        color=MUTE, ha="center", zorder=3)
# composition feeds the known-total gate — routed outside the panels so the
# dashed line never crosses the concentration box
for a, b in [((0.862, 0.722), (0.994, 0.722)), ((0.994, 0.722), (0.994, 0.138))]:
    ov.plot([a[0], b[0]], [a[1], b[1]], color=CONN, lw=0.7, ls=(0, (3, 2)),
            zorder=3)
arrow(0.994, 0.138, 0.974, 0.138, ls=(0, (3, 2)), lw=0.7)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_architecture_dl.{ext}"), dpi=600,
                bbox_inches="tight", pad_inches=0.02, facecolor="white")
print("saved fig_architecture_dl.png/.pdf")
