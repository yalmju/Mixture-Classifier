"""Paper figure — the full DL architecture as deployed (composition + quantification).

Matches the md section-11 "Model and quantification" panel 1, updated for the
calibration-residual µM head adopted 2026-08-26. Same labfig style as the rest of
the figure set; colors follow Pure/colors.json.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labfig
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

labfig.setup()
CO = labfig.CO
INK = "#1b2430"
MUTE = "#6b7683"
LINE = "#8b95a1"
DLFILL = "#e8eef7"          # learned blocks
CALFILL = "#fdf6e8"         # user-measured calibration anchor
OUTFILL = "#eef7ee"         # reported outputs

fig, ax = plt.subplots(figsize=(12.6, 4.9))
ax.set_xlim(0, 126); ax.set_ylim(0, 50)
ax.set_axis_off()


def box(x, y, w, h, lines, fill="white", edge=INK, lw=0.9, ls="-", fs=7.6,
        bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                facecolor=fill, edgecolor=edge, linewidth=lw,
                                linestyle=ls, zorder=2))
    n = len(lines)
    for i, t in enumerate(lines):
        wgt = "bold" if (i == 0 and bold_first) else "normal"
        col = INK if (i == 0 and bold_first) else MUTE
        ax.text(x + w / 2, y + h / 2 + (n - 1 - 2 * i) * (fs * 0.115),
                t, ha="center", va="center", fontsize=fs, weight=wgt,
                color=col, zorder=3)


def arrow(x0, y0, x1, y1, label=None, ls="-", lab_dy=1.1, lab_dx=0, fs=6.6):
    ax.annotate("", (x1, y1), (x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9,
                                linestyle=ls, shrinkA=1.5, shrinkB=1.5), zorder=1)
    if label:
        ax.text((x0 + x1) / 2 + lab_dx, (y0 + y1) / 2 + lab_dy, label,
                ha="center", va="bottom", fontsize=fs, color=MUTE, zorder=3)


# ---- stream headers ------------------------------------------------------------
ax.text(9, 48.5, "Input & gating", fontsize=8.4, weight="bold", color=MUTE, ha="center")
ax.text(63, 48.5, "Composition — one shared model", fontsize=8.4, weight="bold",
        color=MUTE, ha="center")
ax.text(63, 0.6, "Quantification — anchored to the measured calibration",
        fontsize=8.4, weight="bold", color=MUTE, ha="center")

# ---- top row: composition path -------------------------------------------------
box(1, 33, 15, 11, ["Raman map", "N pixels ×", "500–2500 cm$^{-1}$"])
box(20, 33, 15, 11, ["NNLS screening", "hit / background", "mask (fixed)"])
box(39, 33, 16, 11, ["log(1+x)", "full spectrum", "per hit pixel"])
box(59, 33, 14, 11, ["shared MLP", "256 → 64"], fill=DLFILL)
box(77, 33, 15, 11, ["softmax 4", "", ""])
# substance chips inside the softmax box
for i, (s, c) in enumerate([("DQ", CO["DQ"]), ("TBZ", CO["TBZ"]),
                            ("THI", CO["THI"]), ("BLK", "#9aa3ad")]):
    cx = 79.4 + i * 2.75
    ax.add_patch(plt.Rectangle((cx, 36.2), 1.2, 1.2, facecolor=c, edgecolor="none",
                               zorder=3))
    ax.text(cx + 0.6, 34.9, s, ha="center", va="center", fontsize=5.4, color=MUTE,
            zorder=3)

box(96, 39.5, 29, 7, ["per-pixel component maps", "spatial DQ / TBZ / THI / BLK"],
    fill=OUTFILL)
box(96, 29.5, 29, 8, ["map composition",
                      "mean pool · BLK removed · renormalised"], fill=OUTFILL)

arrow(16, 38.5, 20, 38.5)
arrow(35, 38.5, 39, 38.5)
arrow(55, 38.5, 59, 38.5)
arrow(73, 38.5, 77, 38.5)
arrow(92, 40.5, 96, 42.5)
arrow(92, 36.5, 96, 33.8)

# ---- bottom row: quantification path -------------------------------------------
box(39, 7, 16, 11, ["marker-band signal", "I$_{eq}$ · prob-weighted", "1570 / 1270 / 1367 cm$^{-1}$"])
box(59, 7, 16, 11, ["calibration inversion", "I$_{eq}$ = a·log$_{10}$C + b",
                    "→ C$_{cal}$ (range-clipped)"], fill=CALFILL)
box(79, 7, 15, 11, ["residual net", "12 map features",
                    "128 → 32 → $\Delta$log$_{10}$C"], fill=DLFILL)
box(98, 7, 27, 11, ["concentration (µM)", "C = C$_{cal}$ · 10$^{\Delta}$",
                    "validated window + OOD flags"], fill=OUTFILL)
ax.text(67, 5.2, "user-measured dilution series — embedded in the .dlm",
        fontsize=6.2, color=MUTE, ha="center", style="italic")

arrow(47, 33, 47, 18, label="raw band intensity", lab_dx=-8.4, lab_dy=-1.2)
arrow(80, 33, 52, 18, label="pixel probabilities", lab_dx=9.5, lab_dy=0.6)
arrow(55, 12.5, 59, 12.5)
arrow(75, 12.5, 79, 12.5)
arrow(94, 12.5, 98, 12.5)

# ---- optional known-total gate --------------------------------------------------
box(96, 20, 29, 6.5, ["known total (declared metadata)",
                      "C$_i$ = p$_i$ × C$_{total}$ — constrained"], ls="--", lw=0.8)
arrow(110.5, 29.5, 110.5, 26.5, ls="--")
arrow(110.5, 20, 110.5, 18, ls="--")

fig.tight_layout()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_architecture_dl.{ext}"), **labfig.SAVE)
print("saved fig_architecture_dl.png/.pdf")
