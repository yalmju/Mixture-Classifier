# -*- coding: utf-8 -*-
"""Architecture drawn 1:1 from the code (dl_model.py / dl_quantify.py / page_real.py).

Every number below was read from the deployed bundle
(260831_Model_FINAL/mlp_composition_260831_final.dlm + sidecars) and the code:
  composition net  : _spec_net  — 1,290 → 256 [FC+BN+ReLU+Dropout 0.15] → 64 [FC+ReLU] → 4 softmax
  map composition  : ratio_nb = A[:, non-BLK] / sum  (BLK removed, renormalised)
  concentration net: _residual_net_torch — 12 → 128 [FC+BN+ReLU+Dropout 0.25] → 32 [FC+ReLU] → 3
                     output = Δlog10 (clipped ±2 decades); C = Ccal · 10^Δ
  12 map features  : log10 Ccal ×3 · composition ×3 · log1p band signal ×3 · log-total p10/p50/p90
  Ccal             : log-linear calibration inversion per substance band, clipped 9–144 µM
  pixel library    : 7-feature pixel signature (log1p band ×3, log1p total, composition ×3),
                     z-scored; k = 15 nearest of 6,828 library pixels; log-space distance-weighted mean
  route (auto)     : map nearest-library distance ≤ 3 → net; else pixel library if median
                     pixel distance ≤ 3; else no answer
실행:  python fig_architecture_exact.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labfig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

labfig.setup()
CO = labfig.CO
INK, MUTE = "#30343a", "#6a7178"
SUBS = ["THI", "TBZ", "DQ"]

fig = plt.figure(figsize=(13.2, 5.2))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()


def box(x, y, w, h, lines, fs=8.2, fill="white", edge=INK, lw=1.0, ls="-", title=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.002,rounding_size=0.008",
                                facecolor=fill, edgecolor=edge, linewidth=lw, linestyle=ls, zorder=2))
    if title:
        ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="center", fontsize=fs + 1.2,
                weight="bold", color=INK, zorder=3)
    y0 = y + h - (0.075 if title else 0.03)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y0 - i * 0.042, t, ha="center", va="center", fontsize=fs,
                color=INK, zorder=3)


def arrow(x0, y0, x1, y1, txt=None, col=INK):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 color=col, lw=1.2, zorder=4))
    if txt:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.022, txt, ha="center", fontsize=7.4, color=MUTE)


# ---------------- left: compositional unmixing ----------------
ax.text(0.20, 0.95, "Compositional unmixing", ha="center", fontsize=12.5, weight="bold", color=INK)
box(0.02, 0.62, 0.12, 0.22, ["one spectrum", "per pixel", "500–2500 cm$^{-1}$", "1,290 points"],
    title="Pixel spectrum")
arrow(0.14, 0.73, 0.175, 0.73, "NNLS gate\n(background px dropped)")
box(0.18, 0.50, 0.21, 0.34,
    ["1,290 → 256   FC · BN · ReLU · Dropout 0.15",
     "256 → 64       FC · ReLU",
     "64 → 4          FC · softmax",
     "outputs: THI · TBZ · DQ · BLK",
     "trained on prepared mixtures (L1 loss)"],
    title="Composition MLP  f(x)", fs=7.6)
arrow(0.39, 0.67, 0.425, 0.67)
box(0.43, 0.56, 0.15, 0.22,
    ["BLK removed,", "3 analytes renormalised", "→ per-pixel composition pᵢ",
     "(sums to 100 %)"], title="Map composition")

# ---------------- right: concentration ----------------
ax.text(0.79, 0.95, "Concentration (µM)", ha="center", fontsize=12.5, weight="bold", color=INK)
# route A: residual net
box(0.62, 0.60, 0.35, 0.30,
    ["inputs (12, map-level): log10 Ccal ×3 · composition ×3",
     "log1p band signal ×3 · log-total p10 / p50 / p90",
     "12 → 128   FC · BN · ReLU · Dropout 0.25",
     "128 → 32   FC · ReLU        32 → 3   FC (Δlog10, |Δ| ≤ 2)",
     "C = Ccal · 10^Δ ;  Ccal = log-linear band calibration (9–144 µM)"],
    title="Route A — concentration net  g(map)", fs=7.4)
# route B: pixel library
box(0.62, 0.20, 0.35, 0.28,
    ["pixel signature (7): log1p band ×3 · log1p total · composition ×3",
     "z-scored → k = 15 nearest of 6,828 library pixels",
     "(93 prepared maps ≤ 100 µM + 175 calibration spectra)",
     "µM = distance-weighted mean in log space (per pixel)"],
    title="Route B — pixel library k-NN", fs=7.4)
# router
box(0.43, 0.16, 0.15, 0.30,
    ["map nearest-library", "distance ≤ 3 → Route A", "else pixel median", "distance ≤ 3 → Route B",
     "else: no answer"], title="auto", fs=7.4)
arrow(0.58, 0.67, 0.62, 0.75, col=INK)      # composition → route A
arrow(0.505, 0.56, 0.505, 0.46, col=INK)    # composition → router
arrow(0.58, 0.34, 0.62, 0.34, col=INK)      # router → route B
arrow(0.58, 0.40, 0.62, 0.70, col=MUTE)     # router → route A (thin)
# outputs
for i, s in enumerate(SUBS):
    ax.scatter(0.985, 0.78 - i * 0.05, s=70, color=CO[s], zorder=5, clip_on=False)
    ax.scatter(0.985, 0.38 - i * 0.05, s=70, color=CO[s], zorder=5, clip_on=False)
ax.text(0.985, 0.88, "µM", ha="center", fontsize=8, color=INK)
ax.text(0.985, 0.48, "µM", ha="center", fontsize=8, color=INK)
# declared-total overlay
box(0.43, 0.02, 0.54, 0.12,
    ["Ĉᵢ = pᵢ · C_total — used only when the sample's total concentration is declared"],
    title="Known total (optional)", fs=7.4, ls="--", edge=MUTE)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "fig_architecture_exact")
fig.savefig(out + ".png", dpi=400, bbox_inches="tight", facecolor="white")
fig.savefig(out + ".pdf", bbox_inches="tight", facecolor="white")
print("saved", out + ".png")
