"""잎 시료 4맵 — 논문용 그림 2장.

  형식은 `analysis/dq9-sus-reproducibility/scripts/fig_main.py` 와 같다:
  ui_common.py 팔레트(DQ 파랑 · TBZ 초록 · THI 핑크), Helvetica 8.5 pt,
  축 안에 글자 없음(범례는 축 바깥), 벡터 PDF(Type42) + 400 dpi PNG.

  Fig_leaf_maps       4 판독값(잉크·DQ·TBZ·THI) x 4 시료 = 16 패널 히트맵
  Fig_leaf_spectra    대표 스펙트럼(맵 100픽셀 중앙값 + 사분위 범위)

  실행:  python3 -u 02_figures.py      (먼저 01_extract.py)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figures")

# ============ ui_common.py 팔레트 ============
PAGE, INK, MUTE, FAINT, LINE = "#fafbfc", "#1c2430", "#5b6673", "#98a1ac", "#e3e8ee"
BLUE, PINK, GREEN, CORAL = "#1a73e8", "#c85a8f", "#4a9e2a", "#d8542a"
SER = {"DQ": BLUE, "TBZ": GREEN, "THI": PINK}
MARKERS = {"DQ": [1179, 1521, 1576], "TBZ": [779, 1011, 1549], "THI": [556, 1138, 1365]}
SAMPLES = ["leaf_dq", "leaf_tb1", "leaf_thi2", "cov3"]
SHOW = {"leaf_dq": "leaf · DQ", "leaf_tb1": "leaf · TBZ",
        "leaf_thi2": "leaf · THI", "cov3": "cov3"}
SCOL = {"leaf_dq": BLUE, "leaf_tb1": GREEN, "leaf_thi2": PINK, "cov3": MUTE}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.linewidth": .7, "axes.edgecolor": MUTE,
    "xtick.major.width": .7, "ytick.major.width": .7,
    "xtick.color": MUTE, "ytick.color": MUTE, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.labelcolor": INK, "text.color": INK,
    "figure.facecolor": PAGE, "axes.facecolor": PAGE, "savefig.facecolor": PAGE,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=3, pad=2)


def ramp(hexcol, lo="#f4f6f8"):
    """옅은 회백색 -> 성분색.  0 이 배경과 이어지고 강도만 색으로 읽힌다."""
    return LinearSegmentedColormap.from_list("r", [lo, hexcol])


D = np.load(os.path.join(DATA, "pixels.npz"), allow_pickle=True)
R = np.load(os.path.join(DATA, "representative.npz"), allow_pickle=True)
WN = D["wn"]


def grid(s, key):
    rows, cols = D[f"{s}__rows"], D[f"{s}__cols"]
    M = np.full((rows.max() + 1, cols.max() + 1), np.nan)
    M[rows, cols] = D[f"{s}__{key}"]
    return M


def step_um(s):
    ux = np.unique(D[f"{s}__XY"][:, 0])
    return float(np.median(np.diff(ux)))


# ================================================================ Fig 1 — maps
ROWS = [("ink2137", "Ink reporter\n2137 cm$^{-1}$", ramp(INK, "#f2f4f6")),
        ("DQ_over_ink", "DQ marker\n/ ink", ramp(SER["DQ"])),
        ("TBZ_over_ink", "TBZ marker\n/ ink", ramp(SER["TBZ"])),
        ("THI_over_ink", "THI marker\n/ ink", ramp(SER["THI"]))]

fig = plt.figure(figsize=(7.2, 7.35))
gs = fig.add_gridspec(4, 4, hspace=.10, wspace=.10,
                      left=.115, right=.855, top=.845, bottom=.075)
side = step_um(SAMPLES[0]) * 9                       # 10 픽셀 x 50 um = 450 um

for i, (key, rlab, cmap) in enumerate(ROWS):
    allv = np.concatenate([D[f"{s}__{key}"] for s in SAMPLES])
    vmin, vmax = np.percentile(allv, 2), np.percentile(allv, 98)
    for j, s in enumerate(SAMPLES):
        ax = fig.add_subplot(gs[i, j])
        im = ax.imshow(grid(s, key), cmap=cmap, vmin=vmin, vmax=vmax,
                       origin="lower", interpolation="nearest", aspect="equal",
                       extent=[0, side, 0, side])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(LINE); sp.set_linewidth(.7)
        if i == 3 and j == 0:                        # 눈금은 좌하단 한 패널에만
            ax.set_xticks([0, side]); ax.set_yticks([0, side])
            ax.set_xticklabels(["0", f"{side:.0f}"]); ax.set_yticklabels(["0", f"{side:.0f}"])
            ax.tick_params(length=2.5, pad=1.5)
            ax.set_xlabel("µm", fontsize=7.5, labelpad=1)
        else:
            ax.set_xticks([]); ax.set_yticks([])
        if i == 0:                                   # 열 머리글 (축 바깥)
            ax.set_title(SHOW[s], fontsize=8.5, pad=4, color=SCOL[s], fontweight="bold")
        if j == 0:                                   # 행 라벨 (축 바깥)
            ax.set_ylabel(rlab, fontsize=8, labelpad=6, linespacing=1.35)
        if j == 3:                                   # 행마다 컬러바 하나
            bb = ax.get_position()
            cax = fig.add_axes([bb.x1 + .014, bb.y0 + bb.height * .12,
                                .013, bb.height * .76])
            cb = fig.colorbar(im, cax=cax)
            cb.outline.set_visible(False)
            cb.ax.tick_params(length=2, width=.6, labelsize=7, colors=MUTE, pad=1.5)
            cb.set_ticks([vmin, vmax])
            fmt = "%.0f" if key == "ink2137" else "%.2f"
            cb.set_ticklabels([fmt % vmin, fmt % vmax])

fig.text(.5, .945, "Raman maps of the four leaf-surface preparations",
         ha="center", va="bottom", fontsize=11, fontweight="bold", color=INK)
fig.text(.5, .905, "10 × 10 pixels, 50 µm step (450 × 450 µm) · ALS-baselined · "
                   "row-wise common intensity scale",
         ha="center", va="bottom", fontsize=8, color=MUTE)
out = os.path.join(FIGS, "Fig_leaf_maps.pdf")
fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=400)
print(out)

# ======================================================== Fig 2 — 대표 스펙트럼
fig = plt.figure(figsize=(7.2, 4.7))
gs = fig.add_gridspec(1, 2, width_ratios=[2.55, 1.0], wspace=.30,
                      left=.075, right=.895, top=.80, bottom=.115)

FP = (WN >= 400) & (WN <= 1800)
SIL = (WN >= 2050) & (WN <= 2250)
norm = {s: R[f"{s}__med"] / R[f"{s}__ink"] for s in SAMPLES}   # 잉크 밴드로 정규화
span = max(np.ptp(norm[s][FP]) for s in SAMPLES)
OFF = span * .78

# ---------- a : fingerprint ----------
ax = fig.add_subplot(gs[0, 0]); clean(ax)
for nm, bands in MARKERS.items():
    for b in bands:
        ax.axvline(b, color=SER[nm], lw=.7, alpha=.30, zorder=0)
for k, s in enumerate(SAMPLES[::-1]):
    y0 = OFF * k
    ax.fill_between(WN[FP], y0 + R[f"{s}__q25"][FP] / R[f"{s}__ink"],
                    y0 + R[f"{s}__q75"][FP] / R[f"{s}__ink"],
                    color=SCOL[s], alpha=.16, linewidth=0, zorder=2)
    ax.plot(WN[FP], y0 + norm[s][FP], color=SCOL[s], lw=.9, zorder=3)
    ax.text(1812, y0 + norm[s][FP][-1], SHOW[s], fontsize=7.8, color=SCOL[s],
            va="center", ha="left", clip_on=False)          # 축 바깥 라벨
ax.set_xlim(400, 1800)
ax.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=8.5)
ax.set_ylabel("Intensity / ink 2137 cm$^{-1}$ band area   (offset)", fontsize=8.5)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xticks([400, 700, 1000, 1300, 1600, 1800])
ax.legend([Line2D([0], [0], color=SER[n], lw=1.6, alpha=.6) for n in SER],
          [f"{n}  " + ", ".join(str(b) for b in MARKERS[n]) for n in SER],
          loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3, fontsize=7.4,
          handlelength=1.0, handletextpad=.4, columnspacing=1.4,
          borderpad=0, borderaxespad=0)

# ---------- b : silent region (raw counts) ----------
ax = fig.add_subplot(gs[0, 1]); clean(ax)
ax.axvline(2137, color=FAINT, lw=.7, ls=(0, (3, 3)), zorder=0)
sp2 = max(np.ptp(R[f"{s}__med"][SIL]) for s in SAMPLES)
for k, s in enumerate(SAMPLES[::-1]):
    y0 = sp2 * .78 * k
    ax.fill_between(WN[SIL], y0 + R[f"{s}__q25"][SIL], y0 + R[f"{s}__q75"][SIL],
                    color=SCOL[s], alpha=.16, linewidth=0, zorder=2)
    ax.plot(WN[SIL], y0 + R[f"{s}__med"][SIL], color=SCOL[s], lw=.9, zorder=3)
ax.set_xlim(2050, 2250); ax.set_xticks([2050, 2137, 2250])
ax.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=8.5)
ax.set_ylabel("Intensity (counts, offset)", fontsize=8.5, labelpad=2)
ax.yaxis.set_label_position("right")                     # a 패널의 시료 라벨과 안 겹치게
ax.set_yticks([]); ax.spines["left"].set_visible(False)

for _ax, _L in zip(fig.axes, "ab"):
    bb = _ax.get_position()
    fig.text(max(bb.x0 - .058, .004), bb.y1 + .045, _L, fontsize=11, fontweight="bold",
             va="bottom", ha="left", color=INK)
fig.text(.5, .945, "Representative spectra of the four maps",
         ha="center", va="bottom", fontsize=11, fontweight="bold", color=INK)
fig.text(.5, .905, "Median of the 100 spectra within one preparation; shading = "
                   "interquartile range across those pixels",
         ha="center", va="bottom", fontsize=8, color=MUTE)
out = os.path.join(FIGS, "Fig_leaf_spectra.pdf")
fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=400)
print(out)
