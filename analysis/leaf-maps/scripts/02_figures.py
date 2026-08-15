"""잎 시료 맵 시리즈 — 논문용 그림 3장.

  형식은 `analysis/dq9-sus-reproducibility/scripts/fig_main.py` 와 같다:
  ui_common.py 팔레트(DQ 파랑 · TBZ 초록 · THI 핑크), Helvetica 8.5 pt,
  축 안에 글자 없음(범례·시료 라벨은 축 바깥), 벡터 PDF(Type42) + 400 dpi PNG.

  Fig_leaf_maps       4 판독값(잉크·DQ·TBZ·THI) x 7 시료 = 28 패널 히트맵
  Fig_leaf_spectra    대표 스펙트럼 (맵 픽셀 중앙값 + 사분위 범위)
  Fig_leaf_contrast   대조군(잎+잉크) 을 뺀 차이 스펙트럼 + 픽셀 검출률

  혼합 시리즈 3맵은 성분색(파랑/초록/핑크)과 겹치지 않게 **한 색상의 명도 3단 + 선종류**
  로 구분한다. 어느 그림에서도 색이 유일한 부호가 아니다 — 트레이스마다 축 바깥에
  이름이 붙고, 히트맵의 색은 행(판독값)이 정한다.

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
PAGE, INK, MUTE, FAINT, LINE = "#ffffff", "#141b24", "#4a5460", "#98a1ac", "#e3e8ee"
BLUE, PINK, GREEN, AMBER, CORAL = "#1a73e8", "#c85a8f", "#4a9e2a", "#c98a15", "#d8542a"
SER = {"DQ": BLUE, "TBZ": GREEN, "THI": PINK}
MARK = {"DQ": "o", "TBZ": "s", "THI": "^"}              # 색 외 2차 부호화
MARKERS = {"DQ": [1179, 1521, 1576], "TBZ": [779, 1011, 1549], "THI": [556, 1138, 1365]}

# 그림에 넣는 맵.  01_extract.py 는 7맵 전부를 처리하고 CSV 에도 전부 들어 있다 —
# 여기서 고르는 것은 **그림에 그릴 열**뿐이다.  다시 넓히려면 이 목록만 늘리면 된다.
CONTROL = "cov3"
SAMPLES = [CONTROL, "leavesmix2peel", "leavesmix2peel2"]
GROUPS = [("control", [CONTROL]),
          ("mixture on leaves, peeled", ["leavesmix2peel", "leavesmix2peel2"])]
SHOW = {"cov3": "ink only", "leavesmix2peel": "peel", "leavesmix2peel2": "peel 2"}
LONG = {"cov3": "leaf · ink only", "leavesmix2peel": "leaves · mix, peel",
        "leavesmix2peel2": "leaves · mix, peel 2"}
SHORT = {"cov3": "ink only", "leavesmix2peel": "mix, peel",
         "leavesmix2peel2": "mix, peel 2"}
AMB = ["#8a5e0a", AMBER]                                # peel 시리즈 — 한 색상 명도 2단
SCOL = {"cov3": MUTE, "leavesmix2peel": AMB[1], "leavesmix2peel2": AMB[0]}
SDASH = {s: "-" for s in SAMPLES}
SDASH.update({"leavesmix2peel2": (0, (3.2, 1.6))})

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


def _shade(hexcol, k):
    """같은 색상의 명도 변형.  k<1 어둡게, k>1 흰쪽으로."""
    r, g, b = to_rgb(hexcol)
    if k <= 1:
        return (r * k, g * k, b * k)
    t = k - 1
    return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)


def ramp(hexcol, lo="#ffffff"):
    """흰색 -> 성분색 -> 더 짙은 같은 색.  세 정거장이라 높은 쪽 대비가 살아난다."""
    return LinearSegmentedColormap.from_list(
        "r", [lo, _shade(hexcol, 1.55), hexcol, _shade(hexcol, .55)])


D = np.load(os.path.join(DATA, "pixels.npz"), allow_pickle=True)
R = np.load(os.path.join(DATA, "representative.npz"), allow_pickle=True)
WN = D["wn"]
DET = {}
for line in open(os.path.join(DATA, "detection_vs_control.csv"), encoding="utf-8-sig"):
    p = line.strip().split(",")
    if p[0] == "sample" or len(p) < 11:
        continue
    DET[(p[0], p[1])] = dict(med=float(p[2]), fold=float(p[5]), snr=float(p[7]),
                             pct=float(p[9]))


def grid(s, key):
    rows, cols = D[f"{s}__rows"], D[f"{s}__cols"]
    M = np.full((rows.max() + 1, cols.max() + 1), np.nan)
    M[rows, cols] = D[f"{s}__{key}"]
    return M


# ================================================================ Fig 1 — maps
ROWS = [("ink2137", "Ink reporter\n2137 cm$^{-1}$", ramp("#2b3a4a")),
        ("DQ_over_ink", "DQ marker\n/ ink", ramp(SER["DQ"])),
        ("TBZ_over_ink", "TBZ marker\n/ ink", ramp(SER["TBZ"])),
        ("THI_over_ink", "THI marker\n/ ink", ramp(SER["THI"]))]

fig = plt.figure(figsize=(4.9, 5.35))
gs = fig.add_gridspec(4, len(SAMPLES), hspace=.09, wspace=.09,
                      left=.205, right=.815, top=.755, bottom=.030)

for i, (key, rlab, cmap) in enumerate(ROWS):
    allv = np.concatenate([D[f"{s}__{key}"] for s in SAMPLES])
    vmin, vmax = np.percentile(allv, 4), np.percentile(allv, 96)
    for j, s in enumerate(SAMPLES):
        ax = fig.add_subplot(gs[i, j])
        im = ax.imshow(grid(s, key), cmap=cmap, vmin=vmin, vmax=vmax,
                       origin="lower", interpolation="nearest", aspect="equal")
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(FAINT); sp.set_linewidth(.8)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:                                   # 열 머리글 (축 바깥)
            n = len(np.unique(D[f"{s}__cols"]))
            ax.set_title(SHOW[s], fontsize=8, pad=12, color=SCOL[s])
            ax.text(.5, 1.015, f"{n}×{n} · {float(D[f'{s}__span']):.0f} µm",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=6.2, color=MUTE)
        if j == 0:                                   # 행 라벨 (축 바깥)
            ax.set_ylabel(rlab, fontsize=7.6, labelpad=5, linespacing=1.35)
        if j == len(SAMPLES) - 1:                    # 행마다 컬러바 하나
            bb = ax.get_position()
            cax = fig.add_axes([bb.x1 + .017, bb.y0 + bb.height * .10,
                                .017, bb.height * .80])
            cb = fig.colorbar(im, cax=cax)
            cb.outline.set_visible(False)
            cb.ax.tick_params(length=2, width=.6, labelsize=6.5, colors=MUTE, pad=1.5)
            cb.set_ticks([vmin, vmax])
            fmt = "%.0f" if key == "ink2137" else "%.2f"
            cb.set_ticklabels([fmt % vmin, fmt % vmax])

# 그룹 묶음 (열 머리글 위, 축 바깥)
for gname, members in GROUPS:
    a = fig.axes[SAMPLES.index(members[0])].get_position()
    b = fig.axes[SAMPLES.index(members[-1])].get_position()
    y = .838
    fig.lines.append(Line2D([a.x0, b.x1], [y, y], transform=fig.transFigure,
                            color=FAINT, lw=.7))
    fig.text((a.x0 + b.x1) / 2, y + .011, gname, ha="center", va="bottom",
             fontsize=7.6, color=MUTE)

fig.text(.5, .952, "Raman maps of the peeled mixture series",
         ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=INK)
fig.text(.5, .941, "ALS-baselined · marker bands normalised to the ink\nreporter of the "
                   "same pixel · scale shared within a row",
         ha="center", va="top", fontsize=7.4, color=MUTE, linespacing=1.35)
out = os.path.join(FIGS, "Fig_leaf_maps.pdf")
fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=400)
print(out)

# ======================================================== Fig 2 — 대표 스펙트럼
fig = plt.figure(figsize=(7.2, 4.2))
gs = fig.add_gridspec(1, 2, width_ratios=[2.40, 1.0], wspace=.55,
                      left=.085, right=.880, top=.775, bottom=.135)

FP = (WN >= 400) & (WN <= 1800)
SIL = (WN >= 2050) & (WN <= 2250)
norm = {s: R[f"{s}__med"] / R[f"{s}__ink"] for s in SAMPLES}
OFF = max(np.ptp(norm[s][FP]) for s in SAMPLES) * .70

# ---------- a : fingerprint ----------
ax = fig.add_subplot(gs[0, 0]); clean(ax)
for nm, bands in MARKERS.items():
    for b in bands:
        ax.axvline(b, color=SER[nm], lw=.9, alpha=.40, zorder=0)
for k, s in enumerate(SAMPLES[::-1]):
    y0 = OFF * k
    ax.fill_between(WN[FP], y0 + R[f"{s}__q25"][FP] / R[f"{s}__ink"],
                    y0 + R[f"{s}__q75"][FP] / R[f"{s}__ink"],
                    color=SCOL[s], alpha=.24, linewidth=0, zorder=2)
    ax.plot(WN[FP], y0 + norm[s][FP], color=SCOL[s], lw=1.15, ls=SDASH[s], zorder=3)
    ax.text(1812, y0 + norm[s][FP][-1], LONG[s], fontsize=7.6, color=SCOL[s],
            va="center", ha="left", clip_on=False)          # 축 바깥 라벨
ax.set_xlim(400, 1800)
ax.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=8.5)
ax.set_ylabel("Intensity / ink band area   (offset)", fontsize=8.5, labelpad=4)
ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xticks([400, 700, 1000, 1300, 1600, 1800])
ax.legend([Line2D([0], [0], color=SER[n], lw=1.6, alpha=.6) for n in SER],
          [f"{n}  " + ", ".join(str(b) for b in MARKERS[n]) for n in SER],
          loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3, fontsize=7.4,
          handlelength=1.0, handletextpad=.4, columnspacing=1.4,
          borderpad=0, borderaxespad=0)

# ---------- b : silent region (raw counts) ----------
ax = fig.add_subplot(gs[0, 1]); clean(ax)
ax.axvline(2137, color=FAINT, lw=.7, ls=(0, (3, 3)), zorder=0)
sp2 = max(np.ptp(R[f"{s}__med"][SIL]) for s in SAMPLES) * .70
for k, s in enumerate(SAMPLES[::-1]):
    y0 = sp2 * k
    ax.fill_between(WN[SIL], y0 + R[f"{s}__q25"][SIL], y0 + R[f"{s}__q75"][SIL],
                    color=SCOL[s], alpha=.24, linewidth=0, zorder=2)
    ax.plot(WN[SIL], y0 + R[f"{s}__med"][SIL], color=SCOL[s], lw=1.15, ls=SDASH[s], zorder=3)
ax.set_xlim(2050, 2250); ax.set_xticks([2050, 2137, 2250])
ax.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=8.5)
ax.set_ylabel("Intensity (counts, offset)", fontsize=8.5, labelpad=2)
ax.yaxis.set_label_position("right")                 # a 패널의 시료 라벨과 안 겹치게
ax.set_yticks([]); ax.spines["left"].set_visible(False)

for _ax, _L in zip(fig.axes, "ab"):
    bb = _ax.get_position()
    fig.text(max(bb.x0 - .054, .004), bb.y1 + .035, _L, fontsize=11, fontweight="bold",
             va="bottom", ha="left", color=INK)
fig.text(.5, .930, "Representative spectra of the peeled mixture series",
         ha="center", va="bottom", fontsize=11, fontweight="bold", color=INK)
fig.text(.5, .893, "Median of the pixels within one preparation; shading = interquartile "
                   "range across those pixels",
         ha="center", va="bottom", fontsize=7.8, color=MUTE)
out = os.path.join(FIGS, "Fig_leaf_spectra.pdf")
fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=400)
print(out)

# ============================================ Fig 3 — 대조군 대비 (차이 + 검출률)
OTHER = [s for s in SAMPLES if s != CONTROL]
fig = plt.figure(figsize=(7.2, 3.8))
gs = fig.add_gridspec(1, 2, width_ratios=[2.30, 1.0], wspace=.50,
                      left=.078, right=.800, top=.755, bottom=.150)

# ---------- a : 차이 스펙트럼 ----------
ax = fig.add_subplot(gs[0, 0]); clean(ax)
for nm, bands in MARKERS.items():
    for b in bands:
        ax.axvline(b, color=SER[nm], lw=.9, alpha=.40, zorder=0)
dif = {s: R[f"{s}__diff"] for s in OTHER}
OFF3 = max(np.ptp(dif[s][FP]) for s in OTHER) * .72
for k, s in enumerate(OTHER[::-1]):
    y0 = OFF3 * k
    ax.axhline(y0, color=LINE, lw=.6, zorder=1)          # 각 트레이스의 0 선
    ax.plot(WN[FP], y0 + dif[s][FP], color=SCOL[s], lw=1.15, ls=SDASH[s], zorder=3)
    ax.text(1812, y0, SHORT[s], fontsize=7.6, color=SCOL[s], va="center", ha="left",
            clip_on=False)
ax.set_xlim(400, 1800)
ax.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=8.5)
ax.set_ylabel("Sample − control   (offset)", fontsize=8.5, labelpad=4)
ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xticks([400, 700, 1000, 1300, 1600, 1800])
ax.legend([Line2D([0], [0], color=SER[n], lw=1.6, alpha=.6) for n in SER],
          [f"{n} marker bands" if n == "DQ" else n for n in SER],
          loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3, fontsize=7.4,
          handlelength=1.0, handletextpad=.4, columnspacing=1.2,
          borderpad=0, borderaxespad=0)

# ---------- b : 픽셀 검출률 ----------
ax = fig.add_subplot(gs[0, 1]); clean(ax)
y = np.arange(len(SAMPLES))[::-1]
ax.axvline(0, color=LINE, lw=.7, zorder=0)
for j, nm in enumerate(SER):
    off = (j - 1) * .24
    v = [DET[(s, nm)]["pct"] for s in SAMPLES]
    ax.scatter(v, y + off, s=30, marker=MARK[nm], facecolor=SER[nm], edgecolor=PAGE,
               linewidth=.7, zorder=3, label=nm)
    for yy, vv in zip(y + off, v):
        ax.plot([0, vv], [yy, yy], color=SER[nm], lw=1.1, alpha=.55, zorder=2)
ax.set_yticks(y)
ax.set_yticklabels([LONG[s] for s in SAMPLES], fontsize=7.4)
ax.yaxis.tick_right()                                # a 패널의 시료 라벨과 안 겹치게
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0, pad=3)
ax.set_ylim(-.6, len(SAMPLES) - .4)
ax.set_xlim(-3, 103); ax.set_xticks([0, 50, 100])
ax.set_xlabel("Pixels above control mean + 3 SD (%)", fontsize=8.5)
ax.legend(loc="lower right", bbox_to_anchor=(1, 1.005), ncol=3, fontsize=7.4,
          handletextpad=.25, columnspacing=1.0, borderpad=0, borderaxespad=0)

for _ax, _L in zip(fig.axes, "ab"):
    bb = _ax.get_position()
    fig.text(max(bb.x0 - .058, .004), bb.y1 + .042, _L, fontsize=11, fontweight="bold",
             va="bottom", ha="left", color=INK)
fig.text(.5, .922, "Analyte signal above the ink-only leaf control",
         ha="center", va="bottom", fontsize=11, fontweight="bold", color=INK)
fig.text(.5, .882, "Control = leaf carrying ink alone; both panels use its own pixels as "
                   "the blank",
         ha="center", va="bottom", fontsize=7.8, color=MUTE)
out = os.path.join(FIGS, "Fig_leaf_contrast.pdf")
fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=400)
print(out)
