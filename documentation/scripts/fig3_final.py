"""Figure 3 최종 초안 — 검량선 → 마스터 곡선 → 등몰 → DFT → 알고리즘.

  a  농도별 검량선: 단독 희석계열 + 직선 적합 (고농도 처짐 = working range 한계)
  b  혼합 실측(비율 축): 마스터 곡선 — 색 = 성분, 빈 ○ 이원 / 채운 ◆ 삼원
  c  혼합 실측(농도 축): 등몰 1:1:1 — 포화 절벽
  d/e  DFT 그림 삽입 (흡착 배향 · 에너지 다이어그램). 시뮬-실측 간극은 SI.
  f  알고리즘 = 최종 배포 DUET-Net (260831_Model_FINAL): NNLS 게이트 → 조성망 →
     농도망(검량 + 잔차) → 조성 · 겉보기 µM. 그림은 fig3f_network.draw (신경망 그림체).
     물리 사전학습은 쓰지 않는다 (2026-08-19 결정).

  초안이다 — 최종 외형은 Origin/일러스트. 수치는 전부 확정 문서값과 같은 경로.

  실행:  ../../.venv/bin/python3 -u fig3_final.py  →  fig3_final.png
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, paths.REPO)

from dl_model import _refs
from io_utils import load_calibration_csv
from calibration import calibrate, fit_sips
from competitive import coverages

SUB = ["DQ", "TBZ", "THI"]
COLS = {"DQ": "#1f78c8", "TBZ": "#78c240", "THI": "#e3425f"}
idx = {s: i for i, s in enumerate(SUB)}

# ── 물리: 채택 구성과 같은 검량·지수 ────────────────────────────────────────
subs, wn, mask, P, lo, hi = _refs(paths.PURE, True, None)
ax_c, nc, dils = load_calibration_csv(paths.calibration())
mc = (ax_c >= lo) & (ax_c <= hi)
dil = [(dils[nc.index(s)][0], np.asarray(dils[nc.index(s)][1])[:, mc]) for s in SUB]
Pm = np.asarray([P[subs.index(s)] for s in SUB], float)
cal = calibrate(dil, Pm, SUB)
iso_m = np.ones(3); iso_m[0] = fit_sips(cal.C_series[0], cal.B_series[0])[2]


def sim_frac(conc_M):
    th = coverages(np.asarray(conc_M, float), cal.K, iso_m)
    b = cal.gA * th
    return b / (b.sum() or 1.0)


# ── 실측: 마스터 곡선 npz + 확정 상수 ───────────────────────────────────────
d = np.load(f"{HERE}/binary_response.npz", allow_pickle=True)
A_FIT = {"DQ": -0.28, "TBZ": 0.0, "THI": +0.92}; W_FIT = 0.72   # 최종-단독 합동 적합
TERN = [((3.3, 333.3, 333.3), (0.00, 0.00, 1.00)), ((33.3, 333.3, 333.3), (0.00, 0.00, 1.00)),
        ((333.3, 3.3, 333.3), (0.00, 0.00, 1.00)), ((333.3, 33.3, 333.3), (0.00, 0.00, 1.00)),
        ((333.3, 333.3, 3.3), (0.65, 0.29, 0.05)), ((333.3, 333.3, 33.3), (0.51, 0.15, 0.34)),
        ((333.3, 333.3, 333.3), (0.00, 0.01, 0.99))]
EQ_UM = [10, 33.3, 100, 333.3]
EQ = {"DQ": [33, 21, 0, 0], "TBZ": [21, 39, 0, 0], "THI": [46, 41, 100, 100]}
MEAS_X0 = {"THI/DQ": -1.05, "THI/TBZ": -0.71}
gamma = 1 / (W_FIT * np.log(10))


def x_eff(conc, i):
    z = np.array([(c * 10 ** A_FIT[s]) ** gamma if c > 0 else 0.0
                  for c, s in zip(conc, SUB)])
    return W_FIT * np.log(z[i] / (z.sum() - z[i]))


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 7.5, "axes.linewidth": 0.6,
                     "xtick.direction": "out", "ytick.direction": "out"})

fig = plt.figure(figsize=(10.8, 6.2))
gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.34)


def _ax(cell, letter):
    ax = fig.add_subplot(cell)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(-0.18, 1.08, letter, transform=ax.transAxes, fontsize=11,
            weight="bold", va="top")
    return ax


# ── a: 농도별 검량선 (직선 적합 — 고농도 점이 직선 아래로 처진다) ───────────
ax = _ax(gs[0, 0], "a")
LIN_MAX = 30e-6                                     # 선형 적합 구간 (≈ 작업구간)
for i, s in enumerate(SUB):
    C = np.asarray(cal.C_series[i]); B = np.asarray(cal.B_series[i])
    sc = 1.0 / B.max()
    sel = (C > 0) & (C <= 120e-6)
    ax.plot(C[sel] * 1e6, B[sel] * sc, "o", color=COLS[s], ms=3.4, mew=0, label=s)
    k = (C > 0) & (C <= LIN_MAX)
    slope = float(np.sum(B[k] * C[k]) / np.sum(C[k] ** 2))     # 원점 통과 직선
    cg = np.linspace(0, 120e-6, 2)
    ax.plot(cg * 1e6, slope * cg * sc, "-", color=COLS[s], lw=0.9)
ax.axvspan(0, LIN_MAX * 1e6, color="0.94", zorder=0)
ax.text(15, 1.02, "linear range", fontsize=6, color="0.4", ha="center")
ax.set_xlim(0, 120); ax.set_ylim(0, 1.08)
ax.set_xlabel("Concentration (µM)")
ax.set_ylabel("Signal (norm.)")
ax.set_title("Calibration is linear only in a\nnarrow range — then falls off", fontsize=8)
ax.legend(frameon=False, fontsize=6.5, loc="lower right")

# ── b: 혼합 실측, 비율 축 (마스터 곡선) ──────────────────────────────────────
ax = _ax(gs[0, 1], "b")
gx = np.linspace(-3.2, 3.6, 300)
ax.plot(gx, 100 / (1 + np.exp(-gx / W_FIT)), "-", color="0.2", lw=1.0)
for name, x, f, t in zip(d["pair"], d["x"], d["f"], d["total"]):
    A, B = name.split("/")
    x0 = A_FIT[B] - A_FIT[A]
    keep = np.isfinite(np.asarray(t, float))
    ax.plot(np.asarray(x, float)[keep] - x0, 100 * np.asarray(f, float)[keep],
            "o", color=COLS[A], ms=3.2, mew=0.8, mfc="none", lw=0)
for conc, meas in TERN:
    for i, s in enumerate(SUB):
        ax.plot(x_eff(conc, i), 100 * meas[i], "D", color=COLS[s], ms=3.6, mew=0)
ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax.set_xticklabels([r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", "1", "10",
                    r"$10^{2}$", r"$10^{3}$"])
ax.set_xlabel(r"Normalized molar ratio, $r/r_{50}$")
ax.set_ylabel("Signal fraction (%)")
ax.set_title("Mixtures: competition bends the scale\n"
             r"($r_{50}$: 16× DQ or 8× TBZ offsets THI)", fontsize=8)

# ── c: 혼합 실측, 농도 축 (등몰) ────────────────────────────────────────────
ax = _ax(gs[0, 2], "c")
ax.axhline(100 / 3, color="0.8", lw=0.7, ls=(0, (3, 2)))
ax.text(11, 36, "solution: 33% each", fontsize=6, color="0.45")
for s in SUB:
    ax.plot(EQ_UM, EQ[s], "o-", color=COLS[s], ms=3.2, lw=0.9)
ax.set_xscale("log")
ax.set_xlabel("Equimolar conc. per component (µM)")
ax.set_ylabel("Signal fraction (%)")
ax.set_title("Same 1:1:1 solution —\nsaturation hands the signal to THI", fontsize=8)

# ── d/e: 사용자 DFT 시뮬레이션 그림 삽입 (시뮬-실측 간극 패널은 SI 로) ───────
SIMDIR = "/Users/seungki2/pyproj/sers_simulation_code/FINAL/01_main_figures"
for cell, png, letter, title in (
        (gs[1, 0], "Fig3a_orientation.png", "d", "DFT: adsorption geometry"),
        (gs[1, 1], "Fig3b_energy_diagram.png", "e", "DFT: site competition vs occupancy")):
    ax = _ax(cell, letter)
    ax.imshow(plt.imread(f"{SIMDIR}/{png}"))
    ax.set_axis_off()
    ax.set_title(title, fontsize=8)

# ── f: 알고리즘 (fig3f_network.draw — 논문 브랜치는 조성 단계 PLS 판) ────────
import fig3f_network
ax = _ax(gs[1, 2], "f")
fig3f_network.draw(ax, "pls")
ax.set_title("Algorithm (DUET-Net): composition,\nthen concentration", fontsize=8)

out = f"{HERE}/fig3_final.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print("저장:", out)
