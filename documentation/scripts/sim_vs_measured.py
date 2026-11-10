"""사전학습 시뮬레이터 vs 실측 응답곡선 — "물리가 정말 모델에 들어갔나"의 검증.

  학습이 쓰는 순방향 물리(θ = 경쟁 Sips/Langmuir, 신호 = A·θ, sim_iso="sips_dq"
  채택 구성과 같은 검량·지수)를 그대로 돌려, 이원 비율 스위프와 등몰 농도
  스위프의 **시뮬레이션 신호 분율**을 만들고 실측과 겹친다.

  보여줘야 정직한 것 둘 다:
    · 재현하는 것 — 비율 축의 휘어진 시그모이드(교차점 부호·자릿수)
    · 재현 못 하는 것 — 등몰의 포화 절벽(THI 독점) → 그래서 그 구간은 가드

  실행:  ../../.venv/bin/python3 -u sim_vs_measured.py
  출력:  sim_vs_measured.png + 교차점 비교표(stdout)
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
PAIRS = [("THI", "DQ"), ("THI", "TBZ"), ("DQ", "TBZ")]
MEAS_X0 = {"THI/DQ": -1.05, "THI/TBZ": -0.71, "DQ/TBZ": +0.30}   # 최종 조제 쌍별 적합

subs, wn, mask, P, lo, hi = _refs(paths.PURE, True, None)
calib = paths.calibration()
ax_c, nc, dils = load_calibration_csv(calib)
mc = (ax_c >= lo) & (ax_c <= hi)
dil = [(dils[nc.index(s)][0], np.asarray(dils[nc.index(s)][1])[:, mc]) for s in SUB]
Pm = np.asarray([P[subs.index(s)] for s in SUB], float)
cal = calibrate(dil, Pm, SUB)
iso_m = np.ones(3)
iso_m[0] = fit_sips(cal.C_series[0], cal.B_series[0])[2]        # DQ 만 Sips (채택 구성)
print(f"K = {np.array2string(cal.K, precision=3)}  gA = {np.array2string(cal.gA, precision=3)}"
      f"  iso_m = {np.array2string(iso_m, precision=3)}  (sips_dq)")

idx = {s: i for i, s in enumerate(SUB)}


def frac(conc_M):
    th = coverages(np.asarray(conc_M, float), cal.K, iso_m)
    b = cal.gA * th
    return b / (b.sum() or 1.0)


TOTALS = [200e-6, 550e-6, 1000e-6]                              # 실측 고농도 세트의 총농도
gx = np.linspace(-2.5, 2.5, 251)
print("\n== 이원 교차점: 시뮬 vs 실측 (log10 비율) ==")
sim_curves = {}
for A, B in PAIRS:
    sim_curves[(A, B)] = []
    row = []
    for tot in TOTALS:
        f = []
        for x in gx:
            r = 10 ** x
            c = np.zeros(3); c[idx[A]] = tot * r / (1 + r); c[idx[B]] = tot / (1 + r)
            v = frac(c)
            f.append(v[idx[A]] / (v[idx[A]] + v[idx[B]] or 1.0))
        f = np.array(f)
        sim_curves[(A, B)].append(f)
        k = int(np.argmin(np.abs(f - 0.5)))
        row.append(gx[k])
    print(f"  {A}/{B}:  시뮬 x0 = " + " / ".join(f"{v:+.2f}" for v in row)
          + f"  (총 {'/'.join(str(int(t*1e6)) for t in TOTALS)} µM)"
          + f"   실측 {MEAS_X0[f'{A}/{B}']:+.2f}")

print("\n== 등몰 1:1:1: 시뮬 신호 조성 vs 실측 ==")
MEAS_EQ = {10: (33, 21, 46), 33.3: (21, 39, 41), 100: (0, 0, 100), 333.3: (0, 0, 100)}
eq_c = np.logspace(np.log10(5e-6), np.log10(5e-4), 100)
eq_sim = np.array([frac(np.full(3, c)) for c in eq_c])
for u, meas in MEAS_EQ.items():
    v = frac(np.full(3, u * 1e-6)) * 100
    print(f"  {u:6g} µM/성분  시뮬 " + "/".join(f"{x:4.0f}" for x in v)
          + "   실측 " + "/".join(f"{x:3.0f}" for x in meas))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.linewidth": 0.6})
COLS = {"DQ": "#1f78c8", "TBZ": "#78c240", "THI": "#e3425f"}
d = np.load(f"{HERE}/binary_response.npz", allow_pickle=True)

fig, axes = plt.subplots(1, 4, figsize=(11.6, 2.9))
for ax, (A, B) in zip(axes[:3], PAIRS):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for f, tot, ls in zip(sim_curves[(A, B)], TOTALS, (":", "-", "--")):
        ax.plot(gx, 100 * f, ls, color="0.25", lw=0.9,
                label=f"sim {int(tot*1e6)} µM")
    i = list(d["pair"]).index(f"{A}/{B}")
    keep = np.isfinite(np.asarray(d["total"][i], float))
    ax.plot(np.asarray(d["x"][i], float)[keep],
            100 * np.asarray(d["f"][i], float)[keep], "o",
            color=COLS[A], ms=3.8, mew=0.9, mfc="none", lw=0, label="measured")
    ax.axvline(MEAS_X0[f"{A}/{B}"], color=COLS[A], lw=0.7, ls=":")
    ax.set_title(f"{A} vs {B}", fontsize=9)
    ax.set_xlabel(f"log10([{A}]/[{B}])")
    if ax is axes[0]:
        ax.set_ylabel("Signal fraction (%)")
        ax.legend(frameon=False, fontsize=6, loc="upper left")
ax = axes[3]
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for s in SUB:
    ax.plot(eq_c * 1e6, 100 * eq_sim[:, idx[s]], "-", color=COLS[s], lw=1.0, label=s)
    ax.plot(list(MEAS_EQ), [m[idx[s]] for m in MEAS_EQ.values()], "o",
            color=COLS[s], ms=3.8, mew=0.9, mfc="none", lw=0)
ax.set_xscale("log")
ax.set_xlabel("Equimolar conc. per component (µM)")
ax.set_title("Equimolar 1:1:1 (lines sim, circles measured)", fontsize=8.5)
ax.legend(frameon=False, fontsize=6.5)
fig.tight_layout()
fig.savefig(f"{HERE}/sim_vs_measured.png", dpi=180, bbox_inches="tight")
print(f"\n저장: sim_vs_measured.png")
