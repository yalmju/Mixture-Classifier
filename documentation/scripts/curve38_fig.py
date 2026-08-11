"""학습곡선 그림 — 저농도 조건 수 vs 조성 오차. cnsplots 형식.

  실행:  python3 -u curve38_fig.py       (curve38.py 를 먼저 돌릴 것)
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import labfig
labfig.setup()
import matplotlib.pyplot as plt

DESK = "/Users/seungki2/Desktop"
d = np.load(f"{HERE}/curve38_result.npz")
sizes, means, sds = d["sizes"], d["means"], d["sds"]
vals = [d[f"v{n}"] for n in sizes]
sem = np.array([v.std(ddof=1) / np.sqrt(len(v)) for v in vals])   # 평균의 표준오차

FLOOR = 9.6                     # 반복 산포가 정하는 하한
NNLS = 27.6                     # 학습 없는 기준선

fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.1))

A = ax[0]
A.axhspan(0, FLOOR, color="#dfe4ea", alpha=.55, lw=0, zorder=0)
A.text(0.6, FLOOR - 1.1, f"repeat-scatter floor {FLOOR}%", fontsize=8.5, color="#5b6673", va="top")
A.axhline(NNLS, ls=(0, (4, 3)), lw=.8, color="#9aa3ad", zorder=1)
A.text(0.6, NNLS + 0.7, f"NNLS baseline {NNLS}%", fontsize=8.5, color="#5b6673")
A.errorbar(sizes, means, yerr=sem, fmt="o-", ms=6, lw=1.8, capsize=3,
           color=labfig.CO["DQ"], zorder=3, label="MLP (held-out)")
A.set_xlabel("low-concentration training conditions")
A.set_ylabel("composition error (%)")
A.set_title("Learning curve — does more data still help?")
A.set_xlim(-1.5, 36); A.set_ylim(0, 32)
A.axvline(35, ls=":", lw=.8, color="#c0c6cd")
A.text(35.3, 29.5, "all 35", fontsize=8, color="#5b6673", rotation=90, va="top")
A.legend(loc="upper right")

# 남은 격차가 얼마나 줄었는지 — 하한까지의 거리
B = ax[1]
gap = means - FLOOR
B.plot(sizes, gap, "o-", ms=6, lw=1.8, color=labfig.CO["THI"], zorder=3)
B.axhline(0, ls=(0, (4, 3)), lw=.8, color="#9aa3ad")
for x, y in zip(sizes, gap):
    B.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 7),
               ha="center", fontsize=8)
B.set_xlabel("low-concentration training conditions")
B.set_ylabel("gap to floor (%p)")
B.set_title("Distance left to the measurement floor")
B.set_xlim(-1.5, 36); B.set_ylim(-1, max(gap) * 1.18)

fig.suptitle("How much does each added condition buy?  "
             "(4-fold, leave-one-condition-out; high-concentration maps always in training)",
             fontsize=10.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .93])
fig.savefig(f"{DESK}/fig3_learning_curve.png", **labfig.SAVE)
print("fig3_learning_curve.png")
for n, m, s in zip(sizes, means, sem):
    print(f"  {n:3d}조건  {m:5.1f}% ± {s:.1f}")
