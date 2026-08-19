"""64조건 완전격자를 THI 고정 블록 4개(각 16조건)로 나눠 본다.

  THI 3 / 6 / 12 / 24 µM 블록마다 TBZ(y) × DQ(x) 의 4×4 — THI가 경쟁을
  주도하므로 블록 순서가 곧 이야기다. 값은 |예측−실제| 조성의 평균으로
  percentage points(pp)다: 'Error(%)'는 상대오차로 오독되고 모델 탓처럼
  읽혀서 버렸다 (조제·측정 재현성 몫이 섞여 있는 양이다).
  라벨은 전부 **최종 농도(µM)**. 원액은 3배.

  실행:  python3 -u grid64.py [결과파일=~/Desktop/retrain64.txt]
"""
import os, sys, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import labfig
labfig.setup()
import matplotlib.pyplot as plt

CO, SUB = labfig.CO, labfig.SUB
COLS = [CO[s] for s in SUB]
DESK = "/Users/seungki2/Desktop"
INK = "#1c2430"
LV = [3, 6, 12, 24]
SRC = sys.argv[1] if len(sys.argv) > 1 else f"{DESK}/retrain64_blanksips.txt"

mlp = {}
for l in open(SRC, encoding="utf-8"):
    if not re.search(r"\|\s+[\d.]+% \|", l):
        continue
    p = [x.strip() for x in l.split("|")]
    k = tuple(int(float(v)) for v in p[0].split("/"))
    mlp[k] = (np.array([float(v) for v in p[1].split("/")]),
              np.array([float(v) for v in p[2].split("/")]))
print(f"{SRC} — {len(mlp)}조건")

# ------------------------------------------------------------------ 오차 히트맵 4블록
fig, axs = plt.subplots(1, 4, figsize=(15.2, 4.2))
E = {}
for bi, thi in enumerate(LV):
    M = np.full((4, 4), np.nan)
    for r, tbz in enumerate(LV):
        for c, dq in enumerate(LV):
            k = (dq, tbz, thi)
            if k in mlp:
                t, p = mlp[k]; M[r, c] = np.abs(p - t).sum() / 2
    E[thi] = M
    ax = axs[bi]
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=30, origin="upper")
    for r in range(4):
        for c in range(4):
            if np.isfinite(M[r, c]):
                ax.text(c, r, f"{M[r, c]:.0f}", ha="center", va="center", fontsize=10,
                        color="white" if M[r, c] > 18 else INK, fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels([str(v) for v in LV], fontsize=9.5)
    ax.set_yticks(range(4)); ax.set_yticklabels([str(v) for v in LV], fontsize=9.5)
    ax.set_xlabel("DQ (μM)", color=CO["DQ"])
    if bi == 0:
        ax.set_ylabel("TBZ (μM)", color=CO["TBZ"])
    ax.set_title(f"THI {thi} μM\nmean {np.nanmean(M):.1f} pp", fontsize=10, color=CO["THI"])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
cb = fig.colorbar(im, ax=axs, fraction=.02, pad=.02)
cb.set_label("composition deviation (pp)"); cb.outline.set_visible(False)
fig.savefig(f"{DESK}/fig9_grid64_error.png", **labfig.SAVE)
print("fig9_grid64_error.png")

# ------------------------------------------------------------------ 성분별 파리티
from scipy.stats import spearmanr
T = np.array([mlp[k][0] for k in sorted(mlp)])
P = np.array([mlp[k][1] for k in sorted(mlp)])
fig2, ax2 = plt.subplots(1, 3, figsize=(11.0, 3.9))
for i, s in enumerate(SUB):
    t, p = T[:, i], P[:, i]
    a, b = np.polyfit(t, p, 1)
    r2 = 1 - ((p - (a * t + b)) ** 2).sum() / ((p - p.mean()) ** 2).sum()
    A = ax2[i]
    labfig.unity(A, 0, 85)
    xx = np.linspace(0, 85, 10)
    A.plot(xx, a * xx + b, lw=1.4, color=CO[s], zorder=2)
    A.scatter(t, p, s=26, c=CO[s], alpha=.8, edgecolor="white", lw=.4, zorder=3)
    A.set_title(s, color=CO[s]); A.set_xlabel("true composition (%)")
    if i == 0:
        A.set_ylabel("predicted (%)")
    A.text(.05, .95, f"slope {a:.2f}\n$R^2$ {r2:.2f}\n$\\rho$ {spearmanr(t,p).correlation:.2f}\n"
                     f"bias {p.mean()-t.mean():+.1f}%p",
           transform=A.transAxes, va="top", ha="left", fontsize=8.5)
fig2.tight_layout()
fig2.savefig(f"{DESK}/fig10_parity64.png", **labfig.SAVE)
print("fig10_parity64.png")

# ------------------------------------------------------------------ 콘솔 요약
for thi in LV:
    M = E[thi]
    print(f"\nDQ {dq} μM   평균 {np.nanmean(M):.1f}%")
    print("         DQ " + "  ".join(f"{v:>5}" for v in LV))
    for r, tbz in enumerate(LV):
        print(f"  TBZ {tbz:>3}  " + "  ".join(
            f"{M[r, c]:5.1f}" if np.isfinite(M[r, c]) else "    -" for c in range(4)))
print("\n성분별")
for i, s in enumerate(SUB):
    t, p = T[:, i], P[:, i]
    a, b = np.polyfit(t, p, 1)
    r2 = 1 - ((p - (a * t + b)) ** 2).sum() / ((p - p.mean()) ** 2).sum()
    print(f"  {s:<4} 기울기 {a:.2f} · R² {r2:.2f} · 편향 {p.mean()-t.mean():+.1f}%p · MAE {np.abs(p-t).mean():.1f}%p")
