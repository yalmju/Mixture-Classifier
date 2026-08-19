"""1) 총농도(2137 밴드) x 조성(분석물 비) = 개별 성분 농도, LOO
   2) 템플릿 범위를 silent region까지 넓히면 blank 누출(=LOD)이 줄어드나"""
import os as _os, sys as _sys                 # paths 부트스트랩 — 기계마다 마운트가 다르다
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import sys, math, os
import numpy as np
from scipy.optimize import nnls

REPO = paths.REPO
sys.path.insert(0, REPO)
sys.path.insert(0, "/private/tmp/claude-501/-Users-seungki2-Documents-GitHub-Mixture-Classifier--claude-worktrees-ink-check-6maps-run-ad8cec/61a536fa-e12d-4dee-8d68-95377dbe80ce/scratchpad")
import unmix, silent as S
from real_data import load_map
from sers_mixture import als_baseline

R, COMP, SUB = S.R, S.COMP, S.SUB
maps = sorted(COMP)
C = np.array([COMP[n] for n in maps], float)

# ---------- 1) 결합 정량 ----------
print("=" * 78)
print("1. 총농도(잉크 2137밴드) x 조성(분석물 비) = 개별 성분 농도   [LOO]")
print("=" * 78)
band = np.array([np.mean([math.log10(R[(n, r)]["band_ratio"])
                          for r in (1, 2, 3) if (n, r) in R]) for n in maps])
Sg = np.array([np.median([R[(n, r)]["S"] for r in (1, 2, 3) if (n, r) in R], axis=0)
               for n in maps])
x_tot = np.log10(C.sum(axis=1) / 3)
folds, comp_err = [], []
for k in range(6):
    tr = [q for q in range(6) if q != k]
    sl, ic = np.polyfit(x_tot[tr], band[tr], 1)          # 잉크 검량
    rf = (np.log10(Sg[tr]) - np.log10(C[tr])).mean(axis=0)
    rf = rf - rf.mean()                                   # 응답계수
    tot_hat = 10 ** ((band[k] - ic) / sl) * 3             # 총농도
    frac = Sg[k] / 10 ** rf
    frac = frac / frac.sum()                              # 조성
    est = frac * tot_hat                                  # 개별 농도
    f = np.maximum(est / C[k], C[k] / est)
    folds.append(f)
    comp_err.append(np.abs(frac - C[k] / C[k].sum()).sum() / 2 * 100)
    print(f"  #{maps[k]} 참값 " + "/".join(f"{v:5.1f}" for v in C[k]) +
          "   예측 " + "/".join(f"{v:5.1f}" for v in est) +
          "   오차 " + "/".join(f"{v:4.2f}x" for v in f) +
          f"   총 {tot_hat:5.1f}")
folds = np.array(folds)
print(f"\n  성분별 중앙 오차: " +
      "   ".join(f"{SUB[j]} {np.median(folds[:, j]):.2f}배" for j in range(3)))
print(f"  전체 중앙 {np.median(folds):.2f}배 · 2배이내 {100*(folds<2).mean():.0f}% · 최악 {folds.max():.2f}배")
print(f"  [비교] 분석물 단독(게인 가정): DQ 2.04 · TBZ 1.50 · THI 1.19배")

# ---------- 2) 범위를 넓히면 누출이 주나 ----------
print("\n" + "=" * 78)
print("2. 템플릿 범위 확장 -> blank 누출(LOD 바닥)이 줄어드나")
print("=" * 78)
DB = paths.PURE
names, WN, means = unmix._templates(DB, True, None)
WN = np.asarray(WN, float)
SI = [names.index(s) for s in SUB]
blanks = [f"{DB}/INK-{i}.csv" for i in (1, 2, 3)] + [f"{DB}/BLK-{i}.csv" for i in (1, 2, 3, 4)]


def run(lo, hi):
    m = (WN >= lo) & (WN <= hi)
    P = unmix._l2(unmix._baseline_removed(means[:, m], True))
    out = []
    for p in blanks:
        wn, cube, _mm, _c = load_map(p)
        wn = np.asarray(wn, float)
        A = []
        for v in np.asarray(cube, float):
            y = np.clip(v - als_baseline(v), 0.0, None)
            A.append(nnls(P.T, np.interp(WN[m], wn, y))[0])
        out.append(np.median(np.asarray(A)[:, SI], axis=0))
    B = np.array(out)
    G = P @ P.T
    ik = names.index("INK")
    cos = [G[ik, j] for j in SI]
    return B, cos


for lo, hi, lab in ((300, 1800, "300-1800 (지금까지)"), (300, 2400, "300-2400 (silent 포함)")):
    B, cos = run(lo, hi)
    print(f"\n  --- {lab} ---")
    print(f"     cos(INK, ·)  " + "  ".join(f"{SUB[j]} {cos[j]:.3f}" for j in range(3)))
    print(f"     blank 허위계수 SD  " +
          "  ".join(f"{SUB[j]} {B[:, j].std(ddof=1):7.1f}" for j in range(3)) +
          "   (평균 " + " ".join(f"{B[:, j].mean():6.1f}" for j in range(3)) + ")")
