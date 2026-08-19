"""tert-new-baseline: 6조건 x 3반복 = 18맵. 재현성 + A/B + 전이성 + 성분별 오차."""
import os as _os, sys as _sys                 # paths 부트스트랩 — 기계마다 마운트가 다르다
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import sys, math, os
import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, "/private/tmp/claude-501/-Users-seungki2-Documents-GitHub-Mixture-Classifier--claude-worktrees-ink-check-6maps-run-ad8cec/61a536fa-e12d-4dee-8d68-95377dbe80ce/scratchpad")
import ink6
from real_data import load_map
from sers_mixture import als_baseline

names, wn_t, P = ink6.templates()
SUB = ["DQ", "TBZ", "THI"]
SI = [names.index(s) for s in SUB]
IK = names.index("INK")
D = f"{paths.DB}/tert-new-baseline"
OLD = "/Volumes/Seungki/260805/tert-new"
COMP = {1: (6, 6, 24), 2: (12, 12, 12), 3: (24, 6, 6),
        4: (6, 6, 6), 5: (24, 24, 24), 6: (12, 12, 48)}


def coeffs(path, reals=True):
    """reals=True -> ALS 한 번 더 건다. False -> 이미 baseline 처리된 것으로 보고 clip만."""
    wn, cube, _m, _c = load_map(path)
    wn = np.asarray(wn, float)
    m = (wn >= ink6.LO) & (wn <= ink6.HI)
    wn_m, cube = wn[m], np.asarray(cube, float)[:, m]
    out = []
    for v in cube:
        y = np.clip(v - als_baseline(v), 0.0, None) if reals else np.clip(v, 0.0, None)
        if len(wn_m) != len(wn_t) or not np.allclose(wn_m, wn_t):
            y = np.interp(wn_t, wn_m, y)
        out.append(nnls(P.T, y)[0])
    return np.asarray(out)


def summ(A):
    med = np.median(A, axis=0)
    return dict(S=med[SI], ink=med[IK], frac=med[IK] / med.sum(), tot=med.sum(), A=A)


print("=" * 78)
print("0. 전처리 정합성 — corrected 파일에 ALS를 또 걸어야 하나")
print("=" * 78)
print(f"{'맵':>6} {'raw+ALS (8/5)':>15} {'corrected, ALS 재적용':>22} {'corrected, clip만':>20}")
for n in (1, 2, 5):
    p_old = f"{OLD}/{n}-1.csv"
    ref = summ(coeffs(p_old, True)) if os.path.exists(p_old) else None
    a1 = summ(coeffs(f"{D}/{n}-1_corrected.csv", True))
    a0 = summ(coeffs(f"{D}/{n}-1_corrected.csv", False))
    r = f"{ref['frac']:.3f}" if ref else "  n/a"
    print(f"  #{n}-1 {r:>13}   ink분율 {a1['frac']:.3f}          {a0['frac']:.3f}")

MODE = None
p_old = f"{OLD}/1-1.csv"
if os.path.exists(p_old):
    ref = summ(coeffs(p_old, True))["frac"]
    d1 = abs(summ(coeffs(f"{D}/1-1_corrected.csv", True))["frac"] - ref)
    d0 = abs(summ(coeffs(f"{D}/1-1_corrected.csv", False))["frac"] - ref)
    MODE = d1 < d0
    print(f"\n  -> raw+ALS 기준값에 더 가까운 쪽: "
          f"{'ALS 재적용' if MODE else 'clip만 (이미 처리됨)'}")
else:
    MODE = False
    print("\n  -> 외장 드라이브 없음. clip만 사용 (파일이 이미 baseline 처리됨).")

print("\n" + "=" * 78)
print(f"1. 18맵 판독  (전처리: {'ALS 재적용' if MODE else 'clip만'})")
print("=" * 78)
R = {}
for n in sorted(COMP):
    for r in (1, 2, 3):
        p = f"{D}/{n}-{r}_corrected.csv"
        if os.path.exists(p):
            R[(n, r)] = summ(coeffs(p, MODE))
print(f"{'조건':>16} {'rep':>4} {'INK':>9} {'ink분율':>8} {'DQ':>8} {'TBZ':>8} {'THI':>8} {'총신호':>9}")
for n in sorted(COMP):
    for r in (1, 2, 3):
        if (n, r) not in R:
            continue
        d = R[(n, r)]
        lab = "/".join(str(v) for v in COMP[n]) if r == 1 else ""
        print(f"  #{n} {lab:>11} {r:>4} {d['ink']:9.0f} {d['frac']:8.3f} "
              + " ".join(f"{v:8.0f}" for v in d["S"]) + f" {d['tot']:9.0f}")

print("\n" + "=" * 78)
print("2. ① 재현성 — 같은 조성 3반복의 게인-무관 비 산포")
print("=" * 78)
print(f"{'조건':>16} {'DQ/THI 3반복':>28} {'SD(dex)':>9} {'배':>7}")
sd_dq, sd_tb, sd_ink = [], [], []
for n in sorted(COMP):
    reps = [R[(n, r)] for r in (1, 2, 3) if (n, r) in R]
    if len(reps) < 2:
        continue
    v = [math.log10(d["S"][0] / d["S"][2]) for d in reps]
    w = [math.log10(d["S"][1] / d["S"][2]) for d in reps]
    f = [math.log10(d["frac"]) for d in reps]
    sd_dq.append(np.std(v, ddof=1)); sd_tb.append(np.std(w, ddof=1)); sd_ink.append(np.std(f, ddof=1))
    print(f"  #{n} {'/'.join(str(x) for x in COMP[n]):>11} "
          + "  ".join(f"{10**x:7.4f}" for x in v)
          + f" {np.std(v, ddof=1):9.3f} {10**np.std(v, ddof=1):7.2f}")
pool_dq = math.sqrt(np.mean(np.array(sd_dq) ** 2))
pool_tb = math.sqrt(np.mean(np.array(sd_tb) ** 2))
pool_ik = math.sqrt(np.mean(np.array(sd_ink) ** 2))
print(f"\n  통합 재현성 SD   DQ/THI {pool_dq:.3f} dex ({10**pool_dq:.2f}배)"
      f"   TBZ/THI {pool_tb:.3f} ({10**pool_tb:.2f}배)"
      f"   잉크분율 {pool_ik:.3f} ({10**pool_ik:.2f}배)")
print(f"  [배제항 적합 잔차 = 0.286 dex (1.93배)]")
print("  판정:", end=" ")
if pool_dq <= 0.12:
    print(f"{10**pool_dq:.2f}배 -> 시료는 재현된다. 잔차 1.93배는 모델 misfit.")
elif pool_dq >= 0.20:
    print(f"{10**pool_dq:.2f}배 -> 잔차 대부분이 시료·기판 산포. 배제항 모델링 무의미.")
else:
    print(f"{10**pool_dq:.2f}배 -> 애매 (0.12~0.20 dex).")

print("\n" + "=" * 78)
print("2. ② 잉크 조성 의존성 — #6(THI편중)/#5(기준), 둘 다 총 72 uM")
print("=" * 78)
f5 = [R[(5, r)]["frac"] for r in (1, 2, 3) if (5, r) in R]
f6 = [R[(6, r)]["frac"] for r in (1, 2, 3) if (6, r) in R]
print(f"  #5 잉크분율 {['%.3f' % v for v in f5]}   중앙 {np.median(f5):.3f}")
print(f"  #6 잉크분율 {['%.3f' % v for v in f6]}   중앙 {np.median(f6):.3f}")
ratio = np.median(f6) / np.median(f5)
print(f"  #6/#5 = {ratio:.2f}배     (가설 B 예측 0.54배 · 가설 A 예측 1.00배)")
print(f"  8/5 단일측정값 = 0.55배")
print("  판정:", "가설 B 확정" if 0.35 <= ratio <= 0.75 else
      ("가설 A" if ratio > 0.8 else "예측보다 강함 — 재검토"))

print("\n" + "=" * 78)
print("3. ③④ 반복1로 검량 -> 반복2·3 예측 (기판·시료 전이성 + 성분별 오차)")
print("=" * 78)
tr = [n for n in sorted(COMP) if (n, 1) in R]
S1 = np.array([R[(n, 1)]["S"] for n in tr])
C1 = np.array([COMP[n] for n in tr], float)
L = np.log10(S1) - np.log10(C1)
g_tr = L.mean()
rr = L.mean(axis=0) - L.mean()
comp_err, folds = [], []
for r in (2, 3):
    for n in tr:
        if (n, r) not in R:
            continue
        S = R[(n, r)]["S"]; C = np.array(COMP[n], float)
        est = S / 10 ** (g_tr + rr)
        e = np.abs(est / est.sum() - C / C.sum()).sum() / 2 * 100
        comp_err.append(e)
        folds.append(np.maximum(est / C, C / est))
folds = np.array(folds)
print(f"  조성 오차 평균 {np.mean(comp_err):.1f}%   [set1 내부 LOO = 19.2%]")
print(f"  성분별 절대농도 오차 (중앙): " +
      "   ".join(f"{SUB[j]} {np.median(folds[:, j]):.2f}배" for j in range(3)))
print(f"  [8/5 등급표: DQ 2.6 · TBZ 1.6 · THI 1.3배]")
print(f"  THI 판정: {'정량 유지' if np.median(folds[:,2]) <= 1.5 else '2배 초과 — 정량 주장 재검토'}")
