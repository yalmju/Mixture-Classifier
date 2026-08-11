"""잉크를 2137 cm-1 (silent region) 밴드로 직접 읽는다.

분석물(DQ·TBZ·THI)은 삼중결합이 없어 1800-2800에 밴드가 없다 -> 교차간섭 원리적으로 0.
지금까지의 NNLS는 300-1800만 써서 이 밴드를 통째로 버렸다.
"""
import sys, math, os
import numpy as np
from scipy.optimize import nnls

REPO = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/github/Mixture Classifier"
sys.path.insert(0, REPO)
import unmix
from real_data import load_map
from sers_mixture import als_baseline

DB = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB/Pure"
D = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB/tert-new-baseline"
COMP = {1: (6, 6, 24), 2: (12, 12, 12), 3: (24, 6, 6),
        4: (6, 6, 6), 5: (24, 24, 24), 6: (12, 12, 48)}
SUB = ["DQ", "TBZ", "THI"]

names, WN, means = unmix._templates(DB, True, None)
WN = np.asarray(WN, float)
FP = (WN >= 300) & (WN <= 1800)
P = unmix._l2(unmix._baseline_removed(means[:, FP], True))
SI = [names.index(s) for s in SUB]
IK = names.index("INK")

INK_BAND = 2136.7


def band_h(y, wn, centre=INK_BAND, half=12.0, gap=(20.0, 60.0)):
    """밴드 최대 - 양옆 바닥 (silent region이라 바닥이 깨끗하다)"""
    m = np.abs(wn - centre) <= half
    d = np.abs(wn - centre)
    side = (d > gap[0]) & (d <= gap[1])
    base = np.median(y[side]) if side.sum() else 0.0
    return float(y[m].max() - base)


def read(path, reapply_als=False):
    wn, cube, _m, _c = load_map(path)
    wn = np.asarray(wn, float)
    cube = np.asarray(cube, float)
    out_nnls, out_band, out_fp = [], [], []
    for v in cube:
        y = np.clip(v - als_baseline(v), 0.0, None) if reapply_als else np.clip(v, 0.0, None)
        yf = np.interp(WN[FP], wn, y)
        a, _ = nnls(P.T, yf)
        out_nnls.append(a)
        out_band.append(band_h(y, wn))
        out_fp.append(a[SI].sum())
    return np.asarray(out_nnls), np.asarray(out_band), np.asarray(out_fp)


print("=" * 80)
print("0. 순물질 참조에서 2137 밴드 — 정말 잉크 전용인가")
print("=" * 80)
Xr = unmix._baseline_removed(means, True)
for i, n in enumerate(names):
    print(f"  {n:>4}: 2137 밴드 높이 {band_h(Xr[i], WN):9.1f}    "
          f"fingerprint 최대 {Xr[i][FP].max():9.1f}")
print("  * DQ·TBZ·THI는 삼중결합이 없다 -> 여기 나오는 신호는 기판/잉크 잔여분이다.")

print("\n" + "=" * 80)
print("1. 18맵 — 잉크를 두 방식으로 읽고 비교")
print("=" * 80)
R = {}
for n in sorted(COMP):
    for r in (1, 2, 3):
        p = f"{D}/{n}-{r}_corrected.csv"
        if os.path.exists(p):
            A, B, F = read(p)
            R[(n, r)] = dict(
                nnls_ink=np.median(A[:, IK]),
                nnls_frac=np.median(A[:, IK]) / np.median(A.sum(axis=1)),
                band=np.median(B),
                fp=np.median(F),
                band_ratio=np.median(B) / np.median(F),      # 게인 무관
                S=np.median(A[:, SI], axis=0))
print(f"{'조건':>15} {'rep':>4} {'2137 높이':>10} {'분석물합':>10} "
      f"{'밴드/분석물':>12} {'NNLS 잉크분율':>13}")
for n in sorted(COMP):
    for r in (1, 2, 3):
        if (n, r) not in R:
            continue
        d = R[(n, r)]
        lab = "/".join(str(v) for v in COMP[n]) if r == 1 else ""
        print(f"  #{n} {lab:>10} {r:>4} {d['band']:10.0f} {d['fp']:10.0f} "
              f"{d['band_ratio']:12.4f} {d['nnls_frac']:13.3f}")

print("\n" + "=" * 80)
print("2. 재현성 비교 — 3반복의 산포 (작을수록 좋다)")
print("=" * 80)
for key, lab in (("band_ratio", "2137밴드 / 분석물합"), ("nnls_frac", "NNLS 잉크분율")):
    sds = []
    for n in sorted(COMP):
        v = [math.log10(R[(n, r)][key]) for r in (1, 2, 3) if (n, r) in R]
        if len(v) >= 2:
            sds.append(np.std(v, ddof=1))
    pool = math.sqrt(np.mean(np.array(sds) ** 2))
    print(f"  {lab:>22}:  SD {pool:.3f} dex = {10**pool:.2f}배   "
          + "  ".join(f"#{n}:{10**s:.2f}" for n, s in zip(sorted(COMP), sds)))

print("\n" + "=" * 80)
print("3. 총농도를 읽는가")
print("=" * 80)
print(f"{'조건':>15} {'총 uM':>6} {'밴드/분석물 (3반복 중앙)':>26}")
xs, ys = [], []
for n in sorted(COMP):
    v = [R[(n, r)]["band_ratio"] for r in (1, 2, 3) if (n, r) in R]
    tot = sum(COMP[n])
    xs.append(math.log10(tot / 3)); ys.append(math.log10(np.median(v)))
    print(f"  #{n} {'/'.join(str(x) for x in COMP[n]):>10} {tot:6d} {np.median(v):26.4f}")
sl, ic = np.polyfit(xs, ys, 1)
pred = np.array(xs) * sl + ic
r2 = 1 - ((np.array(ys) - pred) ** 2).sum() / ((np.array(ys) - np.mean(ys)) ** 2).sum()
print(f"\n  log10(밴드/분석물) = {sl:.2f}*log10(총/3) + {ic:.2f}   R2={r2:.3f}")

print("\n4. ② 가설 A/B — #6(THI편중)/#5(기준), 둘 다 총 72 uM")
for key, lab in (("band_ratio", "2137밴드"), ("nnls_frac", "NNLS 잉크분율")):
    v5 = np.median([R[(5, r)][key] for r in (1, 2, 3) if (5, r) in R])
    v6 = np.median([R[(6, r)][key] for r in (1, 2, 3) if (6, r) in R])
    print(f"  {lab:>14}: #6/#5 = {v6/v5:.2f}배     (B 예측 0.54 · A 예측 1.00)")
