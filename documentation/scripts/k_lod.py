"""K, LOD, and the concentration verdict from everything measured so far."""
import sys, math
import numpy as np
from scipy.optimize import nnls, least_squares

sys.path.insert(0, "/private/tmp/claude-501/-Users-seungki2-Documents-GitHub-Mixture-Classifier--claude-worktrees-ink-check-6maps-run-ad8cec/61a536fa-e12d-4dee-8d68-95377dbe80ce/scratchpad")
import ink6
from real_data import load_map
from sers_mixture import als_baseline

names, wn_t, P = ink6.templates()
SUB = ["DQ", "TBZ", "THI"]
SI = [names.index(s) for s in SUB]
IK = names.index("INK")
DB = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB/Pure"
COMP = {1: (6, 6, 24), 2: (12, 12, 12), 3: (24, 6, 6),
        4: (6, 6, 6), 5: (24, 24, 24), 6: (12, 12, 48)}
maps = sorted(COMP)


def pix(path):
    wn, cube, _m, _c = load_map(path)
    wn = np.asarray(wn, float)
    m = (wn >= ink6.LO) & (wn <= ink6.HI)
    wn_m, cube = wn[m], np.asarray(cube, float)[:, m]
    out = []
    for v in cube:
        y = np.clip(v - als_baseline(v), 0.0, None)
        if len(wn_m) != len(wn_t) or not np.allclose(wn_m, wn_t):
            y = np.interp(wn_t, wn_m, y)
        a, _ = nnls(P.T, y)
        out.append(a)
    return np.asarray(out)


A = {n: pix(f"/Volumes/Seungki/260805/tert-new/{n}-1.csv") for n in maps}
S = np.array([[np.median(A[n][:, i]) for i in SI] for n in maps])
INK = np.array([np.median(A[n][:, IK]) for n in maps])
C = np.array([COMP[n] for n in maps], float)
F = S / (S.sum(axis=1, keepdims=True) + INK[:, None])           # gain-free fractions
FINK = INK / (S.sum(axis=1) + INK)

print("=" * 76)
print("1.  K가 혼합물 데이터에서 따로 뽑히는가")
print("=" * 76)
print("""  경쟁 Langmuir:  theta_i = K_i C_i / D,   D = 1 + sum K_j C_j
                  s_i = g * a_i * theta_i
  게인 없는 관측량(분율)에서는 D가 상쇄된다:
                  f_i = R_i C_i / (sum R_j C_j + Omega),   R_i = a_i * K_i
  -> 분율은 R_i(= 밝기 x 친화도)만 정한다. K_i 단독은 들어가지 않는다.""")

# fit R_i and Omega from the fractions
def resid(p):
    R = np.concatenate([[1.0], np.exp(p[:2])])          # R_DQ fixed to 1
    Om = math.exp(p[2])
    out = []
    for k in range(6):
        den = (R * C[k]).sum() + Om
        out.extend(list(R * C[k] / den - F[k]) + [Om / den - FINK[k]])
    return out


sol = least_squares(resid, [math.log(4.2), math.log(7.2), math.log(100.0)])
R = np.concatenate([[1.0], np.exp(sol.x[:2])])
Om = math.exp(sol.x[2])
rms = math.sqrt((np.array(resid(sol.x)) ** 2).mean())
print(f"\n  적합 결과 (분율 4개 x 6맵 = 24 관측, 파라미터 3개):")
print(f"    R (= a*K, 상대)  DQ 1.00 : TBZ {R[1]:.2f} : THI {R[2]:.2f}")
print(f"    Omega (잉크 몫)  {Om:.1f}      잔차 RMS {rms:.4f} (분율 단위)")
print(f"    문서 K (등온선)  DQ 1.00 : TBZ 11.6 : THI 16.9")
print(f"    -> 밝기 a 비율  DQ 1.00 : TBZ {R[1]/11.6:.2f} : THI {R[2]/16.9:.2f}"
      "   (R/K, DQ가 단위피복당 더 밝다)")

# can the ink pin down absolute K?  f0/f - 1 = sum K C   (linear)
print("\n  잉크로 절대 K를 잡을 수 있나  (f0/f_ink - 1 = sum K_j C_j):")
X = np.column_stack([np.ones(6), C])
b, *_ = np.linalg.lstsq(X, 1.0 / FINK, rcond=None)
pred = X @ b
r2 = 1 - ((1 / FINK - pred) ** 2).sum() / ((1 / FINK - (1 / FINK).mean()) ** 2).sum()
print(f"    1/f_ink = {b[0]:+.2f} " + " ".join(f"{b[j+1]:+.4f}*C_{SUB[j]}" for j in range(3)) +
      f"    R2={r2:.3f}")
if b[0] > 0:
    Kabs = b[1:] / b[0]
    print(f"    -> K = " + "  ".join(f"{SUB[j]}:{Kabs[j]:.4f}/uM (반포화 {1/Kabs[j]:6.1f} uM)"
                                     for j in range(3) if Kabs[j] > 0))
    bad = [SUB[j] for j in range(3) if Kabs[j] <= 0]
    if bad:
        print(f"    !! 음수 K: {bad}  -> 물리적으로 불가. 모델이 틀렸다는 뜻.")
else:
    print(f"    !! 절편 {b[0]:.2f} <= 0 -> 물리적으로 불가 (문서 §6b의 '-0.97' 실패와 같은 현상)")

print("\n" + "=" * 76)
print("2.  LOD — 검출한계")
print("=" * 76)
blanks = [f"{DB}/INK-{i}.csv" for i in (1, 2, 3)] + [f"{DB}/BLK-{i}.csv" for i in (1, 2, 3, 4)]
Bs = []
for p in blanks:
    a = pix(p)
    Bs.append(np.median(a[:, SI], axis=0))
Bs = np.array(Bs)
print(f"  blank {len(blanks)}맵(INK 3 + BLK 4)에서 각 성분의 허위 계수:")
for j, s in enumerate(SUB):
    print(f"    {s:>4}: " + "  ".join(f"{v:7.1f}" for v in Bs[:, j]) +
          f"   평균 {Bs[:,j].mean():7.1f}  SD {Bs[:,j].std(ddof=1):7.1f}")

sigma = Bs.std(axis=0, ddof=1)
slope_lo = (S / C).min(axis=0)          # 경쟁 최악
slope_hi = (S / C).max(axis=0)          # 경쟁 최소
print(f"\n  LOD = 3.3 * sigma / (uM당 신호).  경쟁 정도에 따라 기울기가 변하므로 구간으로:")
print(f"{'':>6} {'sigma':>9} {'기울기 최소':>12} {'기울기 최대':>12} {'LOD 최악':>11} {'LOD 최선':>11}")
for j, s in enumerate(SUB):
    print(f"  {s:>4} {sigma[j]:9.1f} {slope_lo[j]:12.0f} {slope_hi[j]:12.0f} "
          f"{3.3*sigma[j]/slope_lo[j]:10.2f}uM {3.3*sigma[j]/slope_hi[j]:10.2f}uM")
print("\n  문서 §7 (blank=INK, 맵 400점): DQ 84 nM · TBZ 94 nM · THI 83 nM")
print("  차이의 이유: 위는 맵 100점 + blank 7맵의 맵간 산포(더 보수적),")
print("               문서 값은 픽셀 노이즈 기준. 실무 한계는 위쪽이 현실적이다.")

print("\n" + "=" * 76)
print("3.  정량 가능 구간")
print("=" * 76)
print(f"{'':>6} {'LOD(최악)':>11} {'매몰 상한':>11} {'가용 dec':>9}")
burial = {"DQ": 111.0, "TBZ": 111.0, "THI": 1000.0}
for j, s in enumerate(SUB):
    lod = 3.3 * sigma[j] / slope_lo[j]
    print(f"  {s:>4} {lod:10.2f}uM {burial[s]:10.0f}uM {math.log10(burial[s]/lod):9.2f}")
