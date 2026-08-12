"""응답인자를 단일성분 검량선에서 계산한다 — MLP도 혼합물도 거치지 않는다.

  체크리스트 3번. 현재 validate.py `_response_factors`는 혼합물의 (관측/참) 비에서
  응답인자를 뽑는다. 관측이 NNLS라 'MLP 순환'은 아니지만, **보정할 대상인 혼합물
  자신으로 보정계수를 적합**하므로 in-sample이다. 여기서는 그 고리를 끊는다:

      단일성분 희석계열 → 대표 피크 세기 vs 농도 선형회귀 → 기울기 → 기울기 비

  두 경로를 나란히 낸다.
    (a) peak  : 대표 밴드 피크 높이(국소 선형 baseline 차감) vs C 선형회귀
    (b) NNLS-B: 템플릿 NNLS 계수 B vs C — Langmuir 적합 후 초기기울기 gA·K

  Langmuir B(C) = gA·K·C/(1+K·C) 이므로 저농도 초기기울기는 gA·K이고,
  선형구간은 C ≪ 1/K다. 혼합물 농도(최종 3–24 µM)가 그 안에 있는지 대조한다.

  실행:  python3 -u rf_calib.py [anchor=TBZ]
"""
import sys, os
import numpy as np

REPO = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/github/Mixture Classifier"
DB = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB"
sys.path.insert(0, REPO)

from io_utils import load_calibration_csv
from calibration import calibrate, fit_isotherm
from dl_model import _refs

PURE = f"{DB}/Pure"
CALIB = f"{DB}/Ratio/results/calibration_spectra.csv"
ANCHOR = sys.argv[1] if len(sys.argv) > 1 else None
MIX_LO, MIX_HI = 3e-6, 24e-6          # 혼합물 최종 농도 구간 (µM 격자)


def local_height(y, ax, w, half=12.0, wing=25.0):
    """w 주변 피크 높이 — 양옆 날개의 중앙값을 잇는 직선을 국소 baseline으로 빼고 잰다."""
    core = np.abs(ax - w) <= half
    lw = (ax >= w - wing - half) & (ax < w - half)
    rw = (ax > w + half) & (ax <= w + wing + half)
    if not core.any():
        return np.nan
    if not (lw.any() and rw.any()):
        return float(y[core].max())
    xl, yl = float(ax[lw].mean()), float(np.median(y[lw]))
    xr, yr = float(ax[rw].mean()), float(np.median(y[rw]))
    base = yl + (y[core] * 0 + 1) * 0                      # shape holder
    base = yl + (ax[core] - xl) * ((yr - yl) / max(xr - xl, 1e-9))
    return float((y[core] - base).max())


def pick_band(P, ax, i, min_excl=0.80):
    """i번 성분의 대표 밴드. 겹치는 밴드를 쓰면 기울기에 남의 신호가 섞이므로
    배타성(그 파수에서 이 성분이 차지하는 비율) ≥ min_excl 인 곳에서만 고른다.
    조건을 만족하는 곳이 없으면 문턱을 낮춰가며 재시도한다."""
    tot = P.sum(0) + 1e-12
    excl = P[i] / tot                                       # 1.0 = 완전 단독
    for thr in (min_excl, 0.7, 0.6, 0.0):
        ok = excl >= thr
        if ok.any():
            j = int(np.flatnonzero(ok)[np.argmax(P[i][ok])])
            return float(ax[j]), float(excl[j]), j
    j = int(np.argmax(P[i]))
    return float(ax[j]), float(excl[j]), j


def linfit(C, S):
    """S = a·C + b 최소제곱 → (기울기, 절편, R²)."""
    C = np.asarray(C, float); S = np.asarray(S, float)
    a, b = np.polyfit(C, S, 1)
    pred = a * C + b
    ss = float(((S - S.mean()) ** 2).sum())
    r2 = 1.0 - float(((S - pred) ** 2).sum()) / ss if ss > 0 else float("nan")
    return float(a), float(b), r2


subs, wn, mask, P, lo, hi = _refs(PURE, baseline=True, trim=None)
ax_c, names, dils = load_calibration_csv(CALIB)
mc = (ax_c >= lo) & (ax_c <= hi)
axw = ax_c[mc]
print(f"창 {lo:.0f}–{hi:.0f} cm⁻¹ · 성분 {subs} · 검량 {names}")

miss = [s for s in subs if s not in names]
if miss:
    sys.exit(f"검량계열에 없는 성분: {miss}")

series = {s: (np.asarray(dils[names.index(s)][0], float),
              np.asarray(dils[names.index(s)][1], float)[:, mc]) for s in subs}

# ---------------------------------------------------------------- (b) NNLS-B + Langmuir
cal = calibrate([series[s] for s in subs], P, subs)
K = dict(zip(subs, cal.K)); gA = dict(zip(subs, cal.gA))

print("\n(b) NNLS-B + Langmuir  —  B(C)=gA·K·C/(1+K·C)")
print(f"{'성분':>6} | {'gA':>10} | {'K (1/M)':>11} | {'초기기울기 gA·K':>15} | {'선형한계 1/K':>12} | 혼합물 위치")
init = {}
for s in subs:
    init[s] = gA[s] * K[s]
    lin_uM = 1e6 / K[s]
    # 혼합물 구간에서의 포화도 K·C — 0.1 미만이면 선형으로 봐도 무방
    sat_lo, sat_hi = K[s] * MIX_LO, K[s] * MIX_HI
    tag = ("선형" if sat_hi < 0.1 else "준선형" if sat_hi < 0.3 else "포화 진입")
    print(f"{s:>6} | {gA[s]:10.3g} | {K[s]:11.3g} | {init[s]:15.4g} | {lin_uM:9.1f} µM | "
          f"K·C {sat_lo:.3f}–{sat_hi:.3f}  {tag}")

# ---------------------------------------------------------------- (a) 대표 피크 선형회귀
# 피크 높이는 h(C) = B(C)·P[i,w] 이다. 즉 피크 기울기 = (dB/dC)·P[i,w] 로,
# 템플릿의 그 밴드 진폭만큼 (b)와 축이 어긋나 있다. P[i,w]로 나눠 같은 축(dB/dC)에 놓아야
# 두 경로가 비교 가능하다. 이걸 빼먹으면 밴드가 뾰족한 성분이 무조건 과대평가된다.
print("\n(a) 대표 피크 세기 vs 농도  —  선형구간에서만 적합")
print(f"{'성분':>6} | {'밴드':>9} | {'배타성':>6} | {'적합점':>6} | {'피크기울기':>11} | "
      f"{'P[i,w]':>8} | {'→ dB/dC':>10} | {'R²':>6}")
slope = {}
for i, s in enumerate(subs):
    w, excl, j = pick_band(P, axw, i)
    C, Y = series[s]
    Hs = np.array([local_height(y, axw, w) for y in Y])
    use = (C >= MIX_LO / 10) & (C <= min(MIX_HI * 10, 1.0 / max(K[s], 1e-12)))
    if use.sum() < 3:                                   # 너무 좁으면 저농도 절반으로
        use = C <= np.median(C)
    a, b, r2 = linfit(C[use], Hs[use])
    amp = float(P[i][j])
    slope[s] = a / amp if amp > 0 else float("nan")     # 템플릿 진폭 보정 → dB/dC
    print(f"{s:>6} | {w:7.1f}  | {excl:6.2f} | {int(use.sum()):6d} | {a:11.4g} | "
          f"{amp:8.4f} | {slope[s]:10.4g} | {r2:6.3f}")

# ---------------------------------------------------------------- 응답인자
# --------------------------------- (c) 같은 저농도 창에서 NNLS-B 직접 선형회귀
# (a)와 (b)는 적합 범위가 다르다. (a)는 성분마다 1/K이 달라 창이 달랐고 (b)는 C→0 외삽이다.
# 범위 교란을 없애려면 세 성분 공통의 저농도 창에서 같은 방식으로 재보아야 한다.
CW = min(1.0 / max(K[s], 1e-12) for s in subs) / 3.0      # 가장 빨리 휘는 성분의 1/3K
print(f"\n(c) 공통 저농도 창 C ≤ {CW*1e6:.1f} µM 에서 NNLS-B 직접 선형회귀 (범위 교란 제거)")
print(f"{'성분':>6} | {'적합점':>6} | {'dB/dC':>12} | {'R²':>6} | {'최대 K·C':>8}")
bslope = {}
for i, s in enumerate(subs):
    C, Y = series[s]
    from calibration import fit_B
    B = np.array([fit_B(y, P)[0][i] for y in Y])
    use = C <= CW
    if use.sum() < 3:
        use = np.argsort(C) < 3
    a, b, r2 = linfit(C[use], B[use])
    bslope[s] = a
    print(f"{s:>6} | {int(use.sum()):6d} | {a:12.4g} | {r2:6.3f} | {K[s]*C[use].max():8.3f}")

anchor = ANCHOR or min(subs, key=lambda s: init[s])      # 기본: 가장 둔감한 성분 = 1
print(f"\n응답인자 (anchor {anchor} = 1)")
print(f"{'성분':>6} | {'(a) 피크':>9} | {'(b) gA·K 외삽':>14} | {'(c) 공통창 B':>13}")
for s in subs:
    ra = slope[s] / slope[anchor] if slope[anchor] else float("nan")
    rb = init[s] / init[anchor] if init[anchor] else float("nan")
    rc = bslope[s] / bslope[anchor] if bslope[anchor] else float("nan")
    print(f"{s:>6} | {ra:9.2f} | {rb:14.2f} | {rc:13.2f}")

print("\n  (a)와 (c)가 맞으면 피크 경로가 건전한 것 — 남은 (b)와의 차이는 Langmuir 외삽 몫이다.")
print("  세 값이 다 다르면 '응답인자 하나'로 굳힐 수 없다는 뜻이고, 그때는 혼합물 농도가")
print("  선형구간을 벗어났는지(위 K·C)부터 봐야 한다.")
print("  고른 값을 validate.py `_response_factors` 대신 **고정** 입력으로 넣어 회수율을")
print("  다시 계산하면 in-sample 고리가 끊어진다.")
