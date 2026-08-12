"""등온선 모델 비교 — Langmuir vs Freundlich, 검량 희석계열로.

  왜 의심스러운가 (2026-08-11):
    · Langmuir 적합의 K 는 TBZ 가 가장 큰데(DQ 1 : TBZ 4.8 : THI 4.4), 실측 등몰 삼원
      시리즈는 고농도에서 THI 만 남는다. 예측과 정반대다.
    · 응답인자를 세 경로로 뽑으면 3~8배로 흩어진다 (rf_calib.py).
    · SERS 기판은 핫스팟이 지배하는 **불균일 표면**이다. Langmuir 의 전제(동일한 자리,
      단분자층, 상호작용 없음)가 성립할 이유가 없다. Freundlich θ = K·C^(1/n) 는
      결합에너지 분포를 가정하므로 이런 표면에 더 흔히 맞는다.

  여기서는 단일성분 희석계열에 두 모델을 적합해 잔차로 가른다. 자유도가 둘로 같아
  (Langmuir gA·K, Freundlich a·1/n) R²·AIC 를 그대로 비교할 수 있다.

  실행:  python3 -u isotherm_cmp.py
"""
import sys
import numpy as np
from scipy.optimize import curve_fit

REPO = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/github/Mixture Classifier"
DB = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB"
sys.path.insert(0, REPO)

from io_utils import load_calibration_csv
from calibration import fit_B
from dl_model import _refs

PURE = f"{DB}/Pure"
CALIB = f"{DB}/Ratio/results/calibration_spectra.csv"
SUB = ["DQ", "TBZ", "THI"]

subs, wn, mask, P, lo, hi = _refs(PURE, True, None)
ax_c, names, dils = load_calibration_csv(CALIB)
mc = (ax_c >= lo) & (ax_c <= hi)


def langmuir(C, gA, K):
    return gA * K * C / (1.0 + K * C)


def freundlich(C, a, inv_n):
    return a * C ** inv_n


def fit(f, C, B, p0, bounds):
    try:
        popt, _ = curve_fit(f, C, B, p0=p0, bounds=bounds, maxfev=40000)
    except Exception:
        return None, np.inf, -np.inf
    res = B - f(C, *popt)
    ss = ((B - B.mean()) ** 2).sum()
    r2 = 1 - (res ** 2).sum() / ss if ss > 0 else np.nan
    n = len(C)
    aic = n * np.log(max((res ** 2).sum() / n, 1e-30)) + 2 * len(popt)
    return popt, aic, r2


print(f"{'성분':>5} | {'모델':>10} | {'파라미터':>34} | {'R²':>6} | {'AIC':>7}")
print("-" * 78)
out = {}
for i, s in enumerate(SUB):
    j = names.index(s)
    C = np.asarray(dils[j][0], float)
    Y = np.asarray(dils[j][1], float)[:, mc]
    B = np.array([fit_B(y, P)[0][i] for y in Y])
    o = np.argsort(C); C, B = C[o], B[o]

    pl, aicl, r2l = fit(langmuir, C, B, [B.max() * 1.3, 1.0 / np.median(C)],
                        ([0, 0], [np.inf, np.inf]))
    pf, aicf, r2f = fit(freundlich, C, B, [B.max(), 0.5],
                        ([0, 0.05], [np.inf, 3.0]))
    out[s] = (pl, pf)
    print(f"{s:>5} | {'Langmuir':>10} | gA {pl[0]:10.3g}  K {pl[1]:10.3g} (1/M)     "
          f"| {r2l:6.3f} | {aicl:7.1f}")
    print(f"{'':>5} | {'Freundlich':>10} | a  {pf[0]:10.3g}  1/n {pf[1]:8.3f}          "
          f"| {r2f:6.3f} | {aicf:7.1f}   {'← 우세' if aicf < aicl else ''}")

print("\n혼합물 구간(성분당 3–24 µM)에서 두 모델이 예측하는 상대 피복률")
print("  (등몰이면 표면 점유 비가 이 값의 비다. 실측은 고농도에서 THI 독점)")
print(f"{'C (µM)':>8} | " + " | ".join(f"{s} L / F" for s in SUB))
for c_um in (3, 12, 24, 100, 333):
    c = c_um * 1e-6
    L = np.array([out[s][0][0] * out[s][0][1] * c / (1 + out[s][0][1] * c) for s in SUB])
    F = np.array([out[s][1][0] * c ** out[s][1][1] for s in SUB])
    L = L / L.sum() * 100; F = F / F.sum() * 100
    print(f"{c_um:8.0f} | " + " | ".join(f"{L[k]:4.0f} / {F[k]:4.0f}" for k in range(3)))

print("\n  Langmuir 는 포화가 있어 고농도에서 셋이 비슷해지고, Freundlich 는 1/n 차이가")
print("  농도가 오를수록 벌어진다. 실측(100 µM 이상 THI 100%)과 어느 쪽이 가까운지 본다.")
