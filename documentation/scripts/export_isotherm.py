"""패널 b — 단일성분 등온선. 측정점 · 적합곡선 · 파라미터를 CSV 로.

  `isotherm_figs.py` 와 **같은 적합**을 쓴다 (Sips 는 K 를 Langmuir 값에 고정하고 m 만
  자유). 셋 다 자유로 두면 DQ 가 K=2e-8, gA=7e8 로 퇴화해서 자기 곡선은 재현하지만
  다른 성분과 축이 안 맞는다.

  검량 경로는 `paths.calibration()` 이 찾는다 — 하드코딩된 `Ratio/results/…` 는 사라졌고,
  그걸 그대로 두면 학습 쪽에서 사전학습이 조용히 꺼진다.

  나오는 것 (`260808_data interpret/panels/`)
    panel_b_isotherm_points.csv   측정점 — compound, concentration_uM, B
    panel_b_isotherm_curves.csv   적합곡선 — compound, model, range, concentration_uM, B
                                  range=full 은 로그 간격(전체 희석계열, x 로그축용),
                                  range=work 은 0–30 µM 선형 간격(작업구간 확대용).
                                  두 벌을 내는 이유: full 을 선형축에 그대로 쓰면 저농도가
                                  뭉치고, work 을 로그축에 쓰면 0 이 안 들어간다.
    panel_b_isotherm_params.csv   K · 1/K · gA · m · 1/n · R² 세 개

  실행:  python3 -u export_isotherm.py
"""
import os, sys, csv
import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

import paths
from io_utils import load_calibration_csv
from calibration import fit_B, fit_isotherm, fit_sips
from dl_model import _refs

DB = paths.DB
OUT = f"{DB}/260808_data interpret/panels"
SUB = ["DQ", "TBZ", "THI"]
WORK = (3.0, 24.0)          # 혼합물 작업 구간 (µM/성분) — 그림의 음영
os.makedirs(OUT, exist_ok=True)

CAL = paths.calibration()
print(f"검량: {os.path.relpath(CAL, DB)}")
subs, wn, mask, P, lo, hi = _refs(f"{DB}/Pure", True, None)
ax_c, names, dils = load_calibration_csv(CAL)
mc = (ax_c >= lo) & (ax_c <= hi)


def freundlich(C, a, inv_n):
    return a * C ** inv_n


def r2_of(B, pred):
    ss = ((B - B.mean()) ** 2).sum()
    return 1 - ((B - pred) ** 2).sum() / ss if ss > 0 else np.nan


points, curves, params = [], [], []
print(f"\n{'성분':>5} | {'K (1/M)':>10} | {'1/K (µM)':>9} | {'gA':>10} | {'m':>6} | "
      f"{'1/n':>6} | {'R² L / S / F':>20}")
print("-" * 84)
for i, s in enumerate(SUB):
    j = names.index(s)
    C = np.asarray(dils[j][0], float)                 # M
    Y = np.asarray(dils[j][1], float)[:, mc]
    B = np.array([fit_B(y, P)[0][i] for y in Y])
    o = np.argsort(C); C, B = C[o], B[o]

    gA, K = fit_isotherm(C, B)
    gA_s, K_s, m = fit_sips(C, B)                     # K 고정, m 만 자유
    try:
        pf, _ = curve_fit(freundlich, C, B, p0=[B.max(), 0.5],
                          bounds=([0, 0.05], [np.inf, 3.0]), maxfev=40000)
    except Exception:
        pf = [np.nan, np.nan]

    uL = K * C
    uS = np.power(K_s * C, m)
    r2L = r2_of(B, gA * uL / (1 + uL))
    r2S = r2_of(B, gA_s * uS / (1 + uS))
    r2F = r2_of(B, freundlich(C, *pf))

    for c, b in zip(C, B):
        points.append([s, f"{c * 1e6:.6g}", f"{b:.6g}",
                       str(int(WORK[0] <= c * 1e6 <= WORK[1]))])

    grids = [("full", np.logspace(np.log10(C.min() * 0.7), np.log10(C.max() * 1.4), 400)),
             ("work", np.linspace(0.0, 30e-6, 400))]
    for tag, xs in grids:
        vL = K * xs
        vS = np.power(K_s * xs, m)
        for model, yy in (("Langmuir", gA * vL / (1 + vL)),
                          ("Sips", gA_s * vS / (1 + vS)),
                          ("Freundlich", freundlich(xs, *pf))):
            for x, y in zip(xs, yy):
                if np.isfinite(y):
                    curves.append([s, model, tag, f"{x * 1e6:.6g}", f"{y:.6g}"])

    params.append([s, f"{K:.6g}", f"{1e6 / K:.4f}", f"{gA:.6g}", f"{gA_s:.6g}",
                   f"{m:.4f}", f"{pf[0]:.6g}", f"{pf[1]:.4f}",
                   f"{r2L:.4f}", f"{r2S:.4f}", f"{r2F:.4f}", str(len(C))])
    print(f"{s:>5} | {K:10.3g} | {1e6 / K:9.1f} | {gA:10.3g} | {m:6.3f} | {pf[1]:6.3f} | "
          f"{r2L:6.3f} / {r2S:5.3f} / {r2F:5.3f}")


def write(name, head, body):
    with open(f"{OUT}/{name}", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(head); w.writerows(body)
    print(f"  ✓ {name:<38} {len(body)}행 × {len(head)}열")


print()
write("panel_b_isotherm_points.csv",
      ["compound", "concentration_uM", "nnls_coefficient_B", "in_working_range"], points)
write("panel_b_isotherm_curves.csv",
      ["compound", "model", "range", "concentration_uM", "nnls_coefficient_B"], curves)
write("panel_b_isotherm_params.csv",
      ["compound", "K_per_M", "inv_K_uM", "gA_langmuir", "gA_sips", "m_sips",
       "a_freundlich", "inv_n_freundlich", "r2_langmuir", "r2_sips", "r2_freundlich",
       "n_points"], params)
print(f"\n작업구간 음영: {WORK[0]}–{WORK[1]} µM (points 의 in_working_range 로도 표시)")
print("Sips 는 K 를 Langmuir 값에 고정하고 m 만 적합했다 — 캡션에 적을 것.")
print("K 순서로 '흡착이 가장 좋다'고 쓰지 말 것: TBZ·THI 신뢰구간이 겹친다.")
