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
OUT = f"{paths.INTERP}/panels"
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


# ------------------------------------------------------------- Origin 이 바로 먹는 넓은 표
# 긴 형식(7200행)은 필터를 걸어야 그릴 수 있다. 희석계열이 세 성분 모두 같은 9점이라
# 성분을 열로 눕히면 Origin 에서 x 한 번, y 세 번으로 끝난다.
def _wide(name, xs, cols, xlabel="concentration_uM"):
    body = [[f"{xs[i]:.6g}"] + [f"{c[1][i]:.6g}" for c in cols] for i in range(len(xs))]
    write(name, [xlabel] + [c[0] for c in cols], body)


pts = {}
for s_, c_, b_, _w in points:
    pts.setdefault(s_, ([], []))
    pts[s_][0].append(float(c_)); pts[s_][1].append(float(b_))
_wide("panel_b_isotherm_points_wide.csv",
      pts[SUB[0]][0], [(s_, pts[s_][1]) for s_ in SUB])

curve = {}
for s_, model, rng, c_, b_ in curves:
    curve.setdefault((model, rng), {}).setdefault(s_, ([], []))
    curve[(model, rng)][s_][0].append(float(c_))
    curve[(model, rng)][s_][1].append(float(b_))
for (model, rng), by_s in curve.items():
    _wide(f"panel_b_isotherm_curve_{model.lower()}_{rng}.csv",
          by_s[SUB[0]][0], [(s_, by_s[s_][1]) for s_ in SUB])

# 패널에 쓰는 곡선은 **세 성분 모두 Sips** 다 (= curve_sips_full.csv).
# 성분마다 R² 최대인 모델을 따로 고르면 (DQ Freundlich · TBZ Sips · THI Langmuir)
# 세 곡선의 함수형이 달라져 파라미터를 나란히 비교할 수 없다. Sips 하나로 묶으면
# 셋 다 R² 0.92–0.99 로 충분히 맞고, **차이가 지수 m 하나에 들어간다** —
# DQ 0.66 · TBZ 0.72 · THI 0.97. m=1 이 이상적 Langmuir 이므로 THI 만 균일한
# 흡착자리를 갖고 DQ 는 결합에너지가 가장 넓게 퍼져 있다는 게 한 숫자로 읽힌다.
_bx = curve[("Sips", "full")][SUB[0]][0]
write("panel_b_isotherm_curve_main.csv",
      ["concentration_uM"] + [f"{s_}_Sips" for s_ in SUB],
      [[f"{_bx[i]:.6g}"] + [f"{curve[('Sips', 'full')][s_][1][i]:.6g}" for s_ in SUB]
       for i in range(len(_bx))])

# 참고용 — 성분마다 R² 최대인 모델. 패널에는 쓰지 않는다.
best = {}
for p in params:
    s_ = p[0]
    r2 = {"Langmuir": float(p[8]), "Sips": float(p[9]), "Freundlich": float(p[10])}
    best[s_] = max(r2, key=r2.get)
print("\n패널 곡선: 세 성분 모두 Sips  ·  R² " +
      " · ".join(f"{p[0]} {float(p[9]):.3f}" for p in params))
print("           지수 m  " + " · ".join(f"{p[0]} {float(p[5]):.3f}" for p in params)
      + "   (m=1 이 이상적 Langmuir)")
print("[참고] R² 최대 모델: " + " · ".join(f"{k} {v}" for k, v in best.items())
      + " — 함수형이 갈려 파라미터 비교가 안 되므로 패널에는 쓰지 않는다")

print(f"\n작업구간 음영: {WORK[0]}–{WORK[1]} µM (points 의 in_working_range 로도 표시)")
print("Sips 는 K 를 Langmuir 값에 고정하고 m 만 적합했다 — 캡션에 적을 것.")
print("K 순서로 '흡착이 가장 좋다'고 쓰지 말 것: TBZ·THI 신뢰구간이 겹친다.")
