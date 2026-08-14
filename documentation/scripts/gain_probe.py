"""게인 추정자 비교 — 맵마다 얼마나 밝은지를 무엇으로 재는 게 가장 좋은가.

  같은 조건을 다시 조제하면 총 신호가 3.8배(5–95 백분위) 흔들린다. 이 게인을 알면
  B/게인 이 자기 농도를 훨씬 잘 따라간다(R² 0.47/0.64/0.67 → 0.61/0.79/0.82). 그
  0.61/0.79/0.82 은 **참 농도를 알아야 계산되는 오라클**이라 실제로는 못 쓴다.

  여기서는 **참 농도 없이 스펙트럼만으로 계산되는** 후보들을 같은 자리에 놓고 잰다.
  점수는 `B_i / ĝ` 대 `C_i` 의 R² (원점 통과 직선), 성분별.

    none        보정 안 함                      바닥
    oracle      총 B / 총 참농도                천장 (실제로는 못 씀)
    ink2137     잉크 2137 cm⁻¹ 밴드 면적        지금 쓰는 것
    ink_px_mean 픽셀별 잉크 밴드의 평균
    ink_px_med  픽셀별 잉크 밴드의 중앙값        밝은 픽셀 몇 개에 안 흔들린다
    B_INK       잉크 템플릿의 NNLS 계수
    B_blank     블랭크 계열 계수의 합
    B_total     모든 템플릿 계수의 합 (분석물 포함)
    area        스펙트럼 전체 면적
    ink×area    두 개의 기하평균
    combo       log 공간 최소제곱 조합, **leave-one-out 교차검증**
                (오라클에 맞추므로 CV 없이는 누수다)

  결과 먼저: **지금 쓰는 ink2137 이 이미 천장의 90% 이고 무엇으로 바꿔도 안 오른다.**
  조합은 교차검증하면 오히려 85% 로 떨어진다 — 과적합이다. 남은 10%는 통계가 아니라
  더 나은 물리적 손잡이가 있어야 메워진다.

  실행:  python3 -u gain_probe.py
"""
import os, sys, re, glob, csv, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

import paths
from unmix import _templates, _baseline_removed, _l2
from calibration import fit_B
from dl_model import _refs, _map_spectra
from real_data import load_map

DB = paths.DB
OUT = f"{paths.INTERP}/panels"
PURE = f"{DB}/Pure"
SUB = ["DQ", "TBZ", "THI"]
INK_BAND = (2100.0, 2175.0)      # 잉크 2137 cm⁻¹ — 분석물 밴드가 없는 침묵 구간
os.makedirs(OUT, exist_ok=True)

subs, wn, mask, P0, lo, hi = _refs(PURE, True, None)
names, wn_t, means = _templates(PURE, True, None)
P = _l2(_baseline_removed(means[:, mask], True))
idx = [names.index(s) for s in SUB]
blank_i = [i for i, n in enumerate(names) if n not in SUB]
wn_m = np.asarray(wn)[mask]
band = (wn_m >= INK_BAND[0]) & (wn_m <= INK_BAND[1])
print(f"템플릿 {names} · 잉크 밴드 {INK_BAND[0]:.0f}–{INK_BAND[1]:.0f} cm⁻¹ "
      f"({int(band.sum())} 채널)")

# 라벨은 `260814_mixture_final` 파일명 = 최종 µM (`mixtures.py`).
import mixtures as MX
C, B, feat = [], [], []
for _m in MX.catalogue(("grid",), replicates=("baseline",)):   # 격자 폴더만 — 65맵
    p = _m.path
    _w, cube, _mm, _c = load_map(p)
    ya = _map_spectra(cube, mask, 0)[0]
    px = np.array(_map_spectra(cube, mask, 100))       # 맵의 픽셀 전부
    b = fit_B(ya, P)[0]
    pxb = px[:, band].sum(1)
    C.append([float(v) for v in _m.conc])
    B.append(b[idx])
    feat.append({"ink2137": float(ya[band].sum()),
                 "ink_px_mean": float(pxb.mean()),
                 "ink_px_med": float(np.median(pxb)),
                 "B_INK": float(b[names.index("INK")]) if "INK" in names else np.nan,
                 "B_blank": float(sum(b[i] for i in blank_i)),
                 "B_total": float(b.sum()),
                 "area": float(ya.sum())})
C = np.array(C); B = np.array(B)
print(f"{len(C)}맵\n")


def r2_through_origin(x, y):
    a = (x @ y) / (x @ x)
    return 1 - ((y - a * x) ** 2).sum() / (((y - y.mean()) ** 2).sum() or 1.0)


cands = {"none": np.ones(len(C)),
         "oracle": B.sum(1) / C.sum(1)}
for k in ("ink2137", "ink_px_mean", "ink_px_med", "B_INK", "B_blank", "B_total", "area"):
    v = np.array([f[k] for f in feat], float)
    # BLK 계수는 항상 0 이다 — NNLS 가 기판 템플릿에 무게를 준 적이 없다. 중앙값이 0 인
    # 후보를 그대로 나누면 inf 가 되므로 여기서 걸러낸다.
    cands[k] = (v / np.median(v)
                if np.isfinite(v).all() and v.std() > 0 and np.median(v) > 0 else None)

# 기하평균과 조합 — 조합은 오라클에 맞추므로 leave-one-out 으로만 채점한다
_ink = np.array([f["ink2137"] for f in feat], float)
_area = np.array([f["area"] for f in feat], float)
_gm = np.sqrt(_ink * _area)
cands["ink x area"] = _gm / np.median(_gm)
_F = np.array([[f[k] for k in ("ink2137", "ink_px_mean", "ink_px_med",
                               "area", "B_total")] for f in feat], float)
_L = np.log(_F); _L = (_L - _L.mean(0)) / _L.std(0); _L = np.c_[np.ones(len(_L)), _L]
_y = np.log(B.sum(1) / C.sum(1))
_pred = np.zeros(len(_y))
for _i in range(len(_y)):
    _m = np.ones(len(_y), bool); _m[_i] = False
    _w = np.linalg.lstsq(_L[_m], _y[_m], rcond=None)[0]
    _pred[_i] = _L[_i] @ _w
cands["combo LOO-CV"] = np.exp(_pred)

rows = []
print(f"{'게인 추정자':>13} | " + " | ".join(f"{s:>7}" for s in SUB) + " |  평균  | 천장대비")
print("-" * 62)
base = None
for k, g in cands.items():
    if g is None:
        continue
    r = []
    for j in range(3):
        y = B[:, j] / g
        r.append(r2_through_origin(C[:, j], y))
    if k == "none":
        base = np.array(r)
    rows.append([k] + [f"{v:.4f}" for v in r] + [f"{np.mean(r):.4f}"])
    print(f"{k:>13} | " + " | ".join(f"{v:7.3f}" for v in r) + f" | {np.mean(r):6.3f} |",
          end="")
    if k == "oracle":
        ceil = np.mean(r)
        print("  ← 천장")
    else:
        print(f" {100 * np.mean(r) / ceil:6.0f}%" if 'ceil' in dir() else "")

with open(f"{OUT}/gain_estimators.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["gain_estimator"] + [f"r2_{s}" for s in SUB] + ["r2_mean"])
    w.writerows(rows)
print(f"\n  ✓ gain_estimators.csv → {OUT}")

# 게인 추정자끼리 얼마나 닮았나 — 오라클과의 상관
o = cands["oracle"]
print("\n오라클 게인과의 상관 (log 공간, 스케일 무관):")
for k in ("ink2137", "ink_px_mean", "ink_px_med", "B_INK", "B_blank", "B_total", "area"):
    if cands.get(k) is None:
        continue
    c = np.corrcoef(np.log(o), np.log(cands[k]))[0, 1]
    print(f"  {k:>9}  r = {c:+.3f}")
