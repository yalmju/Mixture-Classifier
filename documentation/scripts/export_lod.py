"""검출한계(LOD) — 3.3σ/기울기, 잉크 블랭크 기준.

  기존 `k_lod.py` 는 지워진 임시폴더의 `ink6` 를 import 해서 돌지 않는다. 여기서 다시
  세운다. 결과는 성분 셋 전부에 대해 나온다 (문서에는 DQ 2.90 µM 하나만 있었다).

    LOD = 3.3 · σ_blank / slope        (IUPAC/ICH, k=3.3 → α=β=5%)
    LOQ = 10   · σ_blank / slope

  **σ 를 어디서 재느냐가 값을 정한다.** 두 가지를 다 낸다.
    per-map    블랭크 맵의 평균 스펙트럼에서 얻은 B 의 맵간 SD.  n=7 (BLK 4 + INK 3).
               맵 하나를 판독 단위로 쓰는 지금 방식과 맞는다. **이쪽을 보고한다.**
    per-pixel  블랭크 픽셀 전부의 B 의 SD.  n 이 크지만 같은 맵 안의 픽셀은 서로
               상관돼 있어 σ 를 과소평가한다. 참고용.

  **기울기도 두 가지.**
    working    30 µM 이하 검량점 5개(0.1·0.5·1·5·10)로 원점을 지나는 직선. 등온선이
               아직 휘지 않은 구간의 감도라 이쪽을 보고한다. (3–24 µM 안에는 검량점이
               5·10 두 개뿐이라 그 둘만으로는 기울기가 잡음을 그대로 받는다.)
    initial    C→0 극한의 접선 gA·K. 등온선이 휘기 전 최대 감도라 LOD 가 낙관적으로
               나온다.

  블랭크는 **BLK(기판)와 INK(인쇄 잉크)를 합쳐서** 쓴다. 잉크는 실제 시료에 항상 있고
  템플릿에서 빠져 있어 그 몫이 분석물 계수로 새어 든다 (INK·DQ 코사인 0.73). 잉크를
  빼고 기판만 쓰면 σ 가 낙관적이 된다.

  실행:  python3 -u export_lod.py
"""
import os, sys, csv, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

import paths
from io_utils import load_calibration_csv
from calibration import fit_B, fit_isotherm
from competitive import fit_B as _fitB
from dl_model import _refs, _map_spectra
from real_data import load_map

DB = paths.DB
OUT = f"{DB}/260808_data interpret/panels"
PURE = f"{DB}/Pure"
SUB = ["DQ", "TBZ", "THI"]
WORK = (3.0, 24.0)
os.makedirs(OUT, exist_ok=True)

subs, wn, mask, P, lo, hi = _refs(PURE, True, None)

# ---- LOD 전용 기저: 블랭크·잉크 템플릿을 **넣는다**.
# 분석물 3장만으로 맞추면 블랭크 맵의 기판·잉크 스펙트럼이 갈 곳이 없어 분석물 계수로
# 통째로 들어간다 (그렇게 재면 블랭크에서 DQ 계수가 평균 7632, σ 9831 이 나오고 LOD 가
# 77 µM 이 된다 — 3 µM 을 실제로 측정하고 있는 것과 모순).
# 그건 잡음이 아니라 **적합 불가로 생긴 계통 오차**다. 블랭크를 기저에 넣으면 분석물
# 계수가 비로소 "0 + 잡음" 이 되고 σ 가 LOD 의 정의에 맞는 양이 된다.
# (조성 판독을 5성분 NNLS 로 바꾸는 것은 별개 문제이고 그건 더 나빴다 — 여기서는
#  LOD 를 재기 위한 기저일 뿐이다.)
from unmix import _templates, _baseline_removed, _l2
_names, _wn, _means = _templates(PURE, True, None)
P_lod = _l2(_baseline_removed(_means[:, mask], True))
idx = [_names.index(s) for s in SUB]
print(f"조성용 템플릿 {subs}")
print(f"LOD용 기저   {_names}  ← 블랭크·잉크 포함 · 창 {lo:.0f}–{hi:.0f} cm⁻¹")
P = P_lod

# ------------------------------------------------------------------ σ: 블랭크의 B 산포
blank_paths = sorted(glob.glob(f"{PURE}/BLK-*.csv")) + sorted(glob.glob(f"{PURE}/INK-*.csv"))
per_map, per_pix = [], []
for p in blank_paths:
    _w, cube, _m, _c = load_map(p)
    ya = _map_spectra(cube, mask, 0)[0]                       # 맵 평균 (ALS 보정 포함)
    per_map.append(fit_B(ya, P)[0][idx])
    for y in _map_spectra(cube, mask, 40):                    # 픽셀 40장
        per_pix.append(fit_B(y, P)[0][idx])
per_map = np.array(per_map); per_pix = np.array(per_pix)
sig_map = per_map.std(axis=0, ddof=1)
sig_pix = per_pix.std(axis=0, ddof=1)
print(f"\n블랭크 {len(blank_paths)}맵 ({', '.join(os.path.basename(b)[:5] for b in blank_paths)})"
      f" · 픽셀 {len(per_pix)}")
for j, s in enumerate(SUB):
    print(f"  {s:<4} B 평균 {per_map[:, j].mean():10.1f} · σ(맵간) {sig_map[j]:9.1f} · "
          f"σ(픽셀) {sig_pix[j]:9.1f}")

# ------------------------------------------------------------------ 기울기: 검량에서
CAL = paths.calibration()
ax_c, names, dils = load_calibration_csv(CAL)
mc = (ax_c >= lo) & (ax_c <= hi)
slope_work, slope_init = {}, {}
for i, s in enumerate(SUB):
    j = names.index(s)
    C = np.asarray(dils[j][0], float)
    Y = np.asarray(dils[j][1], float)[:, mc]
    B = np.array([fit_B(y, P)[0][i] for y in Y])
    o = np.argsort(C); C, B = C[o], B[o]
    # 희석계열은 0.1·0.5·1·5·10·50·100·500·1000 µM 이라 3–24 µM 안에는 두 점(5·10)뿐이다.
    # 두 점에 직선을 맞추면 기울기가 그 두 점의 잡음을 그대로 받는다. 등온선이 아직
    # 휘지 않은 **30 µM 이하 전부**(5점)로 원점을 지나는 직선을 맞춘다 — LOD 정의가
    # 저농도 선형 응답을 전제하므로 절편을 두지 않는다.
    near = C * 1e6 <= 30
    slope_work[s] = float((C[near] * 1e6 @ B[near]) / (C[near] * 1e6 @ (C[near] * 1e6)))
    gA, K = fit_isotherm(C, B)
    slope_init[s] = float(gA * K * 1e-6)                                      # B per µM
    print(f"  {s:<4} 30 µM 이하 검량점 {int(near.sum())}/{len(C)} 개로 기울기 적합 "
          f"({', '.join(f'{v*1e6:g}' for v in C[near])} µM)")

# ------------------------------------------------------------------ LOD / LOQ
rows = []
print(f"\n{'성분':>5} | {'σ 기준':>8} | {'기울기 기준':>10} | {'LOD (µM)':>9} | {'LOQ (µM)':>9}")
print("-" * 62)
for j, s in enumerate(SUB):
    for sig_tag, sig in (("map", sig_map[j]), ("pixel", sig_pix[j])):
        for sl_tag, sl in (("working", slope_work[s]), ("initial", slope_init[s])):
            lod = 3.3 * sig / sl if sl > 0 else float("nan")
            loq = 10.0 * sig / sl if sl > 0 else float("nan")
            rows.append([s, sig_tag, sl_tag, f"{sig:.4g}", f"{sl:.4g}",
                         f"{lod:.3f}", f"{loq:.3f}"])
            star = "  ←보고" if (sig_tag, sl_tag) == ("map", "working") else ""
            print(f"{s:>5} | {sig_tag:>8} | {sl_tag:>10} | {lod:9.2f} | {loq:9.2f}{star}")

with open(f"{OUT}/lod_loq.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["compound", "sigma_basis", "slope_basis", "sigma_B", "slope_B_per_uM",
                "lod_uM", "loq_uM"])
    w.writerows(rows)
print(f"\n  ✓ lod_loq.csv  {len(rows)}행 → {OUT}")

# ------------------------------------------------------------------ 혼합 LOD
# 단일성분 LOD 는 다른 성분이 없을 때의 값이다. 혼합물에서는 셋이 같은 표면을 나눠
# 쓰므로 자기 농도당 신호(기울기)가 떨어지고, 맵마다 매트릭스가 달라 산포도 커진다.
# 64조건 격자에서 **자기 농도에 대한 B 의 기울기**를 직접 재서 그 둘을 다 반영한다.
import re, glob as _glob, hashlib

LO_MAPS = {}
_seen = set()
for p in sorted(_glob.glob(f"{DB}/Ratio/Baseline260808/DQ*/DQ*_corrected.csv")):
    if "DQ9-sus" in p:
        continue
    h = hashlib.md5(open(p, "rb").read()).hexdigest()
    if h in _seen:
        continue
    _seen.add(h)
    m = re.match(r"DQ(\d+)-TB(\d+)-TH(\d+)", os.path.basename(p))
    LO_MAPS[p] = np.array([int(m.group(i)) / 3.0 for i in (1, 2, 3)])   # 최종 µM

Cm, Bm = [], []
for p, c in LO_MAPS.items():
    _w, cube, _m, _c = load_map(p)
    ya = _map_spectra(cube, mask, 0)[0]
    Cm.append(c); Bm.append(fit_B(ya, P)[0][idx])
Cm = np.array(Cm); Bm = np.array(Bm)
print(f"\n혼합 격자 {len(Cm)}맵 — 자기 농도 대비 B 의 기울기 (원점 통과)")

mix_rows = []
print(f"{'성분':>5} | {'혼합 기울기':>10} | {'단일 대비':>8} | {'잔차 σ':>9} | "
      f"{'LOD(블랭크σ)':>12} | {'LOD(잔차σ)':>11}")
print("-" * 76)
for j, s in enumerate(SUB):
    x, y = Cm[:, j], Bm[:, j]
    sl = float((x @ y) / (x @ x))
    resid = y - sl * x
    sr = float(resid.std(ddof=1))
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss if ss > 0 else np.nan
    lod_b = 3.3 * sig_map[j] / sl
    lod_r = 3.3 * sr / sl
    print(f"{s:>5} | {sl:10.1f} | {100*sl/slope_work[s]:7.0f}% | {sr:9.1f} | "
          f"{lod_b:12.2f} | {lod_r:11.2f}")
    mix_rows.append([s, f"{sl:.4g}", f"{100*sl/slope_work[s]:.1f}", f"{r2:.3f}",
                     f"{sig_map[j]:.4g}", f"{sr:.4g}",
                     f"{lod_b:.3f}", f"{10*sig_map[j]/sl:.3f}",
                     f"{lod_r:.3f}", f"{10*sr/sl:.3f}", str(len(x))])

with open(f"{OUT}/lod_mixture.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["compound", "mixture_slope_B_per_uM", "pct_of_single_compound_slope",
                "r2_B_vs_own_conc", "sigma_blank_B", "sigma_residual_B",
                "lod_uM_blank_sigma", "loq_uM_blank_sigma",
                "lod_uM_residual_sigma", "loq_uM_residual_sigma", "n_maps"])
    w.writerows(mix_rows)
print(f"\n  ✓ lod_mixture.csv  {len(mix_rows)}행")
print("두 가지를 낸다. **블랭크 σ** 는 감도 저하만 반영한 것이고, **잔차 σ** 는 맵마다")
print("매트릭스가 달라 생기는 산포까지 넣은 것이라 실제 혼합물에서 검출을 제한하는 값에")
print("가깝다. 보고할 때 어느 쪽인지 반드시 밝힐 것.")

print("\n보고값은 σ=맵간 · 기울기=작업구간 조합이다 — 판독 단위(맵 하나)와 실제 쓰는")
print("구간의 감도에 맞춘 것. 픽셀 σ 는 같은 맵 픽셀이 상관돼 있어 낙관적이고,")
print("초기 기울기는 등온선이 휘기 전 최대 감도라 역시 낙관적이다.")
print(f"블랭크에 잉크를 포함한 이유: 실제 시료에 늘 있고 템플릿에서 빠져 있어 그 몫이")
print("분석물 계수로 샌다 (INK·DQ 코사인 0.73).")
