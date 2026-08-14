"""패널 a — NNLS 삼각형. 선형 분해가 어디서 무너지는지 보여주는 판.

  a 와 g 는 **같은 그림을 다른 추정기로** 그린 것이다. 색은 정확도(RdYlGn)라
  NNLS 는 노랗고 학습 모델은 초록이다 — 그 대비가 논점이다.
  RGB 삼각형(참 조성으로 색칠한 것)이 아니다.

  NNLS 는 학습이 없으므로 leave-one-out 이 의미가 없다. 모든 맵에 그대로 적용한다.
  `dl_model._fit_predict("nnls", …)` 와 같은 경로를 쓴다 (`surface_composition`).

  나오는 것 (`260808_data interpret/panels/`)
    panel_a_triangle_nnls.png    삼각형 (--size/--fontscale/--bare 는 panels.py 와 같다)
    panel_a_nnls_conditions.csv  조건별 참/예측/오차 — 꼭짓점 회수율까지 여기서 나온다

  실행:  python3 -u export_nnls.py [--size W,H] [--fontscale F] [--bare]
"""
import os, sys, csv, re, glob, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

import labfig
labfig.setup()
import matplotlib
from matplotlib.figure import Figure
import paths
import mixtures as MX
import grid_figs as GF
import triangle_figs as TF
from dl_model import _refs, _map_spectra, _composition_features
from dl_quantify import surface_composition
from real_data import load_map

DB = paths.DB
OUT = f"{paths.INTERP}/panels"
SUB = ["DQ", "TBZ", "THI"]
CO = labfig.CO
COLS = [CO[s] for s in SUB]
SAVE = dict(dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.01)
os.makedirs(OUT, exist_ok=True)

args = sys.argv[1:]


def opt(n, d=None):
    return args[args.index(n) + 1] if n in args and args.index(n) + 1 < len(args) else d


SIZE = tuple(float(v) for v in opt("--size", "5.2,5.0").split(","))
FS = float(opt("--fontscale", "1.0"))
BARE = "--bare" in args
SET = opt("--set", "high")      # high = Ratio_mix, low = 64조건 격자, all = 둘 다
if FS != 1.0:
    for k in ("font.size", "axes.titlesize", "axes.labelsize",
              "xtick.labelsize", "ytick.labelsize", "legend.fontsize"):
        matplotlib.rcParams[k] = matplotlib.rcParams[k] * FS
    TF.SCALE = FS

# ------------------------------------------------------------------ 데이터
# 저농도 격자에서 NNLS 는 초록이다 — 회수율 109/120/112%, 조성오차 15.1%.
# THI 를 크게 과대평가하는 노란 삼각형은 **고농도** 세트다. 거기서는 THI 가 표면을
# 독점해 회수율이 크게 벌어진다.
#   high = 예전 Ratio_mix 34조건 (최종 3–500 µM)   low = 64조건 격자
# 등몰 계열은 여기 안 들어간다 — 예전과 같은 구성이다 (그 세트는 export_tert.py).
_SETS = {"high": ("high",), "low": ("grid",),
         "all": ("high", "grid", "equimolar")}[SET]
LO = MX.groups(_SETS, replicates=True)

keys = sorted(LO)
subs, wn, mask, P, lo, hi = _refs(f"{DB}/Pure", True, None)
idx = [subs.index(s) for s in SUB]
print(f"[{SET}] {len(keys)}조건 {sum(len(v) for v in LO.values())}맵 · 템플릿 {subs}")

rows_pct, concs = [], []
for k in keys:
    per_map = []
    for p in LO[k]:
        _w, cube, _m, _c = load_map(p)
        # px_per_map=0 → 맵당 평균 스펙트럼 한 줄. 앱의 NNLS 판독과 같은 단위다.
        X = np.array([_composition_features(ya) for ya in _map_spectra(cube, mask, 0)])
        raw = np.expm1(np.clip(X, 0, None))
        per_map.append(surface_composition(_composition_features(raw, "legacy_l2"), P).mean(0))
    pred = np.mean(per_map, axis=0)[idx]
    pred = pred / (pred.sum() or 1.0) * 100
    C = np.array(k, float)
    true = C / C.sum() * 100
    rows_pct.append(("/".join(str(v) for v in k) + " µM", true, pred))
    concs.append(k)

errs = np.array([GF.comp_error(r[1], r[2]) for r in rows_pct])
print(f"NNLS 조성 오차  평균 {errs.mean():.1f}%  중앙 {np.median(errs):.1f}%")
for i, s in enumerate(SUB):
    t = np.array([r[1][i] for r in rows_pct]); p_ = np.array([r[2][i] for r in rows_pct])
    # 이원 혼합은 참값이 0인 성분이 있어 회수율이 정의되지 않는다 → 그 조건만 뺀다
    # (np.mean 이면 nan 하나가 전체를 nan 으로 만든다)
    ok = t > 0
    print(f"  {s:<4} 회수율 {np.mean(p_[ok] / t[ok]) * 100:6.0f}%  (n={ok.sum()})  ·  "
          f"편향 {p_.mean() - t.mean():+5.1f}%p")

# ------------------------------------------------------------------ 그림 + 표
rows_frac = [(n, t / 100.0, p_ / 100.0) for n, t, p_ in rows_pct]
fig = TF.accuracy_triangle(rows_frac, SUB, COLS, fig=Figure(figsize=SIZE))
if BARE:
    for lg in list(getattr(fig, "legends", [])):
        lg.remove()
    for ax in fig.axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    for ax in list(fig.axes)[1:]:
        ax.remove()
    ax = fig.axes[0]
    ax.set_xlim(-0.14, 1.14); ax.set_ylim(-0.20, 3 ** 0.5 / 2 + 0.13)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
fig.savefig(f"{OUT}/panel_a_triangle_nnls_{SET}.png", **SAVE)
print(f"\n  ✓ panel_a_triangle_nnls_{SET}.png")

head, body = GF.condition_table(rows_pct, concs, SUB)
with open(f"{OUT}/panel_a_nnls_conditions_{SET}.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(head); w.writerows(body)
print(f"  ✓ panel_a_nnls_conditions_{SET}.csv   {len(body)}행 × {len(head)}열")
print("\na 와 g 는 같은 그림·다른 추정기다. 색이 정확도라 a 는 노랗고 g 는 초록이다.")
print("NNLS 는 학습이 없어 held-out 개념이 없다 — 캡션에 leave-one-out 이라 쓰지 말 것.")
