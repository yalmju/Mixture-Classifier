"""그림 패널을 한 장씩 따로 떨어뜨린다 — 조판은 밖에서 한다.

  삼각형은 Origin 으로 옮기기 어려워 여기서 그린 것을 그대로 쓴다. 그래서 **제목도
  여백도 없이** 삼각형만 나온다 — 붙일 자리 크기에 맞춰 `--size` 로 뽑는다.
  나머지 패널은 Origin 에서 다시 그릴 수 있게 `export_origin.py` 가 CSV 를 낸다.

  파일명은 패널 문자다. 순서를 바꾸려면 아래 PANELS 의 문자만 고치면 된다.

    a  삼각형 — 참 조성 RGB                        (그림 그대로 사용)
    b  삼각형 — 정확도 음영 + 참→예측 화살표        (그림 그대로 사용)
    c  격자 — 위=참 / 아래=예측
    d  격자 — 오차 히트맵
    e  조건별 조성 파이
    f  조건별 10×10 픽셀 조성맵
    g  농도 회수 — 파리티 3개 + 배수오차 누적분포

  주의: matplotlib 의 글자 크기는 **점(pt)** 단위라 그림 크기를 줄이면 글자만 상대적으로
  커진다. 삼각형을 작게 뽑을 때는 `--fontscale` 로 같이 줄여야 비율이 맞는다.

  실행:
      python3 -u panels.py                          기본 크기로 전부
      python3 -u panels.py --size 3.4,3.3 --fontscale 0.55     삼각형만 작게
      python3 -u panels.py --only a,b
"""
import os, sys, csv, json, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

import labfig
labfig.setup()
import matplotlib
from matplotlib.figure import Figure
import grid_figs as GF
import triangle_figs as TF

DB = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB"
INTERP = f"{DB}/260808_data interpret"
CACHE = f"{INTERP}/piemap_260812"
ORIGIN = f"{INTERP}/origin_data"
OUT = f"{INTERP}/panels"
CO, SUB = labfig.CO, labfig.SUB
COLS = [CO[s] for s in SUB]
SAVE = dict(dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.01)

# ---- 인자
args = sys.argv[1:]


def opt(name, default=None):
    return args[args.index(name) + 1] if name in args and args.index(name) + 1 < len(args) \
        else default


TRI_SIZE = tuple(float(v) for v in opt("--size", "5.2,5.0").split(","))
FONTSCALE = float(opt("--fontscale", "1.0"))
ONLY = set(opt("--only", "").split(",")) - {""}
os.makedirs(OUT, exist_ok=True)

if FONTSCALE != 1.0:
    for k in ("font.size", "axes.titlesize", "axes.labelsize",
              "xtick.labelsize", "ytick.labelsize", "legend.fontsize"):
        matplotlib.rcParams[k] = matplotlib.rcParams[k] * FONTSCALE
    # 삼각형은 성분 이름·범례 크기를 rcParams 가 아니라 자기 상수로 잡고 있어서
    # rcParams 만 줄이면 삼각형만 글자가 그대로 남는다.
    TF.SCALE = FONTSCALE

# ---- 자료
d = pickle.load(open(f"{CACHE}/piemap_data.pkl", "rb"))
cond, maps = d["cond"], d["maps"]
keys = sorted(cond)
rows_pct = [("/".join(str(v) for v in k) + " µM", cond[k]["true"], cond[k]["pred"])
            for k in keys]
# triangle_figs 는 0–1 분율을 받는다 (bary·정확도 모두). %로 넘기면 삼각형 밖으로 나간다.
rows_frac = [(n, np.asarray(t, float) / 100.0, np.asarray(p, float) / 100.0)
             for n, t, p in rows_pct]
first = [("/".join(str(v) for v in k) + " µM",
          maps[cond[k]["paths"][0]]["coords"], maps[cond[k]["paths"][0]]["ratio"],
          maps[cond[k]["paths"][0]]["hit"]) for k in keys]


NOLEGEND = "--nolegend" in args


def _strip_legend(fig):
    """범례를 떼면 삼각형 아래 빈 띠가 사라진다. 조판에서 범례를 한 번만 둘 때 쓴다."""
    for lg in list(getattr(fig, "legends", [])):
        lg.remove()
    for ax in fig.axes:
        lg = ax.get_legend()
        if lg is not None:
            lg.remove()
    return fig


def _tri(fn):
    """삼각형은 자기 크기의 Figure 를 스스로 만든다 — 붙일 자리에 맞춰 우리가 준다."""
    def make():
        f = fn(rows_frac, SUB, COLS, fig=Figure(figsize=TRI_SIZE))
        return _strip_legend(f) if NOLEGEND else f
    return make


def _read(name):
    with open(os.path.join(ORIGIN, name), encoding="utf-8-sig") as f:
        r = list(csv.reader(f))
    return r[0], r[1:]


def _conc_tables():
    """패널 g 의 표. `export_origin.py` 가 낸 것을 그대로 넘긴다 — 숫자 출처는 하나."""
    out = []
    for tag, fn in (("parity", "concentration_parity.csv"),
                    ("fold_cdf", "concentration_fold_cdf.csv"),
                    ("summary", "concentration_summary.csv")):
        if os.path.exists(os.path.join(ORIGIN, fn)):
            out.append((tag, *_read(fn)))
    return out


# 삼각형만 그림으로 나간다 (Origin 으로 옮기기 어려워 그린 것을 그대로 쓴다).
# 나머지는 표다 — Origin 이 읽을 것이므로 PNG 를 낼 이유가 없다.
FIGURES = {
    "a": ("triangle_true_rgb", _tri(TF.rgb_triangle)),
    "b": ("triangle_accuracy", _tri(TF.accuracy_triangle)),
}
TABLES = {
    "c": ("grid_true_vs_predicted",
          lambda: [("", *GF.condition_table(rows_pct, keys, SUB))]),
    "d": ("grid_error",
          lambda: [(nm, h, b) for nm, h, b in GF.error_matrix_tables(rows_pct, keys, SUB)]),
    "e": ("composition_pies",
          lambda: [("", *GF.condition_table(rows_pct, keys, SUB))]),
    "f": ("pixel_maps", lambda: [("", *GF.pixel_table(first, SUB))]),
    "g": ("concentration_recovery", _conc_tables),
}

print(f"→ {OUT}\n삼각형 크기 {TRI_SIZE[0]}×{TRI_SIZE[1]} in · 글자 배율 {FONTSCALE}\n")

for letter, (name, make) in FIGURES.items():
    if ONLY and letter not in ONLY:
        continue
    fig = make()
    fig.savefig(f"{OUT}/panel_{letter}_{name}.png", **SAVE)
    print(f"  ✓ panel_{letter}_{name}.png   (그림 — 그대로 사용)")

for letter, (name, make) in TABLES.items():
    if ONLY and letter not in ONLY:
        continue
    tabs = make()
    if not tabs:
        print(f"  ✗ {letter}  {name} — 자료 없음"); continue
    for tag, head, body in tabs:
        fn = f"panel_{letter}_{name}" + (f"_{tag}" if tag else "") + ".csv"
        with open(f"{OUT}/{fn}", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(head); w.writerows(body)
        print(f"  ✓ {fn:<48} {len(body)}행 × {len(head)}열")

print("\n삼각형(a·b)만 PNG — 배경 투명 · 600 dpi. 나머지는 CSV 이고 Origin 에서 그린다.")
print("d 는 히트맵이라 블록마다 행렬 한 장씩이다 (Origin heat map 은 행렬을 받는다).")
print("패널 문자를 바꾸려면 FIGURES / TABLES 의 키만 고치면 된다.")
