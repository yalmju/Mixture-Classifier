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


def _conc():
    """패널 g 는 CSV 만 읽는 별도 스크립트를 그대로 부른다 (숫자 출처를 하나로)."""
    import subprocess
    subprocess.run([sys.executable, "-u", f"{HERE}/fig_h_concentration.py", ORIGIN],
                   check=True, capture_output=True)
    src = f"{INTERP}/figh_concentration64.png"
    return src if os.path.exists(src) else None


PANELS = {
    "a": ("triangle_true_rgb", _tri(TF.rgb_triangle)),
    "b": ("triangle_accuracy", _tri(TF.accuracy_triangle)),
    "c": ("grid_true_vs_predicted", lambda: GF.grid_truepred(rows_pct, keys, SUB, COLS)),
    "d": ("grid_error", lambda: GF.grid_error(rows_pct, keys, SUB)),
    "e": ("composition_pies", lambda: GF.composition_pies(rows_pct, SUB, COLS)),
    "f": ("pixel_maps", lambda: GF.pixel_maps(first, SUB, COLS)),
    "g": ("concentration_recovery", _conc),
}

print(f"→ {OUT}\n삼각형 크기 {TRI_SIZE[0]}×{TRI_SIZE[1]} in · 글자 배율 {FONTSCALE}\n")
for letter, (name, make) in PANELS.items():
    if ONLY and letter not in ONLY:
        continue
    got = make()
    dst = f"{OUT}/panel_{letter}_{name}.png"
    if isinstance(got, str):                       # 이미 파일로 나온 패널
        import shutil
        shutil.copyfile(got, dst)
    elif got is None:
        print(f"  ✗ {letter}  {name} — 자료 없음"); continue
    else:
        got.savefig(dst, **SAVE)
    print(f"  ✓ panel_{letter}_{name}.png")

print("\n조판 밖에서 붙일 때: 전부 배경 투명 · 600 dpi. 패널 문자를 바꾸려면 "
      "panels.py 의 PANELS 키만 고치면 된다.")
