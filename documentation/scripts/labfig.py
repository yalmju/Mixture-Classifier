"""랩미팅 그림 공통 스타일 — 앱(ui_common.py)의 cnsplots 형식과 Pure/colors.json 색을 그대로 쓴다.

  색은 하드코딩하지 않는다. `ACF_PEST_DB/Pure/colors.json` 이 단일 출처이고,
  앱 상단바에서 색을 바꾸면 이 그림도 따라 바뀐다.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler

DB = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB"
SUB = ["DQ", "TBZ", "THI"]

_FALLBACK = {"DQ": "#008cf7", "TBZ": "#95dc2a", "THI": "#f35376"}


def colors(data_dir=f"{DB}/Pure"):
    """앱이 저장한 성분별 색 (없으면 기본값)."""
    try:
        m = json.load(open(os.path.join(data_dir, "colors.json")))
        return {s: m.get(s, _FALLBACK[s]) for s in SUB}
    except Exception:
        return dict(_FALLBACK)


CO = colors()


EXPORT_DPI = 600          # ui_common.EXPORT_DPI — 앱 내보내기와 같은 값
SAVE = dict(dpi=EXPORT_DPI, transparent=True, bbox_inches="tight", pad_inches=0.01)


def setup(korean=False):
    """cnsplots 형식 — 얇은 despine 축, 작은 타이포, 촘촘한 틱, 테두리 없는 범례.
    기본은 앱과 같은 Helvetica다. 한글 라벨이 필요할 때만 korean=True (폰트가 바뀌므로
    앱 내보내기와 형식이 미세하게 달라진다)."""
    fam = (["AppleGothic", "Helvetica", "Arial", "DejaVu Sans"] if korean
           else ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"])
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": fam,
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlelocation": "center", "axes.titlepad": 4,
        "axes.labelsize": 10, "axes.labelpad": 2,
        "axes.linewidth": 0.5, "axes.edgecolor": "black", "axes.labelcolor": "black",
        "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.xmargin": 0.05, "axes.ymargin": 0.05,
        "axes.prop_cycle": cycler(color=[CO[s] for s in SUB]),
        "xtick.color": "black", "ytick.color": "black",
        "xtick.labelcolor": "black", "ytick.labelcolor": "black",
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.pad": 1, "ytick.major.pad": 1,
        "legend.frameon": False, "legend.fontsize": 9,
        "legend.markerscale": 0.9, "legend.handlelength": 0.9,
        "legend.handleheight": 0.7, "legend.handletextpad": 0.4,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "figure.dpi": 110,
    })


def unity(ax, lo, hi, label="1:1"):
    """1:1 기준선 — 회색 점선, 데이터 뒤로."""
    ax.plot([lo, hi], [lo, hi], ls=(0, (4, 3)), lw=0.8, color="#9aa3ad",
            zorder=1, label=label)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
