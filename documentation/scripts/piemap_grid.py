"""한눈에 보는 64조건 판 — 격자 그대로 놓고 참 vs 예측을 나란히 놓는다.

  파이 65개는 "무엇이 맞았는지"를 못 보여준다. 조각 각도를 눈으로 비교해야 하고,
  참값은 제목의 작은 글씨로만 있어서 둘을 대조할 수가 없다.

  대신:
    · 배치가 실험 격자 그대로다 — DQ 블록 4개 × (TBZ 행 × THI 열) 4×4.
      어디가 어려운지가 위치로 읽힌다.
    · 칸마다 100% 누적 막대 두 줄. 위 = 참, 아래 = 예측.
    · **참값 경계에서 아래로 점선을 내린다.** 예측 경계가 그 선에 붙어 있으면 맞춘 것이다.
      각도를 비교할 필요 없이 선 하나만 보면 된다.
    · 오차는 숫자 하나로만 말한다. 배경을 오차로 칠해 봤더니 THI 의 분홍과 섞여
      막대가 이어져 보였다 — 색은 성분에만 쓴다.
    · 우세 성분을 틀린 칸은 'dominant X'. 참값 1·2위가 2%p 이내면 'tie' 로 빼고
      판정하지 않는다 (33/33/33 에서 우세를 재는 건 반올림을 재는 것이다).

  실행:  python3 -u piemap_grid.py
"""
import os as _os, sys as _sys                 # paths 부트스트랩 — 기계마다 마운트가 다르다
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import os, sys, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DB = paths.DB
sys.path.insert(0, f"{paths.INTERP}/64conditions")
import labfig
labfig.setup()
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch, Rectangle

CO, SUB = labfig.CO, labfig.SUB
COLS = [CO[s] for s in SUB]
INK = "#1c2430"
MUTE = "#5b6673"
LV = [3, 6, 12, 24]
FLOOR = 9.6          # 액적 반복 산포 하한 — 이 아래는 측정으로 구분이 안 된다

d = pickle.load(open(f"{HERE}/piemap_data.pkl", "rb"))
cond = d["cond"]
keys = sorted(cond)
errs = np.array([cond[k]["err"] for k in keys])


def dominant(t, p):
    """'우세 성분을 맞췄나' — 참값 1·2위가 2%p 이내로 붙어 있으면 판정하지 않는다.
    33/33/33 같은 조건에서 우세를 틀렸다고 찍는 건 데이터가 아니라 반올림을 재는 것이다.
    (retrain64.py 도 같은 이유로 max-min<1 을 동률로 뺀다.)"""
    s = np.sort(np.asarray(t, float))[::-1]
    if s[0] - s[1] < 2.0:
        return None
    return int(np.argmax(p)) == int(np.argmax(t))


def err_color(err):
    """하한 이내는 회색(측정 한계까지 맞음), 커질수록 붉게."""
    if err <= FLOOR:
        return "#5b6673"
    if err <= 20:
        return "#c8791f"
    return "#c0392f"


def cell(ax, t, p, err, dom_ok, first=False):
    """참(위)·예측(아래) 100% 누적 막대 + 참값 경계에서 내린 점선.

    배경을 오차로 칠해 봤더니 THI 의 분홍과 섞여 막대가 이어져 보였다. 색은 성분에만
    쓰고, 오차는 숫자 하나로만 말한다."""
    ax.set_xlim(-2, 102); ax.set_ylim(0, 1.42)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for y, v, h in ((0.72, t, 0.30), (0.24, p, 0.30)):
        x = 0.0
        for i in range(3):
            ax.add_patch(Rectangle((x, y), v[i], h, facecolor=COLS[i], lw=0, zorder=2))
            x += v[i]
    # 참값 경계 → 아래로. 예측 경계가 이 선에 붙으면 맞춘 것.
    for b in (t[0], t[0] + t[1]):
        ax.plot([b, b], [0.18, 1.08], ls=(0, (2.2, 1.8)), lw=1.0, color=INK, zorder=4)
    ax.text(0, 1.20, f"{err:.0f}%", ha="left", va="bottom", fontsize=7.6,
            color=err_color(err), fontweight="bold")
    if dom_ok is False:
        ax.text(100, 1.20, "dominant X", ha="right", va="bottom", fontsize=6.4,
                color="#c0392f", fontweight="bold")
    elif dom_ok is None:
        ax.text(100, 1.20, "tie", ha="right", va="bottom", fontsize=6.4, color=MUTE)
    if first:
        ax.text(-3.5, 0.87, "true", ha="right", va="center", fontsize=6.2, color=MUTE)
        ax.text(-3.5, 0.39, "pred", ha="right", va="center", fontsize=6.2, color=MUTE)


fig = plt.figure(figsize=(14.0, 11.8))
outer = GridSpec(2, 2, figure=fig, hspace=0.24, wspace=0.14,
                 left=0.065, right=0.985, top=0.878, bottom=0.098)

for bi, dq in enumerate(LV):
    inner = GridSpecFromSubplotSpec(4, 4, subplot_spec=outer[bi // 2, bi % 2],
                                    hspace=0.34, wspace=0.16)
    be = []
    for r, tbz in enumerate(LV):
        for c, thi in enumerate(LV):
            k = (dq, tbz, thi)
            ax = fig.add_subplot(inner[r, c])
            if k not in cond:
                ax.axis("off"); continue
            t, p = cond[k]["true"], cond[k]["pred"]
            e = cond[k]["err"]; be.append(e)
            cell(ax, t, p, e, dominant(t, p),
                 first=(bi == 0 and r == 0 and c == 0))
            if r == 0:
                ax.set_title(f"THI {thi}" if c == 0 else f"{thi}",
                             fontsize=8, color=CO["THI"], pad=4)
            if c == 0:
                ax.set_ylabel(f"TBZ {tbz}" if r == 0 else f"{tbz}",
                              fontsize=8, color=CO["TBZ"],
                              rotation=0, ha="right", va="center", labelpad=22)
    # 블록 제목
    bb = fig.add_subplot(outer[bi // 2, bi % 2], frameon=False)
    bb.set_xticks([]); bb.set_yticks([])
    bb.patch.set_alpha(0)
    n_ok = sum(1 for e in be if e <= FLOOR)
    bb.set_title(f"DQ {dq} μM        mean {np.mean(be):.1f}%   ·   "
                 f"{n_ok}/{len(be)} at the repeat floor",
                 fontsize=10.5, color=CO["DQ"], pad=30)

_dom = [dominant(cond[k]["true"], cond[k]["pred"]) for k in keys]
n_judged = sum(1 for v in _dom if v is not None)
n_dom = sum(1 for v in _dom if v is True)
n_floor = int((errs <= FLOOR).sum())
n_20 = int((errs <= 20).sum())

fig.suptitle("Composition, condition by condition — top bar = true, bottom bar = predicted\n"
             "the dashed lines drop from the TRUE boundaries: where the lower bar meets them, "
             "the model got it right",
             fontsize=11.5, fontweight="bold", y=0.975)

handles = [Patch(facecolor=CO[s], label=s) for s in SUB]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9.5, frameon=False,
           bbox_to_anchor=(0.5, 0.052))
fig.text(0.5, 0.036, f"error %:  grey ≤ {FLOOR} (at the repeat floor)  ·  "
                     f"amber ≤ 20  ·  red > 20        "
                     f"'dominant X' = wrong largest compound  ·  'tie' = true top two within 2%p, not judged",
         ha="center", fontsize=8.5, color=MUTE)
fig.text(0.5, 0.012,
         f"{len(keys)} conditions · leave-one-condition-out · MLP (blank+sips)        "
         f"dominant compound right {n_dom}/{n_judged} (ties excluded)        "
         f"error ≤ 20%: {n_20}/{len(keys)}        "
         f"at the {FLOOR}% repeat floor: {n_floor}/{len(keys)}        "
         f"mean {errs.mean():.1f}% · median {np.median(errs):.1f}%",
         ha="center", fontsize=9, color=MUTE)
fig.savefig(f"{HERE}/fig12_grid_truepred.png", **labfig.SAVE)
plt.close(fig)
print("fig12_grid_truepred.png")

print(f"\n우세 성분 일치 {n_dom}/{n_judged} (동률 {len(keys)-n_judged}개 제외)  ·  오차≤20% {n_20}/{len(keys)}  ·  "
      f"하한({FLOOR}%) 이내 {n_floor}/{len(keys)}")
print(f"평균 {errs.mean():.1f}%  중앙 {np.median(errs):.1f}%")
for dq in LV:
    e = [cond[k]["err"] for k in keys if k[0] == dq and all(v in LV for v in k)]
    print(f"  DQ {dq:>2} μM  평균 {np.mean(e):5.1f}%  (n={len(e)})")
off = [k for k in keys if not all(v in LV for v in k)]
if off:
    print("격자 밖(그림에 없음):", ", ".join(f"{k} err {cond[k]['err']:.0f}%" for k in off))
