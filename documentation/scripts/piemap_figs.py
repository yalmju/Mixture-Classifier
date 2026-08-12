"""파이 + 픽셀맵 재생성 — 2단계: 그림 (piemap_data.pkl 만 읽는다, 빠름).

  fig5_pies65.png        조건별 조성 파이 65개 — map-level, leave-one-condition-out
  fig11_pixelmap65.png   조건별 10×10 픽셀 파이맵 65개 (조건당 첫 맵)
  pixelmaps/<맵>.png     맵 하나짜리 큰 픽셀 파이맵 80장 — 앱 Real data 탭과 같은 그림

  색·형식은 labfig(= 앱 ui_common + Pure/colors.json). 배경 투명, 600 dpi.
  실행:  python3 -u piemap_figs.py
"""
import os, sys, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DB = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB"
sys.path.insert(0, f"{DB}/260808_data interpret/64conditions")
import labfig
labfig.setup()
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch
from matplotlib.collections import PatchCollection

CO, SUB = labfig.CO, labfig.SUB
INK = "#1c2430"
BG_GREY = "#c9ced6"

d = pickle.load(open(f"{HERE}/piemap_data.pkl", "rb"))
cond, maps, keys = d["cond"], d["maps"], sorted(d["cond"])
print(f"{len(keys)}조건 · {len(maps)}맵 · 구성 {d['arm']}")


def nice(k):
    return f"{k[0]}/{k[1]}/{k[2]}"


# ------------------------------------------------------------------ fig5: 조건별 파이 65개
nc = 9
nr = int(np.ceil(len(keys) / nc))
fig, axes = plt.subplots(nr, nc, figsize=(1.62 * nc, 1.92 * nr))
for ax in np.ravel(axes):
    ax.axis("off")
for idx, k in enumerate(keys):
    ax = np.ravel(axes)[idx]
    t, p = cond[k]["true"], cond[k]["pred"]
    keep = [i for i in range(3) if p[i] >= 1.0] or [int(np.argmax(p))]
    ax.pie([p[i] for i in keep], labels=[SUB[i] for i in keep],
           colors=[CO[SUB[i]] for i in keep], autopct="%1.0f%%",
           textprops={"fontsize": 6.0, "color": INK}, radius=1.0)
    ax.set_aspect("equal")
    ax.set_title(f"{nice(k)} μM\ntrue {'/'.join(f'{v:.0f}' for v in t)}  ·  "
                 f"err {cond[k]['err']:.0f}%", fontsize=6.6, pad=1.5)
errs = [cond[k]["err"] for k in keys]
fig.suptitle(f"Predicted composition per condition — MLP (blank+sips), "
             f"leave-one-condition-out · {len(keys)} conditions, "
             f"mean {np.mean(errs):.1f}% · median {np.median(errs):.1f}%",
             fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0, 0.005, 1, 0.965])
fig.savefig(f"{HERE}/fig5_pies65.png", **labfig.SAVE)
plt.close(fig)
print("fig5_pies65.png")


# ------------------------------------------------------------------ 픽셀 파이맵 그리기
def pixel_pies(ax, m, rad_scale=0.46, bg_size=16):
    """앱 page_real._plot_pies 와 같은 그림: hit 픽셀은 파이, 나머지는 회색 사각."""
    x, y = m["coords"][:, 0], m["coords"][:, 1]
    ux = np.unique(x)
    rad = (np.median(np.diff(ux)) * rad_scale) if len(ux) > 1 else rad_scale
    hit = m["hit"]
    cols = [CO.get(c, BG_GREY) for c in m["comps"]]
    if (~hit).any():
        ax.scatter(x[~hit], y[~hit], c=BG_GREY, marker="s", s=bg_size, edgecolors="none")
    wedges, wcols = [], []
    for i in np.where(hit)[0]:
        a0 = 90.0
        for k, frac in enumerate(m["ratio"][i]):
            if frac <= 0.002:
                continue
            a1 = a0 - frac * 360.0
            wedges.append(Wedge((x[i], y[i]), rad, a1, a0))
            wcols.append(cols[k])
            a0 = a1
    if wedges:
        ax.add_collection(PatchCollection(wedges, facecolors=wcols, edgecolors="none"))
    ax.set_xlim(x.min() - 1, x.max() + 1)
    ax.set_ylim(y.max() + 1, y.min() - 1)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return cols


# ------------------------------------------------------------------ fig11: 픽셀맵 65개
first = {k: cond[k]["paths"][0] for k in keys}
fig, axes = plt.subplots(nr, nc, figsize=(1.58 * nc, 1.80 * nr))
for ax in np.ravel(axes):
    ax.axis("off")
for idx, k in enumerate(keys):
    ax = np.ravel(axes)[idx]
    ax.set_axis_on()
    m = maps[first[k]]
    pixel_pies(ax, m, bg_size=9)
    ax.set_title(f"{nice(k)} μM  ·  hit {100*m['hit_frac']:.0f}%", fontsize=6.6, pad=2)
handles = [Patch(facecolor=CO[s], label=s) for s in SUB] + \
          [Patch(facecolor=BG_GREY, label="background")]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, 0.002))
fig.suptitle("Per-pixel composition map (10×10) — NNLS decides hit, the model composes "
             "· leave-one-condition-out", fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0, 0.028, 1, 0.965])
fig.savefig(f"{HERE}/fig11_pixelmap65.png", **labfig.SAVE)
plt.close(fig)
print("fig11_pixelmap65.png")


# ------------------------------------------------------------------ 맵별 개별 그림
os.makedirs(f"{HERE}/pixelmaps", exist_ok=True)
for p, m in sorted(maps.items()):
    k = m["key"]
    fig, ax = plt.subplots(figsize=(4.0, 4.2))
    pixel_pies(ax, m)
    mr = m["mean_ratio"] * 100
    ax.set_title(f"DQ {k[0]} · TBZ {k[1]} · THI {k[2]} μM\n"
                 f"true {'/'.join(f'{v:.0f}' for v in cond[k]['true'])}  ·  "
                 f"pixel mean {'/'.join(f'{v:.0f}' for v in mr)}  ·  "
                 f"hit {100*m['hit_frac']:.0f}%", fontsize=8.5, pad=4)
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.01), ncol=4)
    fig.tight_layout()
    fig.savefig(f"{HERE}/pixelmaps/{os.path.basename(p).replace('_corrected.csv','')}"
                f"__DQ{k[0]}-TBZ{k[1]}-THI{k[2]}.png", **labfig.SAVE)
    plt.close(fig)
print(f"pixelmaps/ — {len(maps)}장")

# ------------------------------------------------------------------ 콘솔 요약
print("\n조건별 (참 → 예측, 오차)")
for k in keys:
    c = cond[k]
    print(f"  {nice(k):>10} μM | " + "/".join(f"{v:5.1f}" for v in c["true"]) +
          " | " + "/".join(f"{v:5.1f}" for v in c["pred"]) + f" | {c['err']:5.1f}%")
print(f"\n평균 {np.mean(errs):.1f}%  중앙 {np.median(errs):.1f}%")
hf = np.array([m["hit_frac"] for m in maps.values()])
print(f"hit 비율  중앙 {100*np.median(hf):.0f}%  범위 {100*hf.min():.0f}–{100*hf.max():.0f}%")
