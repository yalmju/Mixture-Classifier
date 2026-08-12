"""32조건을 DQ 고정 블록 2개(16 + 16)로 나눠 4×4 격자로 본다.

  Baseline260808은 원액 기준으로 DQ 36 · DQ 72 두 블록이고, 각 블록 안에서
  TBZ × THI 가 9/18/36/72 의 4×4 다. 한 줄로 32개를 늘어놓으면 안 보이던 규칙이
  블록으로 자르면 바로 보인다.

    라벨은 전부 **최종 농도(µM)** 다. 원액은 이 값의 3배.

  블록마다 두 가지를 그린다.
    파이   조건별 예측 조성 (앱 page_real._plot_comp 과 같은 형식)
    히트맵 조성 오차 — 어느 칸이 어려운지 한눈에

  실행:  python3 -u grid16.py
"""
import os, sys, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import labfig
labfig.setup()
import matplotlib.pyplot as plt

CO, SUB = labfig.CO, labfig.SUB
COLS = [CO[s] for s in SUB]
DESK = "/Users/seungki2/Desktop"
INK = "#1c2430"
LV = [3, 6, 12, 24]                     # 최종 µM
STOCK = {3: 9, 6: 18, 12: 36, 24: 72}

# ------------------------------------------------------------------ 데이터
mlp = {}
for l in open(f"{DESK}/retrain38.txt"):
    if not re.search(r"\|\s+[\d.]+% \|", l):
        continue
    p = [x.strip() for x in l.split("|")]
    k = tuple(int(float(v)) for v in p[0].split("/"))
    mlp[k] = (np.array([float(v) for v in p[1].split("/")]),
              np.array([float(v) for v in p[2].split("/")]))

BLOCKS = [(12, "DQ 12 μM"), (24, "DQ 24 μM")]


# ------------------------------------------------------------------ 그림 7: 파이 격자
fig = plt.figure(figsize=(13.4, 7.4))
outer = fig.subfigures(1, 2, wspace=0.06)
for bi, (dq, title) in enumerate(BLOCKS):
    sf = outer[bi]
    axes = sf.subplots(4, 4)
    for r, tbz in enumerate(LV):
        for c, thi in enumerate(LV):
            ax = axes[r, c]; ax.axis("off")
            k = (dq, tbz, thi)
            if k not in mlp:
                continue
            t, p = mlp[k]
            pf = p / 100
            keep = [i for i in range(3) if pf[i] >= 0.01] or [int(np.argmax(pf))]
            ax.pie([pf[i] for i in keep], colors=[COLS[i] for i in keep],
                   autopct="%1.0f%%", textprops={"fontsize": 6, "color": INK},
                   radius=1.15)
            ax.set_aspect("equal")
            e = np.abs(p - t).sum() / 2
            ax.set_title(f"true {t[0]:.0f}/{t[1]:.0f}/{t[2]:.0f}   err {e:.0f}%",
                         fontsize=6.2, pad=1, color="#5b6673")
            if c == 0:
                ax.text(-1.95, 0, f"TBZ {tbz}", fontsize=8.5,
                        ha="center", va="center", rotation=90, color=CO["TBZ"],
                        fontweight="bold")
            if r == 0:
                ax.text(0, 1.95, f"THI {thi}", fontsize=8.5,
                        ha="center", va="bottom", color=CO["THI"], fontweight="bold")
    sf.suptitle(title, fontsize=11, fontweight="bold", color=CO["DQ"])
import matplotlib.patches as mp
fig.legend(handles=[mp.Patch(fc=CO[s], label=s) for s in SUB],
           loc="lower center", ncol=3, frameon=False, fontsize=10)
fig.suptitle("Predicted composition per condition — split by fixed DQ (16 + 16)",
             fontsize=12, fontweight="bold")
fig.savefig(f"{DESK}/fig7_grid_pies.png", **labfig.SAVE)
print("fig7_grid_pies.png")

# ------------------------------------------------------------------ 그림 8: 오차 히트맵
fig2, axs = plt.subplots(1, 2, figsize=(10.4, 4.5))
E = {}
for bi, (dq, title) in enumerate(BLOCKS):
    M = np.full((4, 4), np.nan)
    for r, tbz in enumerate(LV):
        for c, thi in enumerate(LV):
            k = (dq, tbz, thi)
            if k in mlp:
                t, p = mlp[k]; M[r, c] = np.abs(p - t).sum() / 2
    E[dq] = M
    ax = axs[bi]
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=30, origin="upper")
    for r in range(4):
        for c in range(4):
            if np.isfinite(M[r, c]):
                ax.text(c, r, f"{M[r, c]:.0f}", ha="center", va="center", fontsize=10,
                        color="white" if M[r, c] > 18 else INK, fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels([str(v) for v in LV], fontsize=9.5)
    ax.set_yticks(range(4)); ax.set_yticklabels([str(v) for v in LV], fontsize=9.5)
    ax.set_xlabel("THI (μM)", color=CO["THI"])
    ax.set_ylabel("TBZ (μM)", color=CO["TBZ"])
    ax.set_title(f"{title}\nmean {np.nanmean(M):.1f}%", fontsize=10, color=CO["DQ"])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
cb = fig2.colorbar(im, ax=axs, fraction=.035, pad=.03)
cb.set_label("composition error (%)")
cb.outline.set_visible(False)
fig2.suptitle("Where is composition hard? — error per grid cell",
              fontsize=11.5, fontweight="bold")
fig2.savefig(f"{DESK}/fig8_grid_error.png", **labfig.SAVE)
print("fig8_grid_error.png")

for dq, title in BLOCKS:
    M = E[dq]
    print(f"\n{title}   평균 {np.nanmean(M):.1f}%")
    print("        THI " + "  ".join(f"{v:>5}" for v in LV))
    for r, tbz in enumerate(LV):
        print(f"  TBZ {tbz:>3}  " + "  ".join(
            f"{M[r, c]:5.1f}" if np.isfinite(M[r, c]) else "    -" for c in range(4)))
