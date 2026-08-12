"""260805 삼원 등몰 시리즈 — 데이터만.

  참 조성은 네 농도 모두 33.3/33.3/33.3.
  왼쪽 NNLS(표면에 실제로 보이는 것) · 오른쪽 MLP(3–24 µM/성분으로 학습, 여기선 외삽).

  해석은 붙이지 않는다. 적합한 K는 TBZ가 가장 큰데(DQ 1 : TBZ 4.8 : THI 4.4) 실측은
  고농도에서 THI만 남는다 — 경쟁 Langmuir로 설명되지 않으므로 전이선도 긋지 않는다.

  실행:  python3 -u tert260805_fig.py
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import labfig
labfig.setup()
import matplotlib.pyplot as plt

CO, SUB = labfig.CO, labfig.SUB
DESK = "/Users/seungki2/Desktop"

d = np.load(f"{HERE}/tert260805_result.npz", allow_pickle=True)
per, pred = d["per_uM"], d["pred"]
lv = sorted(set(per.tolist()))
mlp_mu = np.array([pred[per == v].mean(0) for v in lv])
nn_mu = np.load(f"{HERE}/tert260805_nnls.npz")["comp"]

fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
xs = np.arange(len(lv))
for pi, mu in enumerate([nn_mu, mlp_mu]):
    A = ax[pi]
    bot = np.zeros(len(lv))
    for j, s in enumerate(SUB):
        A.bar(xs, mu[:, j], bottom=bot, color=CO[s], width=.62,
              edgecolor="white", lw=.8, label=s if pi == 0 else None)
        for k in range(len(lv)):
            if mu[k, j] > 7:
                A.text(xs[k], bot[k] + mu[k, j] / 2, f"{mu[k, j]:.0f}", ha="center",
                       va="center", fontsize=9, color="white", fontweight="bold")
        bot += mu[:, j]
    A.axhline(33.3, ls=(0, (4, 3)), lw=.9, color="#5b6673", zorder=5)
    A.set_xticks(xs); A.set_xticklabels([f"{v:.0f}" for v in lv])
    A.set_xlim(-.6, len(lv) - .4); A.set_ylim(0, 100)
    A.set_xlabel("concentration per compound (μM)")
    if pi == 0:
        A.set_ylabel("composition (%)")
ax[0].legend(loc="upper center", ncol=3, bbox_to_anchor=(1.03, -.18))
fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig(f"{DESK}/fig6_tert_series.png", **labfig.SAVE)
print("fig6_tert_series.png\n")
print(f"{'성분당 µM':>10} | {'NNLS DQ/TBZ/THI':>22} | {'MLP DQ/TBZ/THI':>22}")
for i, v in enumerate(lv):
    print(f"{v:9.1f} | " + "/".join(f"{x:6.1f}" for x in nn_mu[i]) +
          " | " + "/".join(f"{x:6.1f}" for x in mlp_mu[i]))
