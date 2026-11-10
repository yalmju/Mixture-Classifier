# -*- coding: utf-8 -*-
"""61 — 재분배 법칙의 체제별 요약 (Discussion 표를 그림으로).

데이터: 50_redistribution_enrichment_RAW.csv (s, E, thi_share, total; 92조건).
체제 정의는 Discussion 표와 동일(비배타): THI 소수(share≤45%) /
THI 다수(share>45%) / 포화(총>50 µM). 성분별 median E + IQR, log2 축,
truth=1 점선. 색 = Pure/colors.json.

실행:  python -u fig61_regime_summary.py
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import labfig  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(HERE, "..", "results")
CO = {s: labfig.CO[s] for s in ("DQ", "TBZ", "THI")}
ORDER = ["THI", "TBZ", "DQ"]
INK = "black"

master = [r for r in csv.DictReader(
    open(os.path.join(RES, "17_composition_all_conditions_master.csv"),
         encoding="utf-8-sig")) if r["imbalance_100x"] == "0"]
# 의심맵 3장 (전용 밴드 기준 라벨과 어긋남 — 07-24/07-27) 제외:
# (thi_share,total) 필터는 거울쌍(DQ0-TB250-TH50 등)과 충돌해서 조건명으로.
SUSPECT = {(250.0, 0.0, 50.0), (250.0, 50.0, 0.0), (500.0, 50.0, 0.0)}
master = [r for r in master
          if (float(r["DQ"]), float(r["TBZ"]), float(r["THI"])) not in SUSPECT]

REG = [("THI minority\n(share ≤ 45%)", lambda sh, to: sh <= 45),
       ("THI majority\n(share > 45%)", lambda sh, to: sh > 45),
       ("saturated\n(total > 50 µM)", lambda sh, to: to > 50)]

vals = {s: [[] for _ in REG] for s in CO}
for r in master:
    dq, tb, th = float(r["DQ"]), float(r["TBZ"]), float(r["THI"])
    to = dq + tb + th
    sh = th / to * 100 if to else 0.0
    for s in CO:
        t = float(r[f"Ratio_{s}_True"])
        if t <= 0:
            continue
        e = float(r[f"Ratio_{s}_NNLS_Pred"]) / t
        for j, (_, f) in enumerate(REG):
            if f(sh, to):
                vals[s][j].append(e)

fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.set_yscale("log", base=2)
ax.axhline(1.0, color=INK, lw=1.1, ls=(0, (4, 3)), zorder=2)
for f in (0.5, 2.0):
    ax.axhline(f, color="0.86", lw=0.6, ls=":", zorder=1)
DX = {"THI": -0.16, "TBZ": 0.0, "DQ": 0.16}
for s in ORDER:
    x = np.arange(len(REG)) + DX[s]
    med = np.array([np.median(v) if v else np.nan for v in vals[s]])
    q1 = np.array([np.percentile(v, 25) if v else np.nan for v in vals[s]])
    q3 = np.array([np.percentile(v, 75) if v else np.nan for v in vals[s]])
    ax.plot(x, med, color=CO[s], lw=1.4, alpha=0.55, zorder=3)
    ax.errorbar(x, med, yerr=[med - q1, q3 - med], fmt="o", ms=6.5,
                color=CO[s], ecolor=CO[s], elinewidth=1.4, capsize=3,
                zorder=4, label=s)
    for xi, mi in zip(x, med):
        if np.isfinite(mi):
            ax.annotate(f"×{mi:.2f}", (xi, mi), textcoords="offset points",
                        xytext=(0, 8 if s == "THI" else -14), fontsize=7.5,
                        color=CO[s], ha="center", fontweight="bold")
    n = [len(v) for v in vals[s]]
    print(s, "n per regime:", n, "median:", np.round(med, 2))
ax.text(2.44, 1.0, "truth", fontsize=8, color=INK, va="center")
ax.set_xticks(range(len(REG)))
ax.set_xticklabels([t for t, _ in REG], fontsize=9, color="#3f454c")
ax.set_yticks([0.25, 0.5, 1, 2])
ax.set_yticklabels(["0.25×", "0.5×", "1×", "2×"])
ax.set_ylabel("surface enrichment E (share vs truth)", fontsize=9,
              color="#5a6067")
ax.tick_params(labelsize=8, colors="#8a919b", length=2.5)
for k in ("top", "right"):
    ax.spines[k].set_visible(False)
for k in ("left", "bottom"):
    ax.spines[k].set_color("#b6bcc4")
ax.legend(frameon=False, fontsize=8.5, loc="lower left")
ax.set_xlim(-0.5, 2.7)
fig.tight_layout()
fig.savefig(os.path.join(RES, "61_regime_summary.png"), dpi=400,
            bbox_inches="tight", facecolor="white")
print("saved 61_regime_summary.png")
