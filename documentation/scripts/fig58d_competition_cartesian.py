# -*- coding: utf-8 -*-
"""58d — 경쟁과 복원, 데카르트 최종형 (스파이럴 배제).

x = 용액의 참 THI 조성(%), y = 참값 대비 배율(log2).
점(저채도) = raw NNLS 표면 = 경쟁흡착의 왜곡,
굵은 곡선(고채도) = MLP held-out 복원 running median.
데이터: documentation/results/17_composition_all_conditions_master.csv
(92조건, imbalance 100x 제외; Ratio_* 열은 % 단위 → 행 정규화).

실행:  python -u fig58d_competition_cartesian.py
"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import labfig  # noqa: E402  — 색 단일 출처 = Pure/colors.json

RES = os.path.join(HERE, "..", "results")
CO = {s: labfig.CO[s] for s in ("DQ", "TBZ", "THI")}
INK = "black"


def mute(c, f=0.68):
    r, g, b = to_rgb(c)
    gr = 0.82
    return (r + (gr - r) * f, g + (gr - g) * f, b + (gr - b) * f)


rows = [r for r in csv.DictReader(
    open(os.path.join(RES, "17_composition_all_conditions_master.csv"),
         encoding="utf-8-sig")) if r["imbalance_100x"] == "0"]

# 의심맵 3장 (전용 밴드 기준 라벨과 어긋남 — 07-24/07-27) 제외:
# 정답 라벨을 신뢰할 수 없는 맵은 truth 기반 그림에 못 쓴다.
SUSPECT = {(250.0, 0.0, 50.0), (250.0, 50.0, 0.0), (500.0, 50.0, 0.0)}
rows = [r for r in rows
        if (float(r["DQ"]), float(r["TBZ"]), float(r["THI"])) not in SUSPECT]

data = {s: {"x": [], "e0": [], "e1": []} for s in CO}
for r in rows:
    dq, tb, thv = float(r["DQ"]), float(r["TBZ"]), float(r["THI"])
    tot = dq + tb + thv
    xs = thv / tot * 100 if tot else 0
    for s in ("DQ", "TBZ", "THI"):
        t = float(r[f"Ratio_{s}_True"])
        if t <= 0:
            continue
        data[s]["x"].append(xs)
        data[s]["e0"].append(float(r[f"Ratio_{s}_NNLS_Pred"]) / t)
        data[s]["e1"].append(float(r[f"Ratio_{s}_MLP_heldout"]) / t)
Er = np.concatenate([np.array(data[s]["e1"]) for s in CO])
q1, q2, q3 = np.percentile(Er, [25, 50, 75])


def runmed(x, e, win=9):
    o = np.argsort(x)
    x, e = x[o], e[o]
    xs = np.linspace(2, 98, 72)
    ys = np.array([np.median(e[np.abs(x - c) <= win])
                   if (np.abs(x - c) <= win).sum() >= 3 else np.nan
                   for c in xs])
    ok = ~np.isnan(ys)
    return xs[ok], ys[ok]


fig, ax = plt.subplots(figsize=(7.0, 4.3))
ax.set_yscale("log", base=2)
ax.axhspan(q1, q3, color="#7d848c", alpha=0.13, zorder=1)
ax.axhline(1.0, color="#9aa3ad", lw=1.3, ls=(0, (4, 3)), zorder=3)
for f in (0.5, 2.0):
    ax.axhline(f, color="0.84", lw=0.6, ls=":", zorder=1)
for s in ("DQ", "TBZ", "THI"):
    x = np.array(data[s]["x"])
    e0 = np.array(data[s]["e0"])
    ax.scatter(x, np.clip(e0, 0.16, None), s=13, color=mute(CO[s], 0.45),
               alpha=0.8, edgecolor="none", zorder=2)
for s in ("DQ", "TBZ", "THI"):
    xs, ys = runmed(np.array(data[s]["x"]), np.array(data[s]["e1"]))
    ax.plot(xs, ys, color=CO[s], lw=3.6, alpha=0.95, zorder=5,
            solid_capstyle="round")
ax.text(11, 2.55, "surface: THI wins", fontsize=9.5,
        color=mute(CO["THI"], 0.35), fontweight="bold", ha="left")
ax.text(72, 0.30, "surface: DQ·TBZ lose", fontsize=9,
        color=mute(CO["DQ"], 0.35), ha="left")
ax.text(97, 1.24, "restored", fontsize=9, color=INK, fontweight="bold",
        ha="right")
ax.text(99.2, 1.0, "truth", fontsize=8, color="#9aa3ad", va="center", ha="left",
        clip_on=False)
ax.set_xlim(0, 100)
ax.set_ylim(0.15, 4.6)
ax.set_yticks([0.25, 0.5, 1, 2, 4])
ax.set_yticklabels(["0.25×", "0.5×", "1×", "2×", "4×"])
ax.set_xlabel("true THI share of solution (%)", fontsize=9, color="#5a6067")
ax.set_ylabel("share vs truth (fold)", fontsize=9, color="#5a6067")
ax.tick_params(labelsize=8, colors="#8a919b", length=2.5)
for k in ("top", "right"):
    ax.spines[k].set_visible(False)
for k in ("left", "bottom"):
    ax.spines[k].set_color("#b6bcc4")
fig.tight_layout()
fig.savefig(os.path.join(RES, "58d_competition_cartesian.png"), dpi=400,
            bbox_inches="tight", facecolor="white")
print("saved 58d")
