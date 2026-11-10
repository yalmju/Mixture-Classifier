# -*- coding: utf-8 -*-
"""44i — 농도(µM) 복원 스파이럴, k-NN 경로.

"농도는 측정한 것(라이브러리)으로만 답한다" 원칙 그대로:
점 = map k-NN LOO 판독의 µM 배율(pred/true), 거부(d>3)·부재 성분은 점 없음.
그라데이션 밴드(±1.25/1.5/1.75/2×)가 곧 within-2× 기준. 문법은 44f/g와 동일.

데이터: knn_um_readout_20260901/loo.json (dmin, pred, true).
의심맵 3장·imbalance(100x) 제외.

실행:  python -u fig44i_spiral_umknn.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import labfig  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

RES = os.path.join(HERE, "..", "results")
SUBS = ["DQ", "TBZ", "THI"]
CO = {s: labfig.CO[s] for s in SUBS}
WINDOW = "#eab53a"
RING = "#c8930f"
D_MAX = 3.0
SUSPECT = {"DQ250-TB0-TH50.csv", "DQ250-TB50-TH0.csv", "DQ500-TB50-TH0.csv"}

RMAX = 2.0
PAD = 0.30
GRADE = [(1.25, 0.45), (1.5, 0.62), (1.75, 0.76), (2.0, 0.88)]


def _band_color(fw):
    r, g, b = to_rgb(WINDOW)
    return (r + (1 - r) * fw, g + (1 - g) * fw, b + (1 - b) * fw)


def rad(x):
    return np.clip(np.log2(np.clip(x, 1e-3, None)), -RMAX, RMAX) + RMAX


loo = json.load(open(os.path.join(RES, "knn_um_readout_20260901", "loo.json"),
                     encoding="utf-8"))
folds = {s: [] for s in SUBS}
refused = 0
for e in loo:
    if e["map"] in SUSPECT:
        continue
    t = np.asarray(e["true"], float)
    if (t > 0).any() and t.max() / max(t[t > 0].min(), 1e-9) >= 99:
        continue  # imbalance 100x
    if float(e.get("dmin", 0)) > D_MAX:
        refused += 1
        continue  # 무응답 — 점 없음
    p = np.asarray(e["pred"], float)
    for k, s in enumerate(SUBS):
        if t[k] > 0:
            folds[s].append((t[k], p[k] / t[k]))
print("refused (d>3):", refused)


def draw_sp(s, ax):
    recs = sorted(folds[s])          # 각도 = 참 µM 순위 (안쪽 시작 = 저농도)
    v = np.asarray([f for _, f in recs], float)
    n = len(v)
    angs = np.deg2rad(100) + np.linspace(0, np.deg2rad(340), n)
    th = np.linspace(0, 2 * np.pi, 240)
    lo = 1.0
    for f, fw in GRADE:
        c = _band_color(fw)
        ax.fill_between(th, rad(lo), rad(f), color=c, lw=0, zorder=0)
        ax.fill_between(th, rad(1 / f), rad(1 / lo), color=c, lw=0, zorder=0)
        lo = f
    ax.plot(th, np.full_like(th, RMAX), color=RING, lw=2.2, zorder=2)
    for f, _ in GRADE:
        for x in (f, 1 / f):
            ax.plot(th, np.full_like(th, rad(x)), color="white", lw=0.7,
                    alpha=0.85, zorder=1)
    ax.scatter(angs, np.clip(rad(v), 0.10, 2 * RMAX - 0.06), s=17,
               color=CO[s], alpha=0.95, edgecolor="white", linewidth=0.3,
               zorder=5)
    w2 = float(((v >= 0.5) & (v <= 2.0)).mean() * 100)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 2 * RMAX + PAD)
    ax.spines["polar"].set_visible(False)
    print(f"{s}: n={n} within-2x={w2:.0f}% median fold={np.median(v):.2f}")
    return n


for s in ("THI", "TBZ", "DQ"):
    fig, ax = plt.subplots(figsize=(4.9, 4.9), subplot_kw=dict(polar=True))
    draw_sp(s, ax)
    fig.savefig(os.path.join(RES, f"44i_spiral_umknn_notext_{s}.png"),
                dpi=300, transparent=True, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
fig, axs = plt.subplots(1, 3, figsize=(13.8, 4.7), subplot_kw=dict(polar=True))
for ax, s in zip(axs, ("THI", "TBZ", "DQ")):
    draw_sp(s, ax)
fig.subplots_adjust(wspace=0.08)
fig.savefig(os.path.join(RES, "44i_spiral_umknn_strip.png"), dpi=300,
            transparent=True, bbox_inches="tight", pad_inches=0.02)
print("saved 44i x4")
