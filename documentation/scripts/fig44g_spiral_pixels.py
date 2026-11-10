# -*- coding: utf-8 -*-
"""44g — guided spiral, 표면을 SERS 맵 픽셀에서 직접.

44f와 문법 동일(노랑 그라데이션 창 + 앰버 과녁, 각도 = 표면 E 순위)하되,
조건당 흰 점 1개(맵 평균) 대신 그 맵의 게이트 통과 픽셀 E 분포를
저채도 점구름으로 뿌린다. 복원(고채도 점)은 맵 단위 MLP held-out.
점 = raw 픽셀(저채도) / 복원 = 고채도 — 확정 문법.

데이터: 61b_pixel_E_raw.csv (fig61b_pixel_regime.py가 맵 unmix로 생성,
의심맵·imbalance 제외) + 17 master(복원·정렬).

실행:  python -u fig44g_spiral_pixels.py
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
from matplotlib.colors import to_rgb

RES = os.path.join(HERE, "..", "results")
MUTE_EDGE = "#a6acb5"
WINDOW = "#eab53a"
RING = "#c8930f"
CO = {s: labfig.CO[s] for s in ("DQ", "TBZ", "THI")}
RNG = np.random.default_rng(7)
PX_PER_COND = 40  # 조건당 표시할 픽셀 수(서브샘플) — 전체는 CSV에 있다


def mute(c, f=0.62):
    r, g, b = to_rgb(c)
    gr = 0.85
    return (r + (gr - r) * f, g + (gr - g) * f, b + (gr - b) * f)


RMAX = 2.0
PAD = 0.30
GRADE = [(1.25, 0.45), (1.5, 0.62), (1.75, 0.76), (2.0, 0.88)]


def _band_color(fw):
    r, g, b = to_rgb(WINDOW)
    return (r + (1 - r) * fw, g + (1 - g) * fw, b + (1 - b) * fw)


def rad(x):
    return np.clip(np.log2(np.clip(x, 1e-3, None)), -RMAX, RMAX) + RMAX


# 픽셀 E (맵에서 추출된 것)
pix = {}
for r in csv.DictReader(open(os.path.join(RES, "61b_pixel_E_raw.csv"),
                             encoding="utf-8-sig")):
    pix.setdefault((r["condition"], r["substance"]), []).append(
        float(r["E_pixel"]))

# 복원(맵 단위 MLP held-out) — master에서
SUSPECT = {(250.0, 0.0, 50.0), (250.0, 50.0, 0.0), (500.0, 50.0, 0.0)}
master = [r for r in csv.DictReader(
    open(os.path.join(RES, "17_composition_all_conditions_master.csv"),
         encoding="utf-8-sig")) if r["imbalance_100x"] == "0"]
master = [r for r in master
          if (float(r["DQ"]), float(r["TBZ"]), float(r["THI"])) not in SUSPECT]


def draw_sp(s, ax):
    recs = []
    for r in master:
        t = float(r[f"Ratio_{s}_True"])
        if t <= 0:
            continue
        name = (f"DQ{float(r['DQ']):g}-TB{float(r['TBZ']):g}"
                f"-TH{float(r['THI']):g}")
        px = pix.get((name, s))
        if not px:
            continue
        recs.append((np.asarray(px, float),
                     float(r[f"Ratio_{s}_MLP_heldout"]) / t))
    recs.sort(key=lambda x: np.median(x[0]))
    n = len(recs)
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
        for v in (f, 1 / f):
            ax.plot(th, np.full_like(th, rad(v)), color="white", lw=0.7,
                    alpha=0.85, zorder=1)
    jit = np.deg2rad(340) / n * 0.32
    for a, (px, _) in zip(angs, recs):
        sub = px if len(px) <= PX_PER_COND else RNG.choice(
            px, PX_PER_COND, replace=False)
        ax.scatter(a + RNG.uniform(-jit, jit, len(sub)),
                   np.clip(rad(sub), 0.05, 2 * RMAX + 0.12), s=2.6,
                   color=mute(CO[s]), alpha=0.5, edgecolor="none", zorder=3)
    rm = np.clip([rad(b) for _, b in recs], 0.10, 2 * RMAX - 0.06)
    ax.scatter(angs, rm, s=17, color=CO[s], alpha=0.95, edgecolor="white",
               linewidth=0.3, zorder=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 2 * RMAX + PAD)
    ax.spines["polar"].set_visible(False)
    return n


for s in ("THI", "TBZ", "DQ"):
    fig, ax = plt.subplots(figsize=(4.9, 4.9), subplot_kw=dict(polar=True))
    n = draw_sp(s, ax)
    fig.savefig(os.path.join(RES, f"44g_spiral_pixels_notext_{s}.png"),
                dpi=300, transparent=True, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    print(s, "conditions:", n)
fig, axs = plt.subplots(1, 3, figsize=(13.8, 4.7), subplot_kw=dict(polar=True))
for ax, s in zip(axs, ("THI", "TBZ", "DQ")):
    draw_sp(s, ax)
fig.subplots_adjust(wspace=0.08)
fig.savefig(os.path.join(RES, "44g_spiral_pixels_strip.png"), dpi=300,
            transparent=True, bbox_inches="tight", pad_inches=0.02)
print("saved 44g x4")
