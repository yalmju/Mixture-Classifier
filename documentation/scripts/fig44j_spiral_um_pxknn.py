# -*- coding: utf-8 -*-
"""44j — µM 복원 스파이럴, pixel k-NN LOO ("픽셀=액적") 픽셀 구름판.

44i(맵당 점 1개)가 휑하다는 지적에 따라, 배포 사이드카 .pxknn.npz의
라이브러리 픽셀 자체를 질의로 써서 픽셀 단위 LOO(자기 조건 픽셀 제외,
k=15 log-공간 기하 가중평균 — page_real._pxknn_lookup과 동일 로직)를 돌린다.
저채도 구름 = 픽셀별 판독 배율(pred/true), 고채도 점 = 맵 중앙값.
각도 = 그 성분 참 µM 순위. 의심맵 3장은 질의·이웃 모두에서 제외(라벨 불신),
imbalance(100x)는 질의만 제외.

실행:  python -u fig44j_spiral_um_pxknn.py
"""
import os
import re
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
NPZ = r"S:\Google Drive\내 드라이브\ACF_PEST_DB\260831_Model_FINAL\mlp_composition_260831_final.pxknn.npz"
SUBS = ["DQ", "TBZ", "THI"]
CO = {s: labfig.CO[s] for s in SUBS}
WINDOW = "#eab53a"
RING = "#c8930f"
RNG = np.random.default_rng(7)
PX_PER_COND = 40
SUSPECT = {"DQ250-TB0-TH50.csv", "DQ250-TB50-TH0.csv", "DQ500-TB50-TH0.csv"}
_EPS = 0.25
K = 15

RMAX = 2.0
PAD = 0.30
GRADE = [(1.25, 0.45), (1.5, 0.62), (1.75, 0.76), (2.0, 0.88)]


def mute(c, f=0.62):
    r, g, b = to_rgb(c)
    gr = 0.85
    return (r + (gr - r) * f, g + (gr - g) * f, b + (gr - b) * f)


def _band_color(fw):
    r, g, b = to_rgb(WINDOW)
    return (r + (1 - r) * fw, g + (1 - g) * fw, b + (1 - b) * fw)


def rad(x):
    return np.clip(np.log2(np.clip(x, 1e-3, None)), -RMAX, RMAX) + RMAX


lib = np.load(NPZ, allow_pickle=True)
Fz = np.asarray(lib["Fz"], float)
Y = np.asarray(lib["Y"], float)
cond = np.asarray(lib["cond"]).astype(str)
keep = ~np.isin(cond, list(SUSPECT))       # 의심맵: 이웃에서도 제거
Fz, Y, cond = Fz[keep], Y[keep], cond[keep]
print("library px:", len(Fz))

# 질의 = 혼합물 맵 픽셀 (imbalance 100x 제외)
is_mix = np.array([bool(re.match(r"DQ[\d.]+-TB[\d.]+-TH[\d.]+", c))
                   for c in cond])
imb = np.zeros(len(cond), bool)
for i, y in enumerate(Y):
    pos = y[y > 0]
    if pos.size and y.max() / pos.min() >= 99:
        imb[i] = True
query = is_mix & ~imb

# 픽셀 LOO: 자기 조건 픽셀 제외, k=15 log-공간 기하 가중평균
P = np.zeros((query.sum(), 3))
qidx = np.where(query)[0]
for j, i in enumerate(qidx):
    d = np.linalg.norm(Fz - Fz[i][None, :], axis=1)
    d[cond == cond[i]] = np.inf
    idx = np.argpartition(d, K)[:K]
    w = 1.0 / np.maximum(d[idx], 1e-9)
    w = w / w.sum()
    P[j] = np.exp((w[:, None] * np.log(Y[idx] + _EPS)).sum(0)) - _EPS
P = np.clip(P, 0, None)

# 조건별 픽셀 fold 수집
data = {s: {} for s in SUBS}
for j, i in enumerate(qidx):
    y = Y[i]
    for k, s in enumerate(SUBS):
        if y[k] > 0:
            data[s].setdefault((float(y[k]), cond[i]), []).append(
                P[j, k] / y[k])

for s in SUBS:
    allv = np.concatenate([np.asarray(v) for v in data[s].values()])
    w2 = ((allv >= 0.5) & (allv <= 2.0)).mean() * 100
    print(f"{s}: {len(data[s])} maps, {len(allv)} px, pixel within-2x={w2:.0f}%")


def draw_sp(s, ax):
    recs = sorted(data[s].items())       # 각도 = 참 µM 순위
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
        for x in (f, 1 / f):
            ax.plot(th, np.full_like(th, rad(x)), color="white", lw=0.7,
                    alpha=0.85, zorder=1)
    jit = np.deg2rad(340) / n * 0.32
    meds = []
    for a, (_, v) in zip(angs, recs):
        v = np.asarray(v, float)
        sub = v if len(v) <= PX_PER_COND else RNG.choice(
            v, PX_PER_COND, replace=False)
        ax.scatter(a + RNG.uniform(-jit, jit, len(sub)),
                   np.clip(rad(sub), 0.05, 2 * RMAX + 0.12), s=2.6,
                   color=mute(CO[s]), alpha=0.5, edgecolor="none", zorder=3)
        meds.append(np.median(v))
    ax.scatter(angs, np.clip(rad(np.asarray(meds)), 0.10, 2 * RMAX - 0.06),
               s=17, color=CO[s], alpha=0.95, edgecolor="white",
               linewidth=0.3, zorder=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 2 * RMAX + PAD)
    ax.spines["polar"].set_visible(False)


for s in ("THI", "TBZ", "DQ"):
    fig, ax = plt.subplots(figsize=(4.9, 4.9), subplot_kw=dict(polar=True))
    draw_sp(s, ax)
    fig.savefig(os.path.join(RES, f"44j_spiral_um_pxknn_notext_{s}.png"),
                dpi=300, transparent=True, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
fig, axs = plt.subplots(1, 3, figsize=(13.8, 4.7), subplot_kw=dict(polar=True))
for ax, s in zip(axs, ("THI", "TBZ", "DQ")):
    draw_sp(s, ax)
fig.subplots_adjust(wspace=0.08)
fig.savefig(os.path.join(RES, "44j_spiral_um_pxknn_strip.png"), dpi=300,
            transparent=True, bbox_inches="tight", pad_inches=0.02)
print("saved 44j x4")
