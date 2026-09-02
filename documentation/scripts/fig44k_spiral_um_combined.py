# -*- coding: utf-8 -*-
"""44k — 44j의 THI·TBZ·DQ 세 채널을 한 스파이럴에 통합.

44i(맵당 점 1개)가 휑하다는 지적에 따라, 배포 사이드카 .pxknn.npz의
라이브러리 픽셀 자체를 질의로 써서 픽셀 단위 LOO(자기 조건 픽셀 제외,
k=15 log-공간 기하 가중평균 — page_real._pxknn_lookup과 동일 로직)를 돌린다.
저채도 구름 = 픽셀별 판독 배율(pred/true), 고채도 점 = 맵 중앙값.
각도 = 그 성분 참 µM 순위. 의심맵 3장은 질의·이웃 모두에서 제외(라벨 불신),
imbalance(100x)는 질의만 제외.

실행:  python -u fig44k_spiral_um_combined.py
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

# 조건(맵)별 픽셀 fold 수집 — 세 채널이 같은 맵 키를 공유
data = {s: {} for s in SUBS}
for j, i in enumerate(qidx):
    y = Y[i]
    for k, s in enumerate(SUBS):
        if y[k] > 0:
            data[s].setdefault(cond[i], []).append(P[j, k] / y[k])

for s in SUBS:
    allv = np.concatenate([np.asarray(v) for v in data[s].values()])
    w2 = ((allv >= 0.5) & (allv <= 2.0)).mean() * 100
    print(f"{s}: {len(data[s])} maps, {len(allv)} px, pixel within-2x={w2:.0f}%")

# 맵 공통 배열축: NNLS의 THI 과대판독 Δ = 표면 THI% − 참 THI% (오름차순)
# — 틈에서 출발해 돌수록 "THI가 비대로 읽힌" 맵. (사용자 지정 기준)
import csv as _csv

delta = {}
for r in _csv.DictReader(open(os.path.join(
        RES, "17_composition_all_conditions_master.csv"),
        encoding="utf-8-sig")):
    name = (f"DQ{float(r['DQ']):g}-TB{float(r['TBZ']):g}"
            f"-TH{float(r['THI']):g}.csv")
    delta[name] = (float(r["Ratio_THI_NNLS_Pred"])
                   - float(r["Ratio_THI_True"]))
maps = sorted({c for s in SUBS for c in data[s]},
              key=lambda c: delta.get(c, 0.0))
# 각도 = Δ 값에 선형 비례 (순위 아님 — "점 400개 = 축 400개" 방지)
D0 = delta.get(maps[0], 0.0)
D1 = delta.get(maps[-1], 0.0)


def d2ang(dv):
    return np.deg2rad(100) + (dv - D0) / (D1 - D0) * np.deg2rad(340)


ANG = {c: float(d2ang(delta.get(c, 0.0))) for c in maps}
print("maps on shared axis:", len(maps),
      "| delta range:", round(D0, 1), "→", round(D1, 1), "%p")


def draw_window(ax):
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


def draw_sub(s, ax, clouds=True):
    jit = np.deg2rad(340) / len(maps) * 0.32
    angs, meds = [], []
    for c, v in data[s].items():
        a = ANG[c]
        v = np.asarray(v, float)
        if clouds:
            sub = v if len(v) <= PX_PER_COND else RNG.choice(
                v, PX_PER_COND, replace=False)
            ax.scatter(a + RNG.uniform(-jit, jit, len(sub)),
                       np.clip(rad(sub), 0.05, 2 * RMAX + 0.12), s=2.4,
                       color=mute(CO[s]), alpha=0.38, edgecolor="none",
                       zorder=3)
        angs.append(a)
        meds.append(np.median(v))
    ax.scatter(angs, np.clip(rad(np.asarray(meds)), 0.10, 2 * RMAX - 0.06),
               s=19, color=CO[s], alpha=0.95, edgecolor="white",
               linewidth=0.4, zorder=5)


def draw_delta_grid(ax):
    """Δ 눈금 격자 — 선형축이라 등간격."""
    for dv in (0, 10, 20, 30, 40, 50, 60):
        if dv < D0 or dv > D1:
            continue
        a = float(d2ang(dv))
        ax.plot([a, a], [rad(1 / 2.2), 2 * RMAX + 0.06], color="#d9dde2",
                lw=0.7, zorder=0.5)
        ax.plot([a, a], [2 * RMAX + 0.06, 2 * RMAX + 0.14], color="#b6bcc4",
                lw=1.0, clip_on=False, zorder=1)
        lab = "Δ 0" if dv == 0 else f"+{dv}"
        ax.text(a, 2 * RMAX + 0.34, lab, fontsize=7.5, color="#8a919b",
                ha="center", va="center")
    ax.text(np.deg2rad(100 + 340 * 0.5), 2 * RMAX + 0.62,
            "NNLS THI over-read (%p) →", fontsize=8, color="#8a919b",
            ha="center")


for tag, clouds in (("44k_spiral_um_combined", True),
                    ("44k_spiral_um_combined_medians", False)):
    fig, ax = plt.subplots(figsize=(7.0, 7.0), subplot_kw=dict(polar=True))
    draw_window(ax)
    draw_delta_grid(ax)
    for s in ("DQ", "TBZ", "THI"):
        draw_sub(s, ax, clouds=clouds)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-0.9, 2 * RMAX + PAD)
    ax.spines["polar"].set_visible(False)
    fig.savefig(os.path.join(RES, f"{tag}.png"), dpi=400,
                transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
print("saved 44k x2")
