# -*- coding: utf-8 -*-
"""62 — 원형 링 히트맵 (circos풍) 프로토타입.

부채꼴 = 조건(참 THI 분율 오름차순, 12시 틈), 링 = 안쪽 3개 표면 E
(NNLS/true: DQ·TBZ·THI), 바깥 3개 복원 E (MLP held-out/true).
색 = log2 배율 다이버징(보라 과소 · 흰 정답 · 주황 과대, ±2 클립),
부재 성분 = 회색. 림 라벨: 복원이 2× 밖인 조건은 빨강.
의심맵 3장·imbalance 제외. 색은 값 인코딩이라 colors.json과 무관.

실행:  python -u fig62_circular_rings.py
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import labfig  # noqa: E402  (스타일 셋업)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

RES = os.path.join(HERE, "..", "results")
SUBS = ["DQ", "TBZ", "THI"]
SUSPECT = {(250.0, 0.0, 50.0), (250.0, 50.0, 0.0), (500.0, 50.0, 0.0)}

rows = [r for r in csv.DictReader(
    open(os.path.join(RES, "17_composition_all_conditions_master.csv"),
         encoding="utf-8-sig")) if r["imbalance_100x"] == "0"]
rows = [r for r in rows
        if (float(r["DQ"]), float(r["TBZ"]), float(r["THI"])) not in SUSPECT]

recs = []
for r in rows:
    dq, tb, th = float(r["DQ"]), float(r["TBZ"]), float(r["THI"])
    tot = dq + tb + th
    e0, e1 = [], []
    for s in SUBS:
        t = float(r[f"Ratio_{s}_True"])
        if t <= 0:
            e0.append(np.nan); e1.append(np.nan)
        else:
            e0.append(np.log2(max(float(r[f"Ratio_{s}_NNLS_Pred"]) / t,
                                  1e-3)))
            e1.append(np.log2(max(float(r[f"Ratio_{s}_MLP_heldout"]) / t,
                                  1e-3)))
    recs.append((th / tot * 100 if tot else 0,
                 f"DQ{dq:g}-TB{tb:g}-TH{th:g}", e0, e1))
recs.sort(key=lambda x: x[0])
n = len(recs)
M = np.full((6, n), np.nan)     # 링 0..2 = 표면, 3..5 = 복원
for j, (_, _, e0, e1) in enumerate(recs):
    M[0:3, j] = e0
    M[3:6, j] = e1
labels = [t[1] for t in recs]
bad = [bool(np.nanmax(np.abs(t[3])) > 1) for t in recs]   # 복원 2× 밖

# 다이버징: 보라(과소) — 흰(정답) — 주황(과대), 레퍼런스 그림 톤
cmap = LinearSegmentedColormap.from_list(
    "pw", ["#5c5a9e", "#a9a7cf", "#f4f2ee", "#f3c977", "#e2952e"])
cmap.set_bad("#d9dde2")
norm = Normalize(-2, 2)

A0 = np.deg2rad(96)             # 12시 근처 틈(12°)
SPAN = np.deg2rad(348)
edges = A0 + SPAN * np.arange(n + 1) / n
R0, DR, GAP = 1.0, 0.28, 0.10   # 안쪽 반지름, 링 두께, 표면/복원 그룹 틈

fig, ax = plt.subplots(figsize=(8.6, 8.6), subplot_kw=dict(polar=True))
for ring in range(6):
    r_in = R0 + ring * DR + (GAP if ring >= 3 else 0)
    for j in range(n):
        v = M[ring, j]
        col = cmap(norm(v)) if np.isfinite(v) else "#d9dde2"
        ax.bar((edges[j] + edges[j + 1]) / 2, DR * 0.92, bottom=r_in,
               width=(edges[1] - edges[0]) * 0.96, color=col,
               edgecolor="white", linewidth=0.25, zorder=2)
# 그룹 라벨은 중앙 구멍에 (링 위 텍스트 충돌 방지)
ax.text(np.deg2rad(90), 0.0, "surface → restored\n(inner → outer)",
        fontsize=9, color="#3f454c", ha="center", va="center", zorder=4)
for k, s in enumerate(SUBS):
    for base in (R0, R0 + GAP + 3 * DR):
        ax.text(np.deg2rad(92.5), base + (k + 0.5) * DR, s,
                fontsize=5.2, color="#6a7178", ha="center", va="center",
                zorder=4)
# 림 라벨: 복원이 2x 밖인 조건만 빨강으로 (레퍼런스처럼 소수 강조)
r_lab = R0 + GAP + 6 * DR + 0.10
for j in range(n):
    if not bad[j]:
        continue
    a = (edges[j] + edges[j + 1]) / 2
    deg = np.rad2deg(a) % 360
    rot = deg - 90 if 0 <= deg <= 180 else deg + 90
    ax.text(a, r_lab, labels[j], fontsize=4.6, color="#c0392b",
            fontweight="bold",
            ha="left" if 0 <= deg <= 180 else "right",
            va="center", rotation=rot, rotation_mode="anchor")
ax.text(np.deg2rad(90), r_lab + 0.65, "true THI share →", fontsize=7,
        color="#8a919b", ha="center")
ax.set_xticks([]); ax.set_yticks([])
ax.set_ylim(0, r_lab + 0.75)
ax.spines["polar"].set_visible(False)
# 컬러바 (log2 fold)
cax = fig.add_axes([0.90, 0.80, 0.015, 0.13])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_ticks([-2, -1, 0, 1, 2])
cb.set_ticklabels(["¼×", "½×", "1×", "2×", "4×"])
cax.tick_params(labelsize=6, length=2)
cb.outline.set_linewidth(0.4)
cax.set_title("vs truth", fontsize=6, color="#3f454c", pad=3)
fig.savefig(os.path.join(RES, "62_circular_rings.png"), dpi=400,
            bbox_inches="tight", facecolor="white")
print(f"saved 62_circular_rings.png ({n} conditions, red rim = restored"
      f" outside 2x: {sum(bad)})")
