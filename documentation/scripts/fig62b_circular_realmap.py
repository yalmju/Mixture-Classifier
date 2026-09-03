# -*- coding: utf-8 -*-
"""62b — 글씨맵(12:12:12) 픽셀별 recovery를 이산 등급색 circos로.

부채꼴 = 게이트 통과 픽셀(행 우선 스캔 — 연속 구간 = 글씨 획).
셀 값 = 픽셀 조성 ÷ 제조 진실(1/3) recovery. 이산 등급(62와 동일):
80–125% 흰색(합격) / 125–150 / 150–200 / >200 주황 3단,
67–80 / 50–67 / <50 보라 3단. 흰 링 = 그 픽셀 복원 성공.

두 버전 저장:
  62b_realmap_recovered.png  recovered 3링만
  62b_realmap_both.png       surface(A_evidence) 3링 + recovered 3링

실행:  python -u fig62b_circular_realmap.py
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import labfig  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import load_preprocess  # noqa: E402
from dl_model import load_model, apply_model_pixels  # noqa: E402
from unmix import unmix_map  # noqa: E402

DLM = r"S:\Google Drive\내 드라이브\ACF_PEST_DB\260831_Model_FINAL\mlp_composition_260831_final.dlm"
PURE = r"S:\Google Drive\내 드라이브\ACF_PEST_DB\Pure"
MAP = r"S:\Google Drive\내 드라이브\ACF_PEST_DB\Pest\260812_12 trio_THI_TBZ_DQ.csv"
RES = os.path.join(ROOT, "documentation", "results")
SUBS = ["DQ", "TBZ", "THI"]
TRUE = 1.0 / 3.0

OVER = ["#f4f2ee", "#f3c977", "#e2952e", "#a85f0d"]
UNDER = ["#f4f2ee", "#b9b7d9", "#726fb0", "#403d85"]
_EDGES = [np.log2(1.25), np.log2(1.5), 1.0]


def classify(v):
    if not np.isfinite(v):
        return "#d9dde2"
    a = abs(v)
    k = (0 if a <= _EDGES[0] else 1 if a <= _EDGES[1]
         else 2 if a <= _EDGES[2] else 3)
    return (OVER if v > 0 else UNDER)[k]


m = load_model(DLM)
cfg = load_preprocess(PURE)
r = unmix_map(data_dir=PURE, test_path=MAP, method="dlpx",
              baseline=bool(m.get("baseline", cfg["baseline"])),
              trim=cfg["trim"], min_frac=0.15, hit_mode="threshold",
              dl_model=m)
pk = np.clip(np.asarray(apply_model_pixels(m, r.wn, r.spectra), float), 0,
             None)
cols = [list(m["subs"]).index(s) for s in SUBS]
Rres = pk[:, cols] / (pk[:, cols].sum(1, keepdims=True) + 1e-12)
ev = np.asarray(getattr(r, "A_evidence", r.A), float)[:, r.nonbg]
Rsurf = ev / (ev.sum(1, keepdims=True) + 1e-12)
sel = np.where(np.asarray(r.hit, bool))[0]
xy = np.asarray(r.coords, float)
sel = sel[np.lexsort((xy[sel, 0], xy[sel, 1]))]
n = len(sel)
print("positive pixels:", n)

Lres = np.log2(np.clip(Rres[sel].T / TRUE, 1e-3, None))   # (3, n)
Lsurf = np.log2(np.clip(Rsurf[sel].T / TRUE, 1e-3, None))
ok_frac = [(np.abs(Lres[k]) <= _EDGES[0]).mean() * 100 for k in range(3)]
print("recovered ok(80-125%) fraction:",
      {s: f"{f:.0f}%" for s, f in zip(SUBS, ok_frac)})

A0 = np.deg2rad(96)
SPAN = np.deg2rad(348)
edges = A0 + SPAN * np.arange(n + 1) / n
th_c = (edges[:-1] + edges[1:]) / 2
w_c = edges[1] - edges[0]


def legend(fig):
    lax = fig.add_axes([0.875, 0.70, 0.11, 0.24])
    lax.set_axis_off()
    items = [(OVER[3], "> 200%"), (OVER[2], "150–200"),
             (OVER[1], "125–150"), (OVER[0], "80–125 (ok)"),
             (UNDER[1], "67–80"), (UNDER[2], "50–67"), (UNDER[3], "< 50%")]
    lax.text(0.02, 1.02, "recovery vs 1/3", fontsize=6.5, color="#3f454c",
             weight="bold")
    for i, (c, t) in enumerate(items):
        y = 0.92 - i * 0.13
        lax.add_patch(plt.Rectangle((0.02, y - 0.05), 0.16, 0.10,
                                    facecolor=c, edgecolor="#b6bcc4",
                                    linewidth=0.3))
        lax.text(0.24, y, t, fontsize=5.8, color="#5a6067", va="center")


def draw(groups, tag, center):
    R0, DR, GAP = 1.0, (0.42 if len(groups) == 1 else 0.30), 0.10
    fig, ax = plt.subplots(figsize=(8.6, 8.6), subplot_kw=dict(polar=True))
    for g, L in enumerate(groups):
        for k in range(3):
            r_in = R0 + (g * 3 + k) * DR + (GAP if g else 0)
            ax.bar(th_c, np.full(n, DR * 0.94), bottom=r_in, width=w_c,
                   color=[classify(v) for v in L[k]], edgecolor="none",
                   zorder=2)
            ax.text(np.deg2rad(92.5), r_in + 0.5 * DR, SUBS[k],
                    fontsize=5.6, color="#6a7178", ha="center", va="center",
                    zorder=4)
    r_out = R0 + len(groups) * 3 * DR + (GAP if len(groups) > 1 else 0)
    ax.text(np.deg2rad(90), 0.0, center, fontsize=9, color="#3f454c",
            ha="center", va="center", zorder=4)
    ax.text(np.deg2rad(90), r_out + 0.42,
            f"{n} gated pixels · scan order →", fontsize=7,
            color="#8a919b", ha="center")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, r_out + 0.55)
    ax.spines["polar"].set_visible(False)
    legend(fig)
    fig.savefig(os.path.join(RES, f"{tag}.png"), dpi=400,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {tag}.png")


draw([Lres], "62b_realmap_recovered",
     "SERS-ink map · 12:12:12\nrecovered composition\nwhite = recovered ok")
draw([Lsurf, Lres], "62b_realmap_both",
     "SERS-ink map · 12:12:12\nsurface → recovered\n(inner → outer)")
