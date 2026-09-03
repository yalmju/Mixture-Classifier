# -*- coding: utf-8 -*-
"""62b — 실제 글씨맵의 positive 픽셀을 circos 링으로 재구성.

부채꼴 = 게이트(NNLS 0.15) 통과 픽셀(표면 THI 분율 오름차순),
링 = 안쪽 3개 표면(NNLS) 조성 / 바깥 3개 복원(MLP) 조성,
색 = 제조 진실(1/3씩) 대비 배율 log2, ½×–2× 클립(판정 기준 2×).
"조성 복원이 된다"를 real map 픽셀 단위로: 안쪽은 얼룩(THI 독식),
바깥은 흰색(1/3씩 복원)이어야 한다.

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
from matplotlib.colors import LinearSegmentedColormap, Normalize

from dataset import load_preprocess  # noqa: E402
from dl_model import load_model, apply_model_pixels  # noqa: E402
from unmix import unmix_map  # noqa: E402

DLM = r"S:\Google Drive\내 드라이브\ACF_PEST_DB\260831_Model_FINAL\mlp_composition_260831_final.dlm"
PURE = r"S:\Google Drive\내 드라이브\ACF_PEST_DB\Pure"
MAP = r"S:\Google Drive\내 드라이브\ACF_PEST_DB\Pest\260812_12 trio_THI_TBZ_DQ.csv"
RES = os.path.join(ROOT, "documentation", "results")
SUBS = ["DQ", "TBZ", "THI"]
TRUE = 1.0 / 3.0                     # 제조 12:12:12

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
Rsurf = np.asarray(r.ratio_nb, float)
sel = np.where(np.asarray(r.hit, bool))[0]
o = np.argsort(Rsurf[sel, 2])        # 표면 THI 분율 오름차순
sel = sel[o]
n = len(sel)
print("positive pixels:", n)

M = np.full((6, n), np.nan)
M[0:3] = np.log2(np.clip(Rsurf[sel].T / TRUE, 1e-3, None))
M[3:6] = np.log2(np.clip(Rres[sel].T / TRUE, 1e-3, None))

cmap = LinearSegmentedColormap.from_list(
    "pw", ["#5c5a9e", "#a9a7cf", "#f4f2ee", "#f3c977", "#e2952e"])
norm = Normalize(-1, 1)              # ½×–2×

A0 = np.deg2rad(96)
SPAN = np.deg2rad(348)
edges = A0 + SPAN * np.arange(n + 1) / n
R0, DR, GAP = 1.0, 0.28, 0.10

fig, ax = plt.subplots(figsize=(8.6, 8.6), subplot_kw=dict(polar=True))
th_c = (edges[:-1] + edges[1:]) / 2
w_c = (edges[1] - edges[0])
for ring in range(6):
    r_in = R0 + ring * DR + (GAP if ring >= 3 else 0)
    ax.bar(th_c, np.full(n, DR * 0.94), bottom=r_in, width=w_c,
           color=cmap(norm(M[ring])), edgecolor="none", zorder=2)
ax.text(np.deg2rad(90), 0.0,
        "SERS-ink map · 12:12:12\nsurface → recovered\n(inner → outer)",
        fontsize=9, color="#3f454c", ha="center", va="center", zorder=4)
for k, s in enumerate(SUBS):
    for base in (R0, R0 + GAP + 3 * DR):
        ax.text(np.deg2rad(92.5), base + (k + 0.5) * DR, s,
                fontsize=5.2, color="#6a7178", ha="center", va="center",
                zorder=4)
r_out = R0 + GAP + 6 * DR
ax.text(np.deg2rad(90), r_out + 0.45,
        f"{n} gated pixels · sorted by surface THI fraction →", fontsize=7,
        color="#8a919b", ha="center")
ax.set_xticks([]); ax.set_yticks([])
ax.set_ylim(0, r_out + 0.6)
ax.spines["polar"].set_visible(False)
cax = fig.add_axes([0.90, 0.80, 0.015, 0.13])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_ticks([-1, 0, 1])
cb.set_ticklabels(["50", "100", "200%"])
cax.tick_params(labelsize=6, length=2)
cb.outline.set_linewidth(0.4)
cax.set_title("recovery", fontsize=6, color="#3f454c", pad=3)
fig.savefig(os.path.join(RES, "62b_circular_realmap.png"), dpi=400,
            bbox_inches="tight", facecolor="white")
print("saved 62b_circular_realmap.png")
