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
# dlpx 모드에서 ratio_nb는 이미 모델 조성이다 — 표면(stage-1 NNLS)은
# A_evidence에서 가져와야 Δ가 0으로 죽지 않는다 (2026-09-03).
ev = np.asarray(getattr(r, "A_evidence", r.A), float)[:, r.nonbg]
Rsurf = ev / (ev.sum(1, keepdims=True) + 1e-12)
sel = np.where(np.asarray(r.hit, bool))[0]
# 스캔 순서(행 우선): THI-분율 정렬은 THI 링을 정의상 그라데이션으로 만들어
# 오독됨 — 공간 순서면 글씨 획이 각도 방향의 연속 구간으로 살아남는다.
xy = np.asarray(r.coords, float)
sel = sel[np.lexsort((xy[sel, 0], xy[sel, 1]))]
n = len(sel)
print("positive pixels:", n)

from matplotlib.colors import to_rgb

D = (Rres[sel] - Rsurf[sel]).T        # (3, n) Δ조성 = recovered − surface
CO = {s: labfig.CO[s] for s in SUBS}
# 링 4 = 복원된 픽셀의 우세 성분색(혼합색은 탁해서 판독 불가)
dom = np.argmax(Rres[sel], axis=1)
mix = np.array([to_rgb(CO[SUBS[k]]) for k in dom])

cmap = LinearSegmentedColormap.from_list(
    "pw", ["#5c5a9e", "#a9a7cf", "#f4f2ee", "#f3c977", "#e2952e"])
norm = Normalize(-0.4, 0.4)           # Δ조성 ±40%p

A0 = np.deg2rad(96)
SPAN = np.deg2rad(348)
edges = A0 + SPAN * np.arange(n + 1) / n
R0, DR, GAP = 1.0, 0.28, 0.10

fig, ax = plt.subplots(figsize=(8.6, 8.6), subplot_kw=dict(polar=True))
th_c = (edges[:-1] + edges[1:]) / 2
w_c = (edges[1] - edges[0])
DR = 0.34                              # 링 4개라 두껍게
for ring in range(3):                  # Δ조성 링 (안→밖: DQ, TBZ, THI)
    ax.bar(th_c, np.full(n, DR * 0.94), bottom=R0 + ring * DR, width=w_c,
           color=cmap(norm(D[ring])), edgecolor="none", zorder=2)
ax.bar(th_c, np.full(n, DR * 0.94), bottom=R0 + 3 * DR + GAP, width=w_c,
       color=mix, edgecolor="none", zorder=2)   # 복원 조성(성분색 혼합)
ax.text(np.deg2rad(90), 0.0,
        "SERS-ink map · 12:12:12\ncorrection Δ (rings 1–3)\n"
        "dominant after recovery (ring 4)",
        fontsize=9, color="#3f454c", ha="center", va="center", zorder=4)
for k, s in enumerate(SUBS):
    ax.text(np.deg2rad(92.5), R0 + (k + 0.5) * DR, s, fontsize=5.6,
            color="#6a7178", ha="center", va="center", zorder=4)
ax.text(np.deg2rad(92.5), R0 + 3 * DR + GAP + 0.5 * DR, "dom.",
        fontsize=5.6, color="#6a7178", ha="center", va="center", zorder=4)
r_out = R0 + GAP + 4 * DR
ax.text(np.deg2rad(90), r_out + 0.45,
        f"{n} gated pixels · scan order (row-major) →", fontsize=7,
        color="#8a919b", ha="center")
ax.set_xticks([]); ax.set_yticks([])
ax.set_ylim(0, r_out + 0.6)
ax.spines["polar"].set_visible(False)
cax = fig.add_axes([0.90, 0.80, 0.015, 0.13])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_ticks([-0.4, 0, 0.4])
cb.set_ticklabels(["−40", "0", "+40 %p"])
cax.tick_params(labelsize=6, length=2)
cb.outline.set_linewidth(0.4)
cax.set_title("Δ share", fontsize=6, color="#3f454c", pad=3)
fig.savefig(os.path.join(RES, "62b_circular_realmap.png"), dpi=400,
            bbox_inches="tight", facecolor="white")
print("saved 62b_circular_realmap.png")
