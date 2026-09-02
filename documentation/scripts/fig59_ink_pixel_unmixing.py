# -*- coding: utf-8 -*-
"""59 — 잉크 글씨맵의 활성 픽셀별 unmixing.

"약 400개 active 점이 있는데 그 점들을 어떻게 unmixing했나 나와야 할 것 같은데"
— 배포 파이프라인(NNLS gate 0.15 → shared MLP) 그대로, 픽셀 하나하나의
조성을 (a) 공간맵 색 혼합, (b) 픽셀별 stacked-bar 바코드로 보여준다.

실행:  python -u fig59_ink_pixel_unmixing.py
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from dataset import load_preprocess  # noqa: E402
from dl_model import load_model, apply_model_pixels  # noqa: E402
from unmix import unmix_map  # noqa: E402

DLM = r"S:\Google Drive\내 드라이브\ACF_PEST_DB\260831_Model_FINAL\mlp_composition_260831_final.dlm"
PURE = r"S:\Google Drive\내 드라이브\ACF_PEST_DB\Pure"
MAP = r"S:\Google Drive\내 드라이브\ACF_PEST_DB\Pest\260812_12 trio_THI_TBZ_DQ.csv"
OUT = os.path.join(ROOT, "documentation", "results")

sys.path.insert(0, HERE)
import labfig  # noqa: E402  — 색 단일 출처 = Pure/colors.json

CO = {s: labfig.CO[s] for s in ("DQ", "TBZ", "THI")}
ORDER = ["DQ", "TBZ", "THI"]  # stacked-bar order, bottom→top

m = load_model(DLM)
cfg = load_preprocess(PURE)
r = unmix_map(data_dir=PURE, test_path=MAP, method="dlpx",
              baseline=bool(m.get("baseline", cfg["baseline"])),
              trim=cfg["trim"], min_frac=0.15, hit_mode="threshold",
              dl_model=m)
pk = np.clip(np.asarray(apply_model_pixels(m, r.wn, r.spectra), float), 0, None)
cols = [list(m["subs"]).index(s) for s in ORDER]
R = pk[:, cols] / (pk[:, cols].sum(1, keepdims=True) + 1e-12)
hit = np.asarray(r.hit, bool)
sel = np.where(hit)[0]
print(f"active pixels: {len(sel)} / {r.n_pixels}")
mean = R[sel].mean(0)
print("map mean (DQ,TBZ,THI):", np.round(mean * 100, 1))

xy = np.asarray(r.coords, float)
rgb_subs = np.array([to_rgb(CO[s]) for s in ORDER])

fig, (axm, axb) = plt.subplots(
    1, 2, figsize=(10.6, 4.4), gridspec_kw=dict(width_ratios=[1.0, 1.35]))

# (a) 공간맵 — 활성 픽셀 색 = 조성 가중 색 혼합
axm.scatter(xy[~hit, 0], xy[~hit, 1], s=9, c="#eceef0", marker="s")
mix = R[sel] @ rgb_subs
axm.scatter(xy[sel, 0], xy[sel, 1], s=11, c=np.clip(mix, 0, 1), marker="s")
axm.set_aspect("equal")
axm.set_axis_off()
axm.set_title(f"{len(sel)} active pixels · color = MLP composition",
              fontsize=9.5, color="#3f454c")

# (b) 바코드 — 픽셀 하나 = 세로 stacked bar 하나 (THI 분율로 정렬)
o = np.argsort(R[sel, 2])
Rs = R[sel][o]
x = np.arange(len(Rs))
bottom = np.zeros(len(Rs))
for k, s in enumerate(ORDER):
    axb.bar(x, Rs[:, k], bottom=bottom, width=1.0, color=CO[s], linewidth=0)
    bottom += Rs[:, k]
for k, s in enumerate(ORDER):
    axb.text(1.012, mean[:k].sum() + mean[k] / 2, f"{s} {mean[k]*100:.0f}%",
             transform=axb.get_yaxis_transform(), fontsize=8.5,
             color=CO[s], va="center", fontweight="bold")
axb.set_xlim(0, len(Rs))
axb.set_ylim(0, 1)
axb.set_xlabel("active pixels (sorted by THI fraction)", fontsize=8.5,
               color="#5a6067")
axb.set_ylabel("pixel composition", fontsize=8.5, color="#5a6067")
axb.tick_params(labelsize=7.5, colors="#8a919b", length=2)
for spn in axb.spines.values():
    spn.set_visible(False)
axb.set_title("each pixel unmixed individually · map mean at right",
              fontsize=9.5, color="#3f454c")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "59_ink_pixel_unmixing.png"), dpi=400,
            bbox_inches="tight", facecolor="white")
print("saved 59_ink_pixel_unmixing.png")
