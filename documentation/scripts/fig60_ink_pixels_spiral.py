# -*- coding: utf-8 -*-
"""60 — 잉크 글씨맵의 활성 픽셀들을 스파이럴(폴라) 문법으로.

제조 진실 12:12:12 → 참 조성 1/3씩. 픽셀마다
  표면(NNLS ratio) = 저채도 굵은 running-median 곡선,
  복원(MLP)       = 고채도 점 + 가는 running-median 곡선,
반지름 = 참값(1/3) 대비 배율(log2, ±2 클립), 점선 원 = truth.
각도 = 표면 THI 분율 순위(310° 스팬). 색 = Pure/colors.json.

실행:  python -u fig60_ink_pixels_spiral.py
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import labfig  # noqa: E402

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

CO = {s: labfig.CO[s] for s in ("DQ", "TBZ", "THI")}
ORDER = ["DQ", "TBZ", "THI"]
INK = "black"
TRUE_SHARE = 1.0 / 3.0  # 제조 12:12:12


def mute(c, f=0.68):
    r, g, b = to_rgb(c)
    gr = 0.82
    return (r + (gr - r) * f, g + (gr - g) * f, b + (gr - b) * f)


RMAX = 2.0
PAD = 0.3


def rad(x):
    return np.clip(np.log2(np.clip(x, 1e-3, None)), -RMAX, RMAX) + RMAX


A0, SPAN = 115, 310


def ang(p):  # p in 0..100
    return np.deg2rad(A0 + np.asarray(p) / 100.0 * SPAN)


m = load_model(DLM)
cfg = load_preprocess(PURE)
r = unmix_map(data_dir=PURE, test_path=MAP, method="dlpx",
              baseline=bool(m.get("baseline", cfg["baseline"])),
              trim=cfg["trim"], min_frac=0.15, hit_mode="threshold",
              dl_model=m)
pk = np.clip(np.asarray(apply_model_pixels(m, r.wn, r.spectra), float), 0, None)
cols = [list(m["subs"]).index(s) for s in ORDER]
Rres = pk[:, cols] / (pk[:, cols].sum(1, keepdims=True) + 1e-12)
Rsurf = np.asarray(r.ratio_nb, float)  # NNLS, order DQ TBZ THI
sel = np.where(np.asarray(r.hit, bool))[0]
print(f"active pixels: {len(sel)}")

o = np.argsort(Rsurf[sel, 2])           # 표면 THI 분율 순위
sel = sel[o]
p = np.arange(len(sel)) / max(len(sel) - 1, 1) * 100.0


def runmed(vals, win=6.0):
    xs = np.linspace(1, 99, 80)
    ys = np.array([np.median(vals[np.abs(p - c) <= win])
                   if (np.abs(p - c) <= win).sum() >= 5 else np.nan
                   for c in xs])
    ok = ~np.isnan(ys)
    return xs[ok], ys[ok]


fig, ax = plt.subplots(figsize=(7.0, 7.0), subplot_kw=dict(polar=True))
th = np.linspace(0, 2 * np.pi, 240)
ax.plot(th, np.full_like(th, RMAX), color=INK, lw=1.1, ls=(0, (4, 3)), zorder=3)
for f in (0.5, 2.0):
    ax.plot(th, np.full_like(th, rad(f)), color="0.84", lw=0.6, ls=":", zorder=1)

# 표면(NNLS): 저채도 굵은 곡선
for k, s in enumerate(ORDER):
    e = Rsurf[sel, k] / TRUE_SHARE
    xs, ys = runmed(e)
    ax.plot(ang(xs), np.clip(rad(ys), 0.06, 2 * RMAX + 0.1), color=mute(CO[s]),
            lw=4.0, zorder=2, solid_capstyle="round")
# 복원(MLP): 고채도 점 + 가는 곡선
for k, s in enumerate(ORDER):
    e = Rres[sel, k] / TRUE_SHARE
    ax.scatter(ang(p), np.clip(rad(e), 0.06, 2 * RMAX - 0.04), s=4.5,
               color=CO[s], alpha=0.30, edgecolor="none", zorder=4)
    xs, ys = runmed(e)
    ax.plot(ang(xs), np.clip(rad(ys), 0.06, 2 * RMAX - 0.04), color=CO[s],
            lw=2.2, alpha=0.95, zorder=5, solid_capstyle="round")

ax.text(np.deg2rad(A0 - 15), RMAX + 0.02, "truth", fontsize=8, color=INK,
        ha="center", va="center")
ax.text(np.deg2rad(A0 - 15), rad(2.0) + 0.10, "2×", fontsize=7.5,
        color="#8a919b", ha="center")
ax.text(np.deg2rad(A0 - 15), rad(0.5) - 0.16, "0.5×", fontsize=7.5,
        color="#8a919b", ha="center")
ax.text(float(ang(50)), 2 * RMAX + 0.55,
        f"{len(sel)} active pixels · sorted by surface THI fraction →",
        fontsize=8.5, color="#8a919b", ha="center")
ax.text(float(ang(88)), rad(2.9), "surface (NNLS)", fontsize=9,
        color=mute(CO["THI"], 0.4), fontweight="bold", ha="center")
ax.text(float(ang(12)), rad(0.55), "restored (MLP)", fontsize=9, color=INK,
        fontweight="bold", ha="center")
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim(0, 2 * RMAX + PAD)
ax.spines["polar"].set_visible(False)
fig.savefig(os.path.join(OUT, "60_ink_pixels_spiral.png"), dpi=400,
            bbox_inches="tight", facecolor="white")
print("saved 60_ink_pixels_spiral.png")
