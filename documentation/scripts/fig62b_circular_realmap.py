# -*- coding: utf-8 -*-
"""62b — 글씨맵(12:12:12) 픽셀 circos, 저채도 3색.

부채꼴 = 게이트 통과 픽셀(행 우선 스캔 — 연속 구간 = 글씨 획).

62b_realmap_correction.png (3링): 셀 색 = 그 픽셀·성분의 **왜곡 교정률**
  c = 1 − |log2 recovered/truth| / |log2 surface/truth|
  노랑 = 잘 교정(c ≥ 2/3), 흰 = 중간·원래 무왜곡, 보라 = 교정 안 됨(c < 1/3).
62b_realmap_both.png (6링): 안 = 표면, 밖 = 복원, over/ok/under 저채도 3색.

중앙 = 측정된 사실만(표면 → 복원 조성, 제조값 병기). recovery/교정률은
진실을 아는 검증에서만 존재하는 수 — 검증 전시로만 쓴다.

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

# 저채도 팔레트
C_OK = "#f4f2ee"
C_OVER = "#ddb977"      # muted amber (over)
C_UNDER = "#a3a1c8"     # muted violet (under)
C_GOOD = "#e2c26f"      # muted yellow (교정 잘 됨)
C_MID = "#f4f2ee"
C_BAD = "#9a98c2"       # muted violet (교정 안 됨)
C_NA = "#e2e5e9"
_TOL = np.log2(1.25)

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


def cls_ou(v):
    """over/ok/under 저채도 3색."""
    if not np.isfinite(v):
        return C_NA
    if abs(v) <= _TOL:
        return C_OK
    return C_OVER if v > 0 else C_UNDER


def cls_corr(ls, lr):
    """왜곡 교정률 3색: 노랑 = 잘 교정, 보라 = 안 됨."""
    if not (np.isfinite(ls) and np.isfinite(lr)):
        return C_NA
    if abs(ls) <= _TOL:               # 애초에 왜곡이 없던 픽셀
        return C_MID
    c = 1 - abs(lr) / abs(ls)
    return C_GOOD if c >= 2 / 3 else (C_MID if c >= 1 / 3 else C_BAD)


corr_colors = [[cls_corr(Lsurf[k, j], Lres[k, j]) for j in range(n)]
               for k in range(3)]
good_frac = [np.mean([c == C_GOOD for c in corr_colors[k]]) * 100
             for k in range(3)]
print("corrected>=2/3 fraction:",
      {s: f"{f:.0f}%" for s, f in zip(SUBS, good_frac)})

A0 = np.deg2rad(96)
SPAN = np.deg2rad(348)
edges = A0 + SPAN * np.arange(n + 1) / n
th_c = (edges[:-1] + edges[1:]) / 2
w_c = edges[1] - edges[0]

cs = Rsurf[sel].mean(0); cs = cs / cs.sum() * 100
cr = Rres[sel].mean(0); cr = cr / cr.sum() * 100
CENTER = ("composition (DQ:TBZ:THI)\n"
          f"surface {cs[0]:.0f} : {cs[1]:.0f} : {cs[2]:.0f}\n"
          f"recovered {cr[0]:.0f} : {cr[1]:.0f} : {cr[2]:.0f}\n"
          "formulation 33 : 33 : 33")
print(CENTER.replace("\n", " | "))


def legend(fig, items, title):
    lax = fig.add_axes([0.865, 0.80, 0.125, 0.14])
    lax.set_axis_off()
    lax.text(0.02, 1.04, title, fontsize=6.5, color="#3f454c",
             weight="bold")
    for i, (c, t) in enumerate(items):
        y = 0.80 - i * 0.30
        lax.add_patch(plt.Rectangle((0.02, y - 0.11), 0.16, 0.22,
                                    facecolor=c, edgecolor="#b6bcc4",
                                    linewidth=0.3))
        lax.text(0.24, y, t, fontsize=6.2, color="#5a6067", va="center")


def draw(rings, tag, leg_items, leg_title):
    n_ring = len(rings)
    R0 = 1.0
    DR = 0.42 if n_ring == 3 else 0.30
    GAP = 0.10
    fig, ax = plt.subplots(figsize=(8.6, 8.6), subplot_kw=dict(polar=True))
    for i, (name, colors_i) in enumerate(rings):
        r_in = R0 + i * DR + (GAP if (n_ring == 6 and i >= 3) else 0)
        ax.bar(th_c, np.full(n, DR * 0.94), bottom=r_in, width=w_c,
               color=colors_i, edgecolor="none", zorder=2)
        ax.text(np.deg2rad(92.5), r_in + 0.5 * DR, name, fontsize=5.6,
                color="#6a7178", ha="center", va="center", zorder=4)
    r_out = R0 + n_ring * DR + (GAP if n_ring == 6 else 0)
    ax.text(np.deg2rad(90), 0.0, CENTER, fontsize=8.5, color="#3f454c",
            ha="center", va="center", zorder=4)
    ax.text(np.deg2rad(90), r_out + 0.42,
            f"{n} gated pixels · scan order →", fontsize=7,
            color="#8a919b", ha="center")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, r_out + 0.55)
    ax.spines["polar"].set_visible(False)
    legend(fig, leg_items, leg_title)
    fig.savefig(os.path.join(RES, f"{tag}.png"), dpi=400,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {tag}.png")


draw([(SUBS[k], corr_colors[k]) for k in range(3)],
     "62b_realmap_correction",
     [(C_GOOD, "corrected ≥ 2/3"), (C_MID, "1/3–2/3 · no distortion"),
      (C_BAD, "< 1/3")],
     "distortion corrected")

ou_surf = [[cls_ou(v) for v in Lsurf[k]] for k in range(3)]
ou_res = [[cls_ou(v) for v in Lres[k]] for k in range(3)]
draw([(SUBS[k], ou_surf[k]) for k in range(3)]
     + [(SUBS[k], ou_res[k]) for k in range(3)],
     "62b_realmap_both",
     [(C_OVER, "> 125%  over"), (C_OK, "80–125  ok"),
      (C_UNDER, "< 80%  under")],
     "recovery vs 1/3")
