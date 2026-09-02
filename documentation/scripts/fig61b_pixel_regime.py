# -*- coding: utf-8 -*-
"""61b — 체제별 표면 농축 E를 SERS 맵 픽셀에서 직접 추출.

61(조건당 1점 = 맵 평균 NNLS)과 달리, 각 혼합물 맵을 NNLS로 unmix해서
게이트(0.15) 통과 픽셀 하나하나의 표면 조성 → E = 픽셀 share / 참 share.
의심맵 3장·imbalance(100x) 제외. 출력:
  61b_pixel_E_raw.csv          픽셀 단위 long (condition, substance, E, ...)
  61b_regime_summary_pixels.csv 체제×성분 median/IQR/n(픽셀)
  61b_regime_summary_pixels.png 61과 같은 판형, 픽셀 분포 기준

실행:  python -u fig61b_pixel_regime.py   (맵 ~90장 unmix — 수 분)
"""
from __future__ import annotations

import csv
import io
import json
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

from dataset import load_preprocess  # noqa: E402
from unmix import unmix_map  # noqa: E402

PURE = r"S:\Google Drive\내 드라이브\ACF_PEST_DB\Pure"
RES = os.path.join(ROOT, "documentation", "results")
SUBS = ["DQ", "TBZ", "THI"]
CO = {s: labfig.CO[s] for s in SUBS}
SUSPECT = {(250.0, 0.0, 50.0), (250.0, 50.0, 0.0), (500.0, 50.0, 0.0)}
REG = [("THI minority (share<=45%)", lambda sh, to: sh <= 45),
       ("THI majority (share>45%)", lambda sh, to: sh > 45),
       ("saturated (total>50uM)", lambda sh, to: to > 50)]

cfg = load_preprocess(PURE)
entries = json.load(open(os.path.join(PURE, "mixtures.json"), encoding="utf-8"))

raw = []
for i, e in enumerate(entries):
    conc = e.get("conc") or {}
    y = np.array([float(conc.get(s, 0.0)) * 1e6 for s in SUBS])
    tot = y.sum()
    if tot <= 0:
        continue
    if (y > 0).any() and y.max() / max(y[y > 0].min(), 1e-9) >= 99:
        continue  # imbalance 100x
    if tuple(np.round(y, 1)) in {tuple(map(float, t)) for t in SUSPECT}:
        print(f"[{i}] suspect skip: DQ{y[0]:g}-TB{y[1]:g}-TH{y[2]:g}", flush=True)
        continue
    sh = y[2] / tot * 100
    true_share = y / tot
    try:
        r = unmix_map(data_dir=PURE, test_path=e["path"], method="nnls",
                      baseline=cfg["baseline"], trim=cfg["trim"],
                      min_frac=0.15, hit_mode="threshold")
    except Exception as ex:
        print(f"[{i}] skip {os.path.basename(e['path'])}: {ex}", flush=True)
        continue
    R = np.asarray(r.ratio_nb, float)[np.asarray(r.hit, bool)]
    name = f"DQ{y[0]:g}-TB{y[1]:g}-TH{y[2]:g}"
    for k, s in enumerate(SUBS):
        if true_share[k] <= 0:
            continue
        for v in R[:, k] / true_share[k]:
            raw.append((name, s, round(float(v), 4), round(sh, 2), tot))
    if i % 10 == 0:
        print(f"[{i}] {name}: {len(R)} px", flush=True)

with open(os.path.join(RES, "61b_pixel_E_raw.csv"), "w", newline="",
          encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["condition", "substance", "E_pixel", "thi_share_pct",
                "total_uM"])
    w.writerows(raw)

vals = {s: [[] for _ in REG] for s in SUBS}
for name, s, v, sh, to in raw:
    for j, (_, freg) in enumerate(REG):
        if freg(sh, to):
            vals[s][j].append(v)

with open(os.path.join(RES, "61b_regime_summary_pixels.csv"), "w", newline="",
          encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["regime", "substance", "n_pixels", "median_E", "q1_E", "q3_E"])
    for j, (rname, _) in enumerate(REG):
        for s in ("THI", "TBZ", "DQ"):
            v = np.array(vals[s][j])
            w.writerow([rname, s, len(v), round(float(np.median(v)), 3),
                        round(float(np.percentile(v, 25)), 3),
                        round(float(np.percentile(v, 75)), 3)])
            print(rname, s, len(v), round(float(np.median(v)), 3))

fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.set_yscale("log", base=2)
ax.axhline(1.0, color="#9aa3ad", lw=1.3, ls=(0, (4, 3)), zorder=2)
for fref in (0.5, 2.0):
    ax.axhline(fref, color="0.86", lw=0.6, ls=":", zorder=1)
DX = {"THI": -0.16, "TBZ": 0.0, "DQ": 0.16}
for s in ("THI", "TBZ", "DQ"):
    x = np.arange(len(REG)) + DX[s]
    med = np.array([np.median(v) for v in vals[s]])
    q1 = np.array([np.percentile(v, 25) for v in vals[s]])
    q3 = np.array([np.percentile(v, 75) for v in vals[s]])
    ax.plot(x, med, color=CO[s], lw=1.4, alpha=0.55, zorder=3)
    ax.errorbar(x, med, yerr=[med - q1, q3 - med], fmt="o", ms=6.5,
                color=CO[s], ecolor=CO[s], elinewidth=1.4, capsize=3,
                zorder=4, label=s)
    for xi, mi in zip(x, med):
        ax.annotate(f"×{mi:.2f}", (xi, mi), textcoords="offset points",
                    xytext=(0, 8 if s == "THI" else -14), fontsize=7.5,
                    color=CO[s], ha="center", fontweight="bold")
ax.text(2.44, 1.0, "truth", fontsize=8, color="#9aa3ad", va="center")
ax.set_xticks(range(len(REG)))
ax.set_xticklabels(["THI minority\n(share ≤ 45%)", "THI majority\n(share > 45%)",
                    "saturated\n(total > 50 µM)"], fontsize=9, color="#3f454c")
ax.set_ylabel("pixel surface enrichment E", fontsize=9, color="#5a6067")
ax.tick_params(labelsize=8, colors="#8a919b", length=2.5)
for k in ("top", "right"):
    ax.spines[k].set_visible(False)
for k in ("left", "bottom"):
    ax.spines[k].set_color("#b6bcc4")
ax.legend(frameon=False, fontsize=8.5, loc="lower left")
ax.set_xlim(-0.5, 2.7)
fig.tight_layout()
fig.savefig(os.path.join(RES, "61b_regime_summary_pixels.png"), dpi=400,
            bbox_inches="tight", facecolor="white")
print("saved 61b csv x2 + png")
