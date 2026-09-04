# -*- coding: utf-8 -*-
"""63 — 성분별 SERS 응답 곡선 (µM당 신호 격차의 영수증).

데이터: 배포 사이드카 .pxknn.npz의 단일성분 검량 픽셀(CAL-*-{9..144},
성분당 레벨 5 × 10–15px). x = 농도, y = 자기 마커 밴드 신호(픽셀 median,
수염 = IQR), log–log. TBZ 곡선이 ~3× 아래에 깔리는 것(응답 격차)과
고농도 기울기 꺾임(자체억제 경향)이 이 그림의 주장 전부다.

실행:  python -u fig63_response_curves.py
"""
import io
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import labfig  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(HERE, "..", "results")
NPZ = r"S:\Google Drive\내 드라이브\ACF_PEST_DB\260831_Model_FINAL\mlp_composition_260831_final.pxknn.npz"
SUBS = ["DQ", "TBZ", "THI"]
CO = {s: labfig.CO[s] for s in SUBS}

lib = np.load(NPZ, allow_pickle=True)
Fz = np.asarray(lib["Fz"], float)
mu = np.asarray(lib["mu"], float)
sd = np.asarray(lib["sd"], float)
cond = np.asarray(lib["cond"]).astype(str)
sig = np.expm1((Fz * sd + mu)[:, :3])      # 밴드 신호 (원 카운트)

fig, ax = plt.subplots(figsize=(5.6, 4.2))
for k, s in enumerate(SUBS):
    xs, med, q1, q3 = [], [], [], []
    for c in (9, 18, 36, 72, 144):
        m = cond == f"CAL-{s}-{c}"
        if not m.any():
            continue
        v = sig[m, k]
        xs.append(c)
        med.append(np.median(v))
        q1.append(np.percentile(v, 25))
        q3.append(np.percentile(v, 75))
    xs = np.asarray(xs, float)
    med = np.asarray(med)
    ax.errorbar(xs, med, yerr=[med - np.asarray(q1), np.asarray(q3) - med],
                fmt="o-", ms=5.5, lw=1.8, capsize=3, color=CO[s], label=s)
    # 저농도 구간 기울기 → counts/µM 감각치
    print(f"{s}: counts/µM @9µM = {med[0] / xs[0]:.0f} · "
          f"log-log slope 9–36 = "
          f"{np.polyfit(np.log(xs[:3]), np.log(med[:3]), 1)[0]:.2f} · "
          f"36–144 = {np.polyfit(np.log(xs[2:]), np.log(med[2:]), 1)[0]:.2f}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([9, 18, 36, 72, 144])
ax.set_xticklabels(["9", "18", "36", "72", "144"])
ax.set_xlabel("concentration (µM)", fontsize=9, color="#5a6067")
ax.set_ylabel("own marker-band signal (counts)", fontsize=9, color="#5a6067")
ax.tick_params(labelsize=8, colors="#8a919b", length=2.5)
for kk in ("top", "right"):
    ax.spines[kk].set_visible(False)
for kk in ("left", "bottom"):
    ax.spines[kk].set_color("#b6bcc4")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(RES, "63_response_curves.png"), dpi=400,
            bbox_inches="tight", facecolor="white")
print("saved 63_response_curves.png")
