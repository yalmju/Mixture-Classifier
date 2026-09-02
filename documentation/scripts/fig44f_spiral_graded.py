# -*- coding: utf-8 -*-
"""44f — guided spiral, 초록 창을 단계 그라데이션으로.

44e에서 바뀐 것: 평평한 0.5–2× 창 대신 truth에서 멀어질수록 옅어지는
동심 밴드(±1.25, 1.5, 1.75, 2×; log 대칭). 점이 어느 밴드에 앉는지로
"얼마나 잘 복원됐는지"가 바로 읽힌다. 창은 중립 회색 — 초록은 TBZ 점 색과
겹쳐서 금지. 점 색 = Pure/colors.json.
개별판은 무자막(조립용), strip에는 작은 배율 라벨.

데이터: 17_composition_all_conditions_master.csv (92조건, imbalance 제외;
성분이 든 조건만 → 패널당 85점: 표면(흰) 85 + 복원(색) 85).

실행:  python -u fig44f_spiral_graded.py
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import labfig  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(HERE, "..", "results")
INK = "#20262e"
MUTE = "#a6acb5"
WINDOW = "#7d848c"  # 중립 회색 (TBZ 초록과 충돌 방지)
CO = {s: labfig.CO[s] for s in ("DQ", "TBZ", "THI")}

rows = [r for r in csv.DictReader(
    open(os.path.join(RES, "17_composition_all_conditions_master.csv"),
         encoding="utf-8-sig")) if r["imbalance_100x"] == "0"]
RMAX = 2.0
PAD = 0.30
# truth에서 바깥쪽으로 옅어지는 밴드 경계(배율)와 흰색 혼합비.
# 투명 PNG에서 알파는 뷰어마다 다르게 보여서 불투명 단계색으로 만든다.
GRADE = [(1.25, 0.45), (1.5, 0.62), (1.75, 0.76), (2.0, 0.88)]


def _band_color(f_white):
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(WINDOW)
    return (r + (1 - r) * f_white, g + (1 - g) * f_white,
            b + (1 - b) * f_white)


def rad(x):
    return np.clip(np.log2(np.clip(x, 1e-3, None)), -RMAX, RMAX) + RMAX


def draw_sp(s, ax, labels=False):
    recs = []
    for r in rows:
        t = float(r[f"Ratio_{s}_True"])
        if t <= 0:
            continue
        recs.append((float(r[f"Ratio_{s}_NNLS_Pred"]) / t,
                     float(r[f"Ratio_{s}_MLP_heldout"]) / t))
    recs.sort(key=lambda x: x[0])
    n = len(recs)
    ang = np.deg2rad(100) + np.linspace(0, np.deg2rad(340), n)
    rs = np.array([rad(a) for a, _ in recs])
    rm = np.clip(np.array([rad(b) for _, b in recs]), 0.10, 2 * RMAX - 0.06)
    th = np.linspace(0, 2 * np.pi, 240)
    lo = 1.0
    for f, fw in GRADE:  # log 대칭: [1/f, 1/lo]와 [lo, f]
        c = _band_color(fw)
        ax.fill_between(th, rad(lo), rad(f), color=c, lw=0, zorder=0)
        ax.fill_between(th, rad(1 / f), rad(1 / lo), color=c, lw=0, zorder=0)
        lo = f
    ax.plot(th, np.full_like(th, RMAX), color="#e6b93c", lw=2.2, zorder=2)
    for f, _ in GRADE:
        for v in (f, 1 / f):
            ax.plot(th, np.full_like(th, rad(v)), color="white", lw=0.7,
                    alpha=0.85, zorder=1)
    for a, r0, r1 in zip(ang, rs, rm):
        if abs(r1 - r0) < 0.05:
            continue
        ax.annotate("", xy=(a, r1), xytext=(a, r0), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=0.6,
                                    alpha=0.55, shrinkA=1, shrinkB=3,
                                    mutation_scale=6))
    ax.scatter(ang, rs, s=11, facecolor="white", edgecolor=MUTE,
               linewidth=0.8, zorder=4)
    ax.scatter(ang, rm, s=17, color=CO[s], alpha=0.9, edgecolor="white",
               linewidth=0.3, zorder=5)
    if labels:
        import matplotlib.patheffects as pe
        halo = [pe.withStroke(linewidth=1.6, foreground="white")]
        a0 = np.deg2rad(96)
        ax.text(a0, RMAX, "1×", fontsize=5.6, color="#8a6d1a", ha="center",
                va="center", zorder=6, path_effects=halo)
        for f, _ in GRADE:
            ax.text(a0, rad(f), f"{f:g}×", fontsize=5.0, color="#3f454c",
                    ha="center", va="center", zorder=6, path_effects=halo)
        ax.text(a0, rad(1 / 2), "0.5×", fontsize=5.0, color="#3f454c",
                ha="center", va="center", zorder=6, path_effects=halo)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 2 * RMAX + PAD)
    ax.spines["polar"].set_visible(False)


for s in ("THI", "TBZ", "DQ"):
    fig, ax = plt.subplots(figsize=(4.9, 4.9), subplot_kw=dict(polar=True))
    draw_sp(s, ax)
    fig.savefig(os.path.join(RES, f"44f_spiral_graded_notext_{s}.png"),
                dpi=300, transparent=True, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
fig, axs = plt.subplots(1, 3, figsize=(13.8, 4.7), subplot_kw=dict(polar=True))
for ax, s in zip(axs, ("THI", "TBZ", "DQ")):
    draw_sp(s, ax, labels=True)
fig.subplots_adjust(wspace=0.08)
fig.savefig(os.path.join(RES, "44f_spiral_graded_strip.png"), dpi=300,
            transparent=True, bbox_inches="tight", pad_inches=0.02)
print("saved 44f x4")
