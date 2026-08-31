# -*- coding: utf-8 -*-
"""FINAL 삼각도 PNG — 27_ternary_truepred_FINAL_{nnls,pls,mlp}.csv 를 조립용 투명
패널로 렌더링. 스타일: 열린 회색 원(true) → 채운 원(pred, 정확도 RdYlGn), 회색
연결선, 20% 내부 격자, 라벨 없음(조립 시 코너 라벨은 조판에서). 꼭짓점 규약:
THI 위 / TBZ 좌하 / DQ 우하 (figure (b) 관례).

실행:  python fig_ternary_final_png.py
출력:  documentation/results/27_ternary_FINAL_{method}.png + _strip.png
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
TOP = np.array([0.5, np.sqrt(3) / 2]); LEFT = np.array([0.0, 0.0]); RIGHT = np.array([1.0, 0.0])
INK = "#20262e"; MUTE = "#8a919b"; GRID = "#d9dde2"


def xy(dq, tbz, thi):
    f = np.array([dq, tbz, thi], float); f = f / f.sum()
    return f[2] * TOP + f[1] * LEFT + f[0] * RIGHT


HIGH_ONLY = True     # 삼각도는 공인 고농도 세트(성분 >=100 µM, 100:1 제외)만 — 92조건
                     # 전부를 얹으면 선형구간이 섞여 방법 간 차이가 씻겨 보인다.


def _is_high(cond):
    import re
    g = re.match(r"DQ(\d+)-TB(\d+)-TH(\d+)", cond)
    return g and max(int(x) for x in g.groups()) >= 100


def draw(method, ax):
    rows = list(csv.DictReader(open(os.path.join(RES, f"27_ternary_truepred_FINAL_{method}.csv"),
                                    encoding="utf-8-sig")))
    if HIGH_ONLY:
        rows = [r for r in rows if r["condition"] and _is_high(r["condition"])]
    pairs = []
    cur = {}
    for r in rows:
        if r["type"] == "true":
            cur = {"t": xy(float(r["DQ_pct_X"]), float(r["TBZ_pct_Y"]), float(r["THI_pct_Z"])),
                   "acc": float(r["accuracy"])}
        elif r["type"] == "pred":
            cur["p"] = xy(float(r["DQ_pct_X"]), float(r["TBZ_pct_Y"]), float(r["THI_pct_Z"]))
            pairs.append(cur)
    # 외곽 + 20% 격자
    tri = np.array([LEFT, RIGHT, TOP, LEFT])
    for k in range(1, 5):
        f = k / 5
        ax.plot(*zip(xy(1 - f, f, 0), xy(1 - f, 0, f)), color=GRID, lw=0.6, zorder=1)
        ax.plot(*zip(xy(f, 1 - f, 0), xy(0, 1 - f, f)), color=GRID, lw=0.6, zorder=1)
        ax.plot(*zip(xy(f, 0, 1 - f), xy(0, f, 1 - f)), color=GRID, lw=0.6, zorder=1)
    ax.plot(tri[:, 0], tri[:, 1], color=INK, lw=1.6, zorder=2)
    cmap = matplotlib.colormaps["RdYlGn"]
    norm = Normalize(vmin=0.4, vmax=1.0, clip=True)   # 관측범위 스트레치 — 0-1 전체론 전부 황록으로 뭉개진다
    for q in pairs:
        ax.annotate("", xy=q["p"], xytext=q["t"], zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=1.0,
                                    alpha=0.8, shrinkA=4, shrinkB=5,
                                    mutation_scale=9))
    for q in pairs:
        ax.scatter(*q["t"], s=46, facecolor="white", edgecolor=MUTE, linewidth=1.1, zorder=4)
        ax.scatter(*q["p"], s=64, facecolor=cmap(norm(q["acc"])), edgecolor="white",
                   linewidth=0.6, zorder=5)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, TOP[1] + 0.05)
    ax.set_aspect("equal"); ax.axis("off")


TAG = "high" if HIGH_ONLY else "all92"
for method in ("nnls", "pls", "mlp"):
    fig, ax = plt.subplots(figsize=(5.4, 4.9))
    draw(method, ax)
    fig.savefig(os.path.join(RES, f"27_ternary_FINAL_{TAG}_{method}.png"),
                dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"27_ternary_FINAL_{TAG}_{method}.png")

fig, axs = plt.subplots(1, 3, figsize=(15.6, 4.9))
for ax, method in zip(axs, ("nnls", "pls", "mlp")):
    draw(method, ax)
fig.savefig(os.path.join(RES, f"27_ternary_FINAL_{TAG}_strip.png"),
            dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02)
print(f"27_ternary_FINAL_{TAG}_strip.png")
