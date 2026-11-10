"""Figure 3f 초안 — DUET-Net 을 신경망 그림체로 (상자 흐름도 대신).

  위 줄  조성망  f(x_i): 픽셀 스펙트럼 1,290 → 256 → 64 → 4 (DQ / TBZ / THI / blank)
         → 맵 평균 → Solution composition (%)
  아래 줄 농도망  g(m): 맵 특징 12 → 128 → 32 → 3 = Δlog10C,
         C = C_cal · 10^Δ → Apparent concentration (µM)

  초안이다 — 최종 외형은 일러스트에서. draw(ax) 를 fig3_final.py 가 그대로 불러 쓴다.

  실행:  ../../.venv/bin/python3 -u fig3f_network.py  →  fig3f_network.png (MLP) · fig3f_network_pls.png (PLS)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = {"DQ": "#1f78c8", "TBZ": "#78c240", "THI": "#e3425f", "blank": "0.7"}
NODE = "#dfe7f3"; EDGE = "0.35"; LINK = "0.78"


def _layer(ax, x, ys, r, fcs, T):
    for y, fc in zip(ys, fcs):
        ax.plot([x], [y], "o", ms=r, mfc=fc, mec=EDGE, mew=0.5, transform=T, zorder=3,
                clip_on=False)


def _links(ax, xa, ya, xb, yb, T):
    for a in ya:
        for b in yb:
            ax.plot([xa, xb], [a, b], "-", color=LINK, lw=0.35, transform=T, zorder=1)


def _net(ax, xs, counts, labels, y0, h, out_cols, T, r=5.2):
    """층별 노드 수 counts 로 그린 작은 MLP. 층 이름은 아래에."""
    prev = None
    for x, n, lab in zip(xs, counts, labels):
        ys = y0 + h * (np.arange(n) - (n - 1) / 2) / max(counts)
        fcs = out_cols if (x == xs[-1] and out_cols) else [NODE] * n
        if prev is not None:
            _links(ax, prev[0], prev[1], x, ys, T)
        _layer(ax, x, ys, r, fcs, T)
        ax.text(x, y0 - h / 2 - 0.035, lab, transform=T, ha="center", va="top",
                fontsize=5.6, color="0.3")
        prev = (x, ys)
    return prev[1]


def _arrow(ax, p, q, T):
    ax.annotate("", q, p, xycoords=T, arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8))


def draw(ax, learner="mlp"):
    """learner="mlp": 조성망 1,290–256–64–4 · "pls": PLS(잠재변수 8개)를 선형 한 층으로."""
    ax.set_axis_off(); T = ax.transAxes
    rng = np.random.default_rng(3)

    # ── 입력: SERS 맵 → NNLS 게이트를 통과한 픽셀 ────────────────────────────
    n = 6; s = 0.026; x0, y0 = 0.00, 0.69
    keep = rng.random((n, n)) < 0.55
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((x0 + j * s, y0 + i * s), s, s, transform=T, lw=0.3,
                                   ec="white", fc="#f2b54a" if keep[i, j] else "0.88"))
    ax.text(x0 + n * s / 2, y0 + n * s + 0.02, "SERS map", transform=T, ha="center",
            va="bottom", fontsize=6.4)
    ax.text(x0 + n * s / 2, y0 - 0.02, "NNLS gate:\nanalyte pixels", transform=T,
            ha="center", va="top", fontsize=5.6, color="0.3", linespacing=1.2)
    _arrow(ax, (0.17, 0.77), (0.24, 0.77), T)

    # ── 위: 조성망 ───────────────────────────────────────────────────────────
    if learner == "pls":
        ax.text(0.47, 1.00, "Composition model (PLS)  $f(x_i)$", transform=T, ha="center",
                va="top", fontsize=6.6, weight="bold")
        outs = _net(ax, [0.28, 0.475, 0.67], [6, 5, 3],
                    ["1,290", "8 latent variables", "3"], 0.77, 0.30,
                    [COLS["THI"], COLS["TBZ"], COLS["DQ"]], T)
        names = ["THI", "TBZ", "DQ"]
    else:
        ax.text(0.47, 1.00, "Composition network  $f(x_i)$", transform=T, ha="center",
                va="top", fontsize=6.6, weight="bold")
        outs = _net(ax, [0.28, 0.41, 0.54, 0.67], [6, 5, 4, 4],
                    ["1,290", "256", "64", "4"], 0.77, 0.30,
                    [COLS["blank"], COLS["THI"], COLS["TBZ"], COLS["DQ"]], T)
        names = ["blank", "THI", "TBZ", "DQ"]
    for y, name in zip(outs, names):
        ax.text(0.695, y, name, transform=T, ha="left", va="center", fontsize=5.4,
                color=COLS[name] if name != "blank" else "0.45")
    _arrow(ax, (0.79, 0.77), (0.85, 0.77), T)
    # 출력 1: 조성 막대
    bx, by, bw, bh = 0.87, 0.66, 0.045, 0.22
    for frac, name, btm in ((0.36, "DQ", 0.0), (0.30, "TBZ", 0.36), (0.34, "THI", 0.66)):
        ax.add_patch(Rectangle((bx, by + btm * bh), bw, frac * bh, transform=T,
                               fc=COLS[name], ec="white", lw=0.4))
    ax.text(bx + bw / 2, by + bh + 0.02, "Solution\ncomposition (%)", transform=T,
            ha="center", va="bottom", fontsize=5.8, linespacing=1.2)
    ax.text(0.82, 0.80, "map\nmean", transform=T, ha="center", va="bottom", fontsize=5.2,
            color="0.4", linespacing=1.1)

    # ── 가운데: 맵 특징 12개로 넘어감 ────────────────────────────────────────
    ax.plot([0.54, 0.54, 0.12, 0.12], [0.535, 0.49, 0.49, 0.23], "-", color="0.3", lw=0.8,
            transform=T, solid_joinstyle="round")
    _arrow(ax, (0.12, 0.23), (0.24, 0.23), T)
    ax.text(0.33, 0.49, "map pooling: 12 features", transform=T, ha="center",
            va="center", fontsize=5.6, color="0.3",
            bbox=dict(fc="white", ec="none", pad=1.0))

    # ── 아래: 농도망 ─────────────────────────────────────────────────────────
    ax.text(0.47, 0.44, "Concentration network  $g(m)$", transform=T, ha="center",
            va="top", fontsize=6.6, weight="bold")
    outs = _net(ax, [0.28, 0.41, 0.54, 0.67], [5, 5, 4, 3], ["12", "128", "32", "3"],
                0.23, 0.26, [COLS["THI"], COLS["TBZ"], COLS["DQ"]], T)
    _arrow(ax, (0.71, 0.23), (0.85, 0.23), T)
    ax.text(0.78, 0.255, r"$\Delta\log_{10}C$", transform=T, ha="center", va="bottom",
            fontsize=5.6)
    ax.add_patch(FancyBboxPatch((0.87, 0.15), 0.115, 0.16, boxstyle="round,pad=0.01",
                                transform=T, fc="#eef5ee", ec="0.45", lw=0.6))
    ax.text(0.9275, 0.23, "Apparent\nconc.\n(µM)", transform=T, ha="center", va="center",
            fontsize=5.6, linespacing=1.2)
    ax.text(0.99, -0.03, r"$C = C_{\mathrm{cal}}\cdot10^{\Delta}$,  $C_{\mathrm{cal}}$ from (a)",
            transform=T, ha="right", va="center", fontsize=5.6, color="0.3")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})
    for learner, name in (("mlp", "fig3f_network.png"), ("pls", "fig3f_network_pls.png")):
        fig = plt.figure(figsize=(3.5, 3.1))
        ax = fig.add_axes([0.03, 0.06, 0.94, 0.83])
        draw(ax, learner)
        fig.text(0.03, 0.97, "f", fontsize=11, weight="bold", va="top")
        out = f"{HERE}/{name}"
        fig.savefig(out, dpi=300); plt.close(fig)
        print("저장:", out)
