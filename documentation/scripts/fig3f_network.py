"""Figure 3f 초안 — 알고리즘 도식. 두 판.

  "pls" (논문 모델) — 신경망이 아니다. 상자 두 개.
    위 줄  픽셀 스펙트럼 x_i (1,290점) → PLS 회귀(잠재변수 8개) → DQ / TBZ / THI / blank 점수
           → 0 아래 자르고 합 1 로 → 맵 평균 → Solution composition (%)
    아래 줄 맵 평균 스펙트럼 x̄ (지문 영역 500–2000 cm⁻¹) ÷ 잉크 밴드 I_ink (2137 cm⁻¹)
           → PLS 회귀(잠재변수 8개)  log10 C_j = PLS[ ln(1 + x̄) − ln(1 + I_ink) ]  → Apparent concentration (µM)
           (조성 몫은 입력이 아니다 — 2026-11-11 채택)

  "mlp" (배포 모델) — 신경망 그림체.
    위 줄  조성망  f(x_i): 1,290 → 256 → 64 → 4 → 맵 평균 → Solution composition (%)
    아래 줄 농도망  g(m): 맵 특징 12 → 128 → 32 → 3 = Δlog10C,  C = C_cal · 10^Δ

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


def _map(ax, T):
    """입력: SERS 맵 → NNLS 게이트를 통과한 픽셀."""
    rng = np.random.default_rng(3)
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


def _composition_bar(ax, T):
    bx, by, bw, bh = 0.87, 0.66, 0.045, 0.22
    for frac, name, btm in ((0.36, "DQ", 0.0), (0.30, "TBZ", 0.36), (0.34, "THI", 0.66)):
        ax.add_patch(Rectangle((bx, by + btm * bh), bw, frac * bh, transform=T,
                               fc=COLS[name], ec="white", lw=0.4))
    ax.text(bx + bw / 2, by + bh + 0.02, "Solution\ncomposition (%)", transform=T,
            ha="center", va="bottom", fontsize=5.8, linespacing=1.2)


def _conc_box(ax, T):
    ax.add_patch(FancyBboxPatch((0.87, 0.15), 0.115, 0.16, boxstyle="round,pad=0.01",
                                transform=T, fc="#eef5ee", ec="0.45", lw=0.6))
    ax.text(0.9275, 0.23, "Apparent\nconc.\n(µM)", transform=T, ha="center", va="center",
            fontsize=5.6, linespacing=1.2)


_BANDS = (("TBZ", 1270.0), ("THI", 1367.0), ("DQ", 1570.0))


def _spectrum(ax, x0, y0, w, h, T, bands=False, ink=False):
    """작은 스펙트럼 그림(개형). bands=True 면 표지 밴드 셋을 성분 색으로, ink=True 면 잉크 밴드(2137)를 표시."""
    x = np.linspace(500, 2500, 400)
    peaks = ((700, 0.35, 18), (1010, 0.55, 16), (1270, 0.62, 16), (1367, 0.80, 16),
             (1570, 0.95, 18), (1650, 0.40, 16), (2137, 0.30, 20))
    y = sum(a * np.exp(-0.5 * ((x - c) / s_) ** 2) for c, a, s_ in peaks) + 0.04
    X = x0 + w * (x - 500) / 2000; Y = y0 + h * y / 1.05
    ax.plot([x0, x0 + w], [y0, y0], "-", color="0.45", lw=0.5, transform=T)
    ax.plot(X, Y, "-", color="0.25", lw=0.6, transform=T, zorder=2)
    if bands:
        for name, c in _BANDS:
            m = np.abs(x - c) <= 45
            ax.fill_between(X[m], y0, Y[m], color=COLS[name], lw=0, transform=T, zorder=1)
    if ink:
        m = np.abs(x - 2137.0) <= 50
        ax.fill_between(X[m], y0, Y[m], color="#f2b54a", lw=0, transform=T, zorder=1)
        ax.text(x0 + w * (2137 - 500) / 2000, y0 + h * 0.42, r"$I_{\mathrm{ink}}$", transform=T,
                ha="center", va="bottom", fontsize=5.4, color="0.3")


def _box(ax, x0, y0, w, h, T):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.008", transform=T,
                                fc=NODE, ec=EDGE, lw=0.6))


def _draw_pls(ax):
    """논문 모델: PLS 조성 + PLS 회귀 농도(스펙트럼 ÷ 잉크 밴드). 신경망 그림체를 쓰지 않는다."""
    T = ax.transAxes
    _map(ax, T)
    _arrow(ax, (0.17, 0.77), (0.215, 0.77), T)

    # ── 위: 조성 — 픽셀 스펙트럼 → PLS → 4개 점수 → 맵 평균 ─────────────────
    ax.text(0.47, 1.00, "Composition: PLS regression", transform=T, ha="center",
            va="top", fontsize=6.6, weight="bold")
    _spectrum(ax, 0.225, 0.70, 0.15, 0.15, T)
    ax.text(0.30, 0.68, "pixel spectrum $x_i$\n1,290 points", transform=T, ha="center",
            va="top", fontsize=5.4, color="0.3", linespacing=1.2)
    _arrow(ax, (0.385, 0.77), (0.43, 0.77), T)
    _box(ax, 0.44, 0.69, 0.165, 0.16, T)
    ax.text(0.5225, 0.77, "PLS\n8 latent\nvariables", transform=T, ha="center", va="center",
            fontsize=5.6, linespacing=1.2)
    _arrow(ax, (0.62, 0.77), (0.665, 0.77), T)
    names = ["blank", "THI", "TBZ", "DQ"]
    ys = 0.77 + 0.20 * (np.arange(4) - 1.5) / 4
    _layer(ax, 0.685, ys, 5.2, [COLS[n_] for n_ in names], T)
    for y, name in zip(ys, names):
        ax.text(0.71, y, name, transform=T, ha="left", va="center", fontsize=5.4,
                color=COLS[name] if name != "blank" else "0.45")
    _arrow(ax, (0.79, 0.77), (0.855, 0.77), T)
    ax.text(0.822, 0.80, "map\nmean", transform=T, ha="center", va="bottom", fontsize=5.2,
            color="0.4", linespacing=1.1)
    _composition_bar(ax, T)

    # ── 가운데: 같은 맵에서 농도 단계로 ─────────────────────────────────────
    ax.plot([0.078, 0.078], [0.585, 0.21], "-", color="0.3", lw=0.8, transform=T)
    _arrow(ax, (0.078, 0.21), (0.135, 0.21), T)

    # ── 아래: 농도 — 맵 평균 스펙트럼 ÷ 잉크 밴드 → PLS 회귀 ────────────────
    ax.text(0.47, 0.44, "Concentration: PLS regression", transform=T, ha="center",
            va="top", fontsize=6.6, weight="bold")
    _spectrum(ax, 0.145, 0.14, 0.23, 0.16, T, ink=True)
    ax.text(0.26, 0.12, "map-mean spectrum\n" r"$\bar{x}\,/\,I_{\mathrm{ink}}$", transform=T,
            ha="center", va="top", fontsize=5.4, color="0.3", linespacing=1.2)
    _arrow(ax, (0.385, 0.21), (0.43, 0.21), T)
    _box(ax, 0.44, 0.13, 0.165, 0.16, T)
    ax.text(0.5225, 0.21, "PLS\n8 latent\nvariables", transform=T, ha="center", va="center",
            fontsize=5.6, linespacing=1.2)
    _arrow(ax, (0.62, 0.21), (0.855, 0.21), T)
    ax.text(0.7375, 0.235, r"$\log_{10}C_j$", transform=T, ha="center", va="bottom", fontsize=5.6)
    _conc_box(ax, T)
    ax.text(0.99, -0.03, "fitted on 3–24 µM mixtures", transform=T, ha="right",
            va="center", fontsize=5.6, color="0.3")


def draw(ax, learner="mlp"):
    """learner="pls": 논문 모델(PLS 조성 + PLS 회귀 농도) · "mlp": 배포 모델(신경망 둘)."""
    ax.set_axis_off(); T = ax.transAxes
    if learner == "pls":
        return _draw_pls(ax)
    _map(ax, T)
    _arrow(ax, (0.17, 0.77), (0.24, 0.77), T)

    # ── 위: 조성망 ───────────────────────────────────────────────────────────
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
    _composition_bar(ax, T)                                   # 출력 1: 조성 막대
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
    _conc_box(ax, T)
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
