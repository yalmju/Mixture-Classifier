# -*- coding: utf-8 -*-
"""27g — 삼각도(true → prediction 화살표, 28d 문법)에 정답/오답 배경(R/B).

입력(--source):
  final92 : 27_ternary_truepred_FINAL_{nnls,pls,mlp}.csv — 92조건, accuracy 열 =
            조성 겹침률 1 − ½Σ|Δ몫| (PLS ≥ MLP 로 나오는 지표)
  grid64  : 09b_grid64_concentration_predictions_long.csv — 64격자 µM 헤드,
            accuracy = 성분별 min(pred/true, true/pred) 평균 (원래 28_concAcc 그림;
            NNLS 0.40 / PLS 0.64 / MLP 0.68)
밴드: 기본은 파일의 accuracy 열(0–1)을 1/fold 로 읽어 ≥0.8 / ≥0.667 / ≥0.5 / 미만
      = 1.25 / 1.5 / 2배 / 초과. --fold-from-shares 를 주면 몫에서 직접 계산
      (존재 성분의 최악 fold = pred 몫 / true 몫). 점 색과 배경은 같은 판정을 쓴다.
정답 정의: 밴드 ≤ 2배 — 그림의 빨간 점이 오답.
배경: 조성 공간(3-벡터, %p)에서 가우시안 커널(σ = SIGMA %p)로 평활한 국소 정답률
      0 = 빨강(오답) … 1 = 파랑(정답). 데이터에서 먼 곳은 투명해진다.
출력: 27g_ternary_rb.png (NNLS | PLS-R | MLP),
      27g_ternary_correct_fraction_{m}_XYZZ.csv (Origin ternary contour 용:
      X = DQ %, Y = TBZ %, Z = THI %, Z2 = 국소 정답률; alpha 열 = 데이터 밀도)

실행:  python -u fig27g_ternary_rb_background.py [--sigma 12] [--correct-band 1.25|1.5|2]
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import labfig  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402

RES = os.path.join(HERE, "..", "results")
CO = labfig.CO
SUBS = ["DQ", "TBZ", "THI"]
BAND_COLORS = ["#1f9d55", "#e8c33a", "#ee7f2d", "#d62728"]   # 1.25 / 1.5 / 2 / beyond
BAND_NAMES = ["within 1.25-fold", "within 1.5-fold", "within 2-fold", "Beyond 2-fold"]
RB = LinearSegmentedColormap.from_list("rb", ["#c8322b", "#f6f1ec", "#2b64b5"])


def load_pairs(method):
    """(condition, true %, pred %, accuracy) — 27 FINAL 파일의 true/pred 행 쌍."""
    path = os.path.join(RES, f"27_ternary_truepred_FINAL_{method}.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    pairs, cur = [], None
    for r in rows:
        if r["type"] == "true":
            cur = [r["condition"], np.array([float(r["DQ_pct_X"]), float(r["TBZ_pct_Y"]),
                                             float(r["THI_pct_Z"])]), None,
                   float(r["accuracy"]) if r.get("accuracy", "").strip() else float("nan"),
                   r.get("subset", "").strip()]
        elif r["type"] == "pred" and cur is not None:
            cur[2] = np.array([float(r["DQ_pct_X"]), float(r["TBZ_pct_Y"]),
                               float(r["THI_pct_Z"])])
            pairs.append(tuple(cur)); cur = None
    return pairs


def worst_fold(t, p):
    """존재 성분(true > 0)의 |log2(pred/true)| 최댓값 (몫 = 선언 총량 하 농도)."""
    m = t > 0
    return float(np.max(np.abs(np.log2(np.maximum(p[m], 1e-3) / t[m]))))


def load_pairs_grid64(method):
    """64격자 µM 헤드 예측(09b): 위치 = µM 을 몫으로 정규화, accuracy =
    성분별 min(pred/true, true/pred) 평균 (= 09 conc_acc, 원래 28_concAcc 그림)."""
    import collections
    col = {"nnls": "nnls_Ccal_uM", "pls": "pls_uM", "mlp": "mlp_uM"}[method]
    rows = list(csv.DictReader(open(os.path.join(RES,
        "09b_grid64_concentration_predictions_long.csv"), encoding="utf-8-sig")))
    conds = collections.OrderedDict()
    for r in rows:
        conds.setdefault(r["condition"], {})[r["component"]] = r
    pairs = []
    for c, d in conds.items():
        t = np.array([float(d[s_]["true_uM"]) for s_ in SUBS])
        pm = np.maximum(np.array([float(d[s_][col]) for s_ in SUBS]), 1e-6)
        acc = float(np.mean(np.minimum(pm / t, t / pm)))
        pairs.append((c, t / t.sum() * 100, pm / pm.sum() * 100, acc, "ternary"))
    return pairs


def band_of(f):
    """log2 fold → 0..3 밴드 (1.25 / 1.5 / 2배 / 초과)."""
    return 0 if f <= np.log2(1.25) else 1 if f <= np.log2(1.5) else 2 if f <= 1.0 else 3


FOLD_FROM_SHARES = False
CORRECT_BAND = 2          # 정답 = 밴드 index ≤ 이 값 (0: 1.25배, 1: 1.5배, 2: 2배)
DISCRETE = False          # 배경 3단 이산색
ALPHA = 0.82
BAND_LABEL = {0: "1.25-fold", 1: "1.5-fold", 2: "2-fold"}


def band(t, p, acc):
    """조건의 밴드 — 파일 accuracy 열(기본) 또는 몫-fold(옵션)."""
    if FOLD_FROM_SHARES or not np.isfinite(acc):
        return band_of(worst_fold(t, p))
    return band_of(-np.log2(max(acc, 1e-6)))


def to_xy(c):
    """(DQ, TBZ, THI) % → 정삼각형 좌표: TBZ 좌하, DQ 우하, THI 꼭대기."""
    c = np.asarray(c, float) / 100.0
    x = c[..., 0] + 0.5 * c[..., 2]
    y = c[..., 2] * np.sqrt(3) / 2
    return x, y


def field(pairs, sigma):
    step = 1.0
    grid = [(d, t, 100 - d - t) for d in np.arange(0, 100 + step, step)
            for t in np.arange(0, 100 + step - d, step)]
    G = np.array(grid, float)
    T = np.array([t for _, t, _, _, _ in pairs])
    C = np.array([1.0 if band(t, p, a) <= CORRECT_BAND else 0.0 for _, t, p, a, _ in pairs])
    d2 = ((G[:, None, :] - T[None, :, :]) ** 2).sum(-1)
    w = np.exp(-0.5 * d2 / sigma ** 2)
    dens = w.sum(1)
    frac = (w * C[None, :]).sum(1) / np.maximum(dens, 1e-12)
    return G, frac, dens


def draw(ax, method, pairs, sigma, title):
    G, frac, dens = field(pairs, sigma)
    gx, gy = to_xy(G)
    tri = mtri.Triangulation(gx, gy)
    # 데이터에서 먼 곳(밀도 < 최대의 5 %)은 투명 → 배경이 근거 없는 색을 안 만든다
    alpha_pt = np.clip(dens / (0.05 * dens.max()), 0, 1)
    tri.set_mask(alpha_pt[tri.triangles].min(1) < 0.2)
    if DISCRETE:
        # 3단: < 1/3 빨강 · 1/3–2/3 흰 · > 2/3 파랑 — 얼룩 대신 영역
        ax.tricontourf(tri, frac, levels=[0, 1 / 3, 2 / 3, 1.0001],
                       colors=["#d9a09b", "#f4f1ec", "#9db6dd"], alpha=ALPHA, zorder=0)
    else:
        ax.tripcolor(tri, frac, cmap=RB, vmin=0, vmax=1, shading="gouraud",
                     alpha=ALPHA, zorder=0, rasterized=True)
    vx, vy = to_xy(np.array([[0, 100, 0], [100, 0, 0], [0, 0, 100], [0, 100, 0]]))
    ax.plot(vx, vy, color="#30343a", lw=1.2, zorder=3)
    for k in range(20, 100, 20):
        for a, b in (((k, 100 - k, 0), (k, 0, 100 - k)),
                     ((100 - k, k, 0), (0, k, 100 - k)),
                     ((0, 100 - k, k), (100 - k, 0, k))):
            x, y = to_xy(np.array([a, b], float))
            ax.plot(x, y, color="#c9ced4", lw=0.35, zorder=1)
    for _, t, p, acc, sub in pairs:
        tx, ty = to_xy(t); px, py = to_xy(p)
        mk = "s" if sub == "binary" else "o"          # binary(부재 성분 있음) = 사각
        ax.annotate("", xy=(px, py), xytext=(tx, ty),
                    arrowprops=dict(arrowstyle="-|>", color="#8a919b", lw=0.7,
                                    shrinkA=3, shrinkB=3), zorder=4)
        ax.scatter([tx], [ty], s=26, marker=mk, facecolor="white", edgecolor="#8a919b",
                   linewidth=0.9, zorder=5)
        ax.scatter([px], [py], s=26, marker=mk, color=BAND_COLORS[band(t, p, acc)],
                   edgecolor="white", linewidth=0.5, zorder=6)
    ax.text(*to_xy(np.array([0, 0, 100.0])), "THI", color=CO["THI"], ha="center",
            va="bottom", fontsize=11, weight="bold")
    ax.text(*to_xy(np.array([0, 100.0, 0])), "TBZ ", color=CO["TBZ"], ha="right",
            va="top", fontsize=11, weight="bold")
    ax.text(*to_xy(np.array([100.0, 0, 0])), " DQ", color=CO["DQ"], ha="left",
            va="top", fontsize=11, weight="bold")
    ok = sum(band(t, p, a) <= CORRECT_BAND for _, t, p, a, _ in pairs)
    okb = sum(band(t, p, a) <= CORRECT_BAND for _, t, p, a, sb in pairs if sb == "binary")
    nb_ = sum(1 for *_, sb in pairs if sb == "binary")
    ax.set_title(f"{title}  ·  within {BAND_LABEL[CORRECT_BAND]} {ok}/{len(pairs)}"
                 + (f"  (binary {okb}/{nb_})" if nb_ else ""), fontsize=9,
                 color="#3f454c")
    ax.set_xlim(-0.08, 1.08); ax.set_ylim(-0.08, 0.95)
    ax.set_aspect("equal"); ax.set_axis_off()
    return G, frac, dens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=12.0, help="kernel width, %p")
    ap.add_argument("--fold-from-shares", action="store_true",
                    help="bands from pred/true share fold instead of the accuracy column")
    ap.add_argument("--correct-band", type=float, default=2.0, choices=[1.25, 1.5, 2.0],
                    help="a condition counts as correct when within this fold")
    ap.add_argument("--source", choices=["final92", "grid64"], default="final92",
                    help="final92: 27 FINAL composition overlap (92 conds); "
                         "grid64: 09b uM-head concentration accuracy (64 conds)")
    ap.add_argument("--discrete", action="store_true", help="3-level background")
    ap.add_argument("--alpha", type=float, default=0.82)
    a = ap.parse_args()
    global FOLD_FROM_SHARES, CORRECT_BAND, DISCRETE, ALPHA
    DISCRETE = bool(a.discrete); ALPHA = float(a.alpha)
    FOLD_FROM_SHARES = bool(a.fold_from_shares)
    CORRECT_BAND = {1.25: 0, 1.5: 1, 2.0: 2}[a.correct_band]
    tag = ((f"_{a.source}" if a.source != "final92" else "")
           + ("" if CORRECT_BAND == 2 else f"_{BAND_LABEL[CORRECT_BAND].replace('-fold', 'x')}")
           + ("_discrete" if DISCRETE else ""))
    loader = load_pairs_grid64 if a.source == "grid64" else load_pairs
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, (m, title) in zip(axes, (("nnls", "NNLS (surface)"), ("pls", "PLS-R"),
                                     ("mlp", "MLP"))):
        pairs = loader(m)
        G, frac, dens = draw(ax, m, pairs, a.sigma, title)
        with open(os.path.join(RES, f"27g_ternary_correct_fraction{tag}_{m}_XYZZ.csv"),
                  "w", newline="", encoding="utf-8-sig") as f:
            wri = csv.writer(f)
            wri.writerow(["DQ_pct_X", "TBZ_pct_Y", "THI_pct_Z",
                          f"correct_fraction_{m}_Z2", "data_density_alpha"])
            for g, fr, dn in zip(G, frac, dens / dens.max()):
                wri.writerow([f"{g[0]:.0f}", f"{g[1]:.0f}", f"{g[2]:.0f}",
                              f"{fr:.4f}", f"{dn:.4f}"])
        print(f"{m}: {len(pairs)} pairs · field mean {frac[dens > 0.05 * dens.max()].mean():.2f}")
    handles = [Line2D([], [], marker=">", color="#8a919b", lw=0.8, label="Prediction"),
               Line2D([], [], marker="o", ls="none", mfc="white", mec="#8a919b",
                      label="True composition (ternary)"),
               Line2D([], [], marker="s", ls="none", mfc="white", mec="#8a919b",
                      label="True composition (binary)")]
    handles += [Line2D([], [], marker="o", ls="none", color=c, label=n)
                for c, n in zip(BAND_COLORS, BAND_NAMES)]
    axes[2].legend(handles=handles, loc="upper right", bbox_to_anchor=(1.34, 1.0),
                   frameon=False, fontsize=8)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    sm = (plt.cm.ScalarMappable(cmap=ListedColormap(["#d9a09b", "#f4f1ec", "#9db6dd"]),
                                norm=BoundaryNorm([0, 1 / 3, 2 / 3, 1], 3)) if DISCRETE
          else plt.cm.ScalarMappable(cmap=RB, norm=plt.Normalize(0, 1)))
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.035,
                      pad=0.02, aspect=40)
    cb.set_label(f"local fraction within {BAND_LABEL[CORRECT_BAND]} (kernel σ = {a.sigma:.0f} %p)"
                 + ("  ·  µM-head concentration accuracy, 64-grid" if a.source == "grid64"
                    else "  ·  composition overlap, 92 conditions"),
                 fontsize=8)
    cb.set_ticks([0, 0.5, 1]); cb.ax.tick_params(labelsize=7)
    out = os.path.join(RES, f"27g_ternary_rb{tag}.png")
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
