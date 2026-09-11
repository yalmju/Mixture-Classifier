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
PO = LinearSegmentedColormap.from_list("po", ["#5b4a8a", "#f7f5ed", "#e8a13a"])   # 사용자 Abundance 팔레트
PALETTES = {"rb": (RB, ["#d9a09b", "#f4f1ec", "#9db6dd"]),
            "po": (PO, ["#a89ec4", "#f4f1ec", "#efc97f"])}
BG = RB
BG3 = PALETTES["rb"][1]


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


def load_pairs_kt(method):
    """27 FINAL 쌍 + known-total 정확도: Cᵢ = 몫ᵢ × C_total 이므로
    accuracy = 존재 성분의 min(pred 몫/true 몫, true/pred) 평균 (28d '선언 총량 하')."""
    out = []
    for c, t, p, _a, sub in load_pairs(method):
        mk = t > 0
        pp = np.maximum(p, 1e-6)
        acc = float(np.mean(np.minimum(pp[mk] / t[mk], t[mk] / pp[mk])))
        out.append((c, t, p, acc, sub))
    return out


def load_pairs_csv(path, method=None):
    """사용자 워크시트(Origin export) 읽기. 두 형식을 받는다.
    (a) 27 형식: DQ_pct_X,TBZ_pct_Y,THI_pct_Z,type(true/pred),condition[,accuracy|band]
    (b) 넓은 형식: condition,DQ_true,TBZ_true,THI_true,DQ_pred,TBZ_pred,THI_pred[,band|accuracy]
    band 열은 0–3 숫자 또는 'within 1.25-fold' 같은 문자열; accuracy 열은 0–1.
    둘 다 없으면 몫-fold(존재 성분 최악)로 밴드를 만든다. 점 색·배경 모두 이 밴드."""
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    if not rows:
        return []
    cols = {k.strip().lower(): k for k in rows[0].keys()}

    def _band_from(r):
        for key in ("band", "class", "fold_band"):
            if key in cols and r[cols[key]].strip():
                v = r[cols[key]].strip().lower()
                if v.isdigit():
                    return int(v)
                for i, n in enumerate(BAND_NAMES):
                    if n.lower().split()[-1].split("-")[0] in v and ("beyond" in v) == (i == 3):
                        return i
                if "beyond" in v:
                    return 3
        if "accuracy" in cols and r[cols["accuracy"]].strip():
            return ("acc", float(r[cols["accuracy"]]))
        return None

    pairs = []
    if "type" in cols:                                   # (a)
        cur = None
        for r in rows:
            ty = r[cols["type"]].strip().lower()
            if ty == "true":
                cur = [r[cols.get("condition", "condition")].strip(),
                       np.array([float(r[cols["dq_pct_x"]]), float(r[cols["tbz_pct_y"]]),
                                 float(r[cols["thi_pct_z"]])]), None, None, "ternary", _band_from(r)]
            elif ty == "pred" and cur is not None:
                cur[2] = np.array([float(r[cols["dq_pct_x"]]), float(r[cols["tbz_pct_y"]]),
                                   float(r[cols["thi_pct_z"]])])
                if cur[5] is None:
                    cur[5] = _band_from(r)
                pairs.append(cur); cur = None
    else:                                                # (b)
        for r in rows:
            t = np.array([float(r[cols["dq_true"]]), float(r[cols["tbz_true"]]), float(r[cols["thi_true"]])])
            p = np.array([float(r[cols["dq_pred"]]), float(r[cols["tbz_pred"]]), float(r[cols["thi_pred"]])])
            pairs.append([r.get(cols.get("condition", ""), "").strip(), t, p, None, "ternary", _band_from(r)])
    out = []
    for c, t, p, _a, sub, b in pairs:
        sub = "binary" if (t <= 0).any() else "ternary"
        if isinstance(b, tuple):                        # raw accuracy → band() decides
            out.append((c, t, p, b[1], sub)); continue
        if b is None:
            b = band_of(worst_fold(t, p))
        # explicit band → encode as an accuracy band() maps back (0.9/0.75/0.55/0.3 under 1/fold)
        out.append((c, t, p, {0: 0.9, 1: 0.75, 2: 0.55, 3: 0.3}[int(b)], sub))
    return out


def is_grid64(cond):
    import re
    m = re.match(r"DQ(\d+)-TB(\d+)-TH(\d+)$", cond)
    return bool(m) and all(int(v) in (3, 6, 12, 24) for v in m.groups())


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
ACC_THRESHOLDS = None     # (t1, t2, t3): accuracy 열을 직접 밴드로 자르는 임계값
FIELD = "fraction"        # 배경: fraction = 정답 비율, mean = accuracy 국소 평균
CORRECT_BAND = 2          # 정답 = 밴드 index ≤ 이 값 (0: 1.25배, 1: 1.5배, 2: 2배)
DISCRETE = False          # 배경 3단 이산색
ALPHA = 0.82
CONTINUOUS = False        # 점 색: 원래 그림처럼 RdYlGn(accuracy 0.4–1) 연속
ACC_NORM = (0.4, 1.0)
BAND_LABEL = {0: "1.25-fold", 1: "1.5-fold", 2: "2-fold"}


def band(t, p, acc):
    """조건의 밴드 — 파일 accuracy 열(기본) 또는 몫-fold(옵션)."""
    if FOLD_FROM_SHARES or not np.isfinite(acc):
        return band_of(worst_fold(t, p))
    if ACC_THRESHOLDS is not None:
        t1, t2, t3 = ACC_THRESHOLDS
        return 0 if acc >= t1 else 1 if acc >= t2 else 2 if acc >= t3 else 3
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
    if FIELD == "mean":
        C = np.array([np.clip(a, 0, 1) if np.isfinite(a) else 0.0 for _, _, _, a, _ in pairs])
    else:
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
                       colors=BG3, alpha=ALPHA, zorder=0)
    else:
        ax.tripcolor(tri, frac, cmap=BG, vmin=0, vmax=1, shading="gouraud",
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
        if CONTINUOUS and np.isfinite(acc):
            _c = plt.get_cmap("RdYlGn")((acc - ACC_NORM[0]) / (ACC_NORM[1] - ACC_NORM[0]))
        else:
            _c = BAND_COLORS[band(t, p, acc)]
        ax.scatter([px], [py], s=26, marker=mk, color=_c,
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
    ax.set_title("" if not title.strip() else
                 f"{title}  ·  within {BAND_LABEL[CORRECT_BAND]} {ok}/{len(pairs)}"
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
    ap.add_argument("--source", choices=["final92", "grid64", "kt"], default="final92",
                    help="final92: 27 FINAL composition overlap (92 conds); "
                         "grid64: 09b uM-head concentration accuracy (64 conds); "
                         "kt: 27 FINAL shares with known-total accuracy (28d legend)")
    ap.add_argument("--subset", choices=["all", "grid64"], default="all",
                    help="restrict conditions to the 3/6/12/24 uM grid")
    ap.add_argument("--palette", choices=list(PALETTES), default="rb")
    ap.add_argument("--acc-thresholds", nargs=3, type=float, default=None,
                    metavar=("T1", "T2", "T3"),
                    help="band the accuracy column at these cut-offs instead of 1/fold "
                         "(e.g. 0.887 0.786 0.5 reproduces the user's MLP figure)")
    ap.add_argument("--field", choices=["fraction", "mean"], default="fraction",
                    help="background: local fraction within --correct-band, or local mean accuracy")
    ap.add_argument("--methods", default="nnls,pls,mlp",
                    help="comma list of repo panels to draw (ignored with --pairs-csv)")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="panel titles for --pairs-csv files (default: file names)")
    ap.add_argument("--pairs-csv", nargs="*", default=None,
                    help="user worksheet(s) instead of the repo sources: one file per "
                         "panel, in the order nnls,pls,mlp (fewer files = fewer panels)")
    ap.add_argument("--continuous", action="store_true",
                    help="point colour = RdYlGn over accuracy 0.4-1 (original look)")
    ap.add_argument("--discrete", action="store_true", help="3-level background")
    ap.add_argument("--alpha", type=float, default=0.82)
    a = ap.parse_args()
    global FOLD_FROM_SHARES, CORRECT_BAND, DISCRETE, ALPHA, CONTINUOUS, BG, BG3, ACC_THRESHOLDS
    DISCRETE = bool(a.discrete); ALPHA = float(a.alpha); CONTINUOUS = bool(a.continuous)
    ACC_THRESHOLDS = tuple(a.acc_thresholds) if a.acc_thresholds else None
    global FIELD
    FIELD = a.field
    BG, BG3 = PALETTES[a.palette]
    FOLD_FROM_SHARES = bool(a.fold_from_shares)
    CORRECT_BAND = {1.25: 0, 1.5: 1, 2.0: 2}[a.correct_band]
    tag = (("_user" if a.pairs_csv else "") + ("_thr" if a.acc_thresholds else "") + ("_mean" if a.field == "mean" else "") + (f"_{a.source}" if a.source != "final92" and not a.pairs_csv else "")
           + (f"_{a.subset}" if a.subset != "all" else "")
           + (f"_{a.palette}" if a.palette != "rb" else "")
           + ("" if CORRECT_BAND == 2 else f"_{BAND_LABEL[CORRECT_BAND].replace('-fold', 'x')}")
           + ("_discrete" if DISCRETE else "") + ("_cont" if CONTINUOUS else ""))
    _base = {"grid64": load_pairs_grid64, "kt": load_pairs_kt}.get(a.source, load_pairs)
    if a.pairs_csv:
        _keys = [f"user{i}" for i in range(len(a.pairs_csv))]
        _files = dict(zip(_keys, a.pairs_csv))
        _base = lambda m_: load_pairs_csv(_files[m_])
    loader = ((lambda m_: [x for x in _base(m_) if is_grid64(x[0])])
              if a.subset == "grid64" else _base)
    _panels = [(k, n) for k, n in (("nnls", "NNLS (surface)"), ("pls", "PLS-R"), ("mlp", "MLP"))
               if k in a.methods.split(",")]
    if a.pairs_csv:
        _names = a.labels or [os.path.splitext(os.path.basename(f))[0] for f in a.pairs_csv]
        _panels = [(f"user{i}", n) for i, n in enumerate(_names)]
    fig, axes = plt.subplots(1, len(_panels), figsize=(4.5 * len(_panels), 4.6),
                             squeeze=False)
    axes = axes[0]
    for ax, (m, title) in zip(axes, _panels):
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
        with open(os.path.join(RES, f"27g_conditions{tag}_{m}.csv"), "w", newline="",
                  encoding="utf-8-sig") as f:
            wri = csv.writer(f)
            wri.writerow(["condition", "subset", "true_DQ", "true_TBZ", "true_THI",
                          "pred_DQ", "pred_TBZ", "pred_THI", "accuracy", "band"])
            for c, t, p_, acc, sub in pairs:
                wri.writerow([c, sub] + [f"{v:.2f}" for v in t] + [f"{v:.2f}" for v in p_]
                             + [f"{acc:.4f}", BAND_NAMES[band(t, p_, acc)]])
    handles = [Line2D([], [], marker=">", color="#8a919b", lw=0.8, label="Prediction"),
               Line2D([], [], marker="o", ls="none", mfc="white", mec="#8a919b",
                      label="True composition (ternary)"),
               Line2D([], [], marker="s", ls="none", mfc="white", mec="#8a919b",
                      label="True composition (binary)")]
    if not CONTINUOUS:
        handles += [Line2D([], [], marker="o", ls="none", color=c, label=n)
                    for c, n in zip(BAND_COLORS, BAND_NAMES)]
    axes[-1].legend(handles=handles, loc="upper right", bbox_to_anchor=(1.34, 1.0),
                    frameon=False, fontsize=8)
    if CONTINUOUS:
        _sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(*ACC_NORM))
        _cb = fig.colorbar(_sm, ax=axes[-1], orientation="vertical", fraction=0.05,
                           pad=0.02, shrink=0.45, anchor=(0.0, 0.0))
        _cb.set_label("accuracy" + (" = mean min(pred/true, true/pred)" if a.source == "grid64"
                                    else " = 1 − ½Σ|Δshare|"), fontsize=7)
        _cb.ax.tick_params(labelsize=7)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    sm = (plt.cm.ScalarMappable(cmap=ListedColormap(BG3),
                                norm=BoundaryNorm([0, 1 / 3, 2 / 3, 1], 3)) if DISCRETE
          else plt.cm.ScalarMappable(cmap=BG, norm=plt.Normalize(0, 1)))
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.035,
                      pad=0.02, aspect=40)
    cb.set_label((f"local mean accuracy (kernel σ = {a.sigma:.0f} %p)" if a.field == "mean" else
                  f"local fraction within {BAND_LABEL[CORRECT_BAND]} (kernel σ = {a.sigma:.0f} %p)")
                 + {"grid64": "  ·  µM-head concentration accuracy",
                    "kt": "  ·  concentration accuracy under declared total",
                    "final92": "  ·  composition overlap"}[a.source]
                 + (", 64-grid" if (a.subset == "grid64" or a.source == "grid64") else ", 92 conditions"),
                 fontsize=8)
    cb.set_ticks([0, 0.5, 1]); cb.ax.tick_params(labelsize=7)
    out = os.path.join(RES, f"27g_ternary_rb{tag}.png")
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
