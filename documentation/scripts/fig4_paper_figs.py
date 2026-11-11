# -*- coding: utf-8 -*-
"""Fig 4 패널 — 논문용 PLS 모델 기준 (원천: fig4_paper_data.py 가 만든 heldout_maps.csv · lettering_pixels.npz).

  fig4b_ternary      64 조건(전부 held-out) — Band intensity | PLS 삼각도 + fold 구간 막대 (본문, 11-10 사용자 결정)
                     왼쪽 = 특성 밴드 세기 + 단일성분 검량선(통상적 SERS 정량, 직선 0.1–10 µM) — "신호 세기만 보면" 의 판독.
                     NNLS 판은 SI: fig4b_ternary_nnls (파이프라인 NNLS, 배경 BLK·INK·LEAF, 열 NNLS) ·
                     fig4b_ternary_nnls_blk (순수 3성분 + 잉크 blank BLK 하나, 열 NNLSb)
  fig4c_lettering    글씨맵 — (i) 밴드 세기  (ii) 모델 조성 (NNLS 게이트 0.15 통과 픽셀만)
  eq_series          등몰 혼합물 held-out 판독 (3–333 µM/성분) — 33 은 읽히고 100 은 안 읽힌다
  fig4_composition_64.csv    64 조건의 뿌린 조성 · 밴드 세기 판독 · PLS 판독 · 조성 오차 · 막대의 fold 구간
  fig4_concentration_64.csv  64 조건의 뿌린 µM · PLS 겉보기 µM · 배수 오차
  summary.txt        그림에 들어간 수치 전부

PLS 판독은 맵 파일을 판독 절차 그대로(전처리 → 게이트 → 조성 → 농도) 읽은 held-out 값이다(fig4_paper_data.py).

정의(10-09 Fig 4b 와 동일): Ĉ = 판독 비율 × 선언 총량, 정확도 = 성분 평균 min(Ĉ/C, C/Ĉ), fold = 1/정확도.
실행:  python documentation/scripts/fig4_paper_figs.py
"""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "fig4_paper_pls_2026-11-10")
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "svg.fonttype": "none", "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5})
COMP = ("DQ", "TBZ", "THI")
COL = {"DQ": "#1f78c8", "TBZ": "#78c240", "THI": "#e3425f"}
BANDC = ["#1a9850", "#f2c14e", "#ef8a3c", "#d73027"]
BANDL = ["≤1.25×", "≤1.5×", "≤2×", ">2×"]
R3 = np.sqrt(3) / 2
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def load():
    with open(os.path.join(OUT, "heldout_maps.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    g = lambda keys: np.array([[float(r[k]) for k in keys] for r in rows])  # noqa: E731
    d = {"name": [r["condition"] for r in rows], "split": np.array([r["split"] for r in rows]),
         "grid": np.array([int(r["grid64"]) for r in rows], bool), "T": g([f"{k}_true_uM" for k in COMP])}
    for m in ("NNLS", "NNLSb", "NNLSm", "PLS", "MLP"):
        d[m] = g([f"{k}_{m}" for k in COMP])
    d["band_h"] = g([f"{k}_band" for k in COMP])
    d["UM"] = np.array([[float(r[f"{k}_PLS_uM"]) if r[f"{k}_PLS_uM"] != "" else np.nan for k in COMP] for r in rows])
    d["gate"] = np.array([int(r["gate_px"]) for r in rows])
    d["npx"] = np.array([int(r["n_px"]) for r in rows])
    return d


# ------------------------------------------------------------------ 밴드 세기 + 단일성분 검량선 (통상적 SERS 정량)
BAND_CAL = "lin10"      # 대표 검량: 0.1–10 µM 직선(Fig 3a 의 working range 와 같은 점). 다른 둘은 견고성 확인용


def _pava(y):
    """단조 증가 회귀(pool-adjacent-violators) — 검량 곡선을 그대로 읽을 때 잡음으로 뒤집힌 점만 편다."""
    v, w = [], []
    for x in map(float, y):
        v.append(x); w.append(1.0)
        while len(v) > 1 and v[-2] > v[-1]:
            a = (v[-2] * w[-2] + v[-1] * w[-1]) / (w[-2] + w[-1])
            w[-2:] = [w[-2] + w[-1]]; v[-2:] = [a]
    return np.repeat(v, np.asarray(w, int))


def band_readers(d):
    """특성 밴드 높이 → 단일성분 검량선 → µM → 3성분 비율. d["BAND"](대표)·d["BAND_<검량>"]·d["BANDuM_<검량>"] 을 채운다.
      lin10  I = a + b·C, 0.1–10 µM 5점 최소제곱 (Fig 3a 의 working range)
      lin50  같은 식, 0.1–50 µM 6점 (혼합물 3–24 µM 을 외삽 없이 덮는다)
      curve  검량 곡선 전체(0.1–1000 µM)를 log C 축에서 구간 선형 보간 — 식 가정 없음"""
    with open(os.path.join(OUT, "band_calibration.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    H = d["band_h"]
    out = {k: np.zeros_like(H) for k in ("lin10", "lin50", "curve")}
    for i, s in enumerate(COMP):
        C = np.array([float(r["C_uM"]) for r in rows if r["compound"] == s])
        I = np.array([float(r["band_height"]) for r in rows if r["compound"] == s])
        o = np.argsort(C)
        C, I = C[o], I[o]
        for key, cmax in (("lin10", 10.0), ("lin50", 50.0)):
            m = C <= cmax + 1e-9
            b, a = np.polyfit(C[m], I[m], 1)
            r2 = np.corrcoef(C[m], I[m])[0, 1] ** 2
            out[key][:, i] = np.clip((H[:, i] - a) / b, 0, None)
            say(f"[band-cal] {s} {key}: I = {a:.0f} + {b:.1f}·C (n = {int(m.sum())}, r² = {r2:.3f})")
        Im = _pava(I)
        out["curve"][:, i] = 10 ** np.interp(H[:, i], Im + np.arange(len(Im)) * 1e-6, np.log10(C))
        say(f"[band-cal] {s} 밴드 높이 {np.round(I).astype(int).tolist()} @ {C.tolist()} µM")
    for key, U in out.items():
        tot = U.sum(1, keepdims=True)
        d[f"BANDuM_{key}"] = U
        d[f"BAND_{key}"] = np.where(tot > 0, U / np.clip(tot, 1e-12, None), 1 / 3)
        g = d["grid"]
        T = d["T"][g]
        fold, acc = fold_of(d[f"BAND_{key}"][g], T)
        c = np.bincount(np.searchsorted([1.25, 1.5, 2.0], fold, side="left"), minlength=4)
        dev = 50 * np.abs(d[f"BAND_{key}"][g] - T / T.sum(1, keepdims=True)).sum(1)
        ua = np.minimum(U[g] / T, T / np.clip(U[g], 1e-12, None))
        say(f"[band] {key}: 조성 오차 {dev.mean():.1f} %p · fold 구간 {c.tolist()} · 2배 이내 {int(c[:3].sum())}/64 · "
            f"정확도 {acc.mean():.3f} · 0 µM 로 읽힌 성분 {int((U[g] == 0).sum())}/192 · "
            f"(총량 선언 없이) 검량선 µM 그대로: 2배 이내 성분 {int((ua >= 0.5).sum())}/192 · "
            f"중앙 Ĉ/C DQ/TBZ/THI {np.round(np.median(U[g] / T, 0), 2).tolist()}")
    d["BAND"] = d[f"BAND_{BAND_CAL}"]


def fold_of(share, T):
    C = share * T.sum(1, keepdims=True)
    acc = np.minimum(C / T, T / np.clip(C, 1e-12, None)).mean(1)
    return 1 / np.clip(acc, 1e-12, None), acc


def xy(c):
    c = np.asarray(c, float)
    c = c / c.sum(-1, keepdims=True)
    return c[..., 0] + c[..., 2] / 2, c[..., 2] * R3


# ------------------------------------------------------------------ (b) ternary + fold bars
MAIN_NNLS = "NNLS"      # NNLS 를 말할 때의 기준 정의 = 파이프라인 NNLS (eq_series 왼쪽 패널, SI 삼각도)


def fig_b(d, nn=MAIN_NNLS, name="fig4b_ternary", lab0="NNLS"):
    """nn="NNLS": 파이프라인 NNLS(배경 BLK·INK·LEAF 전부, 게이트와 같은 계산) — 본문용(11-10 최종 결정).
    nn="NNLSb": 순수 3성분 + 잉크 blank(BLK) 하나만 넣은 NNLS — SI 용.
    nn="NNLSm": 배경 템플릿 없이 모델 내장 3성분만 — 수치만 기록(그림 없음)."""
    k = d["grid"]
    T = d["T"][k]
    assert len(T) == 64 and len({tuple(t) for t in T}) == 64, "64 조건이 아니다"
    methods = [(nn, lab0), ("PLS", "PLS model")]
    fig = plt.figure(figsize=(7.2, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.12, left=0.01, right=0.99, top=0.9, bottom=0.14)
    counts = {}
    for j, (key, lab) in enumerate(methods):
        S = d[key][k]
        fold, acc = fold_of(S, T)
        band = np.searchsorted([1.25, 1.5, 2.0], fold, side="left")
        counts[key] = np.bincount(band, minlength=4)
        dev = 50 * np.abs(S - T / T.sum(1, keepdims=True)).sum(1)
        rng = np.random.default_rng(0)
        bs = np.array([acc[rng.integers(0, 64, 64)].mean() for _ in range(4000)])
        say(f"[b:{name}] {key}: fold 구간 {counts[key].tolist()} · 2배 이내 {int(counts[key][:3].sum())}/64 · "
            f"정확도 {acc.mean():.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}] · "
            f"조성 오차 {dev.mean():.1f} %p · 평균 판독 DQ/TBZ/THI {np.round(S.mean(0) * 100, 1).tolist()}")
        ax = fig.add_subplot(gs[0, j])
        for f in (0.2, 0.4, 0.6, 0.8):                      # 20 % 격자
            for a, b in (([f, 1 - f, 0], [f, 0, 1 - f]), ([1 - f, f, 0], [0, f, 1 - f]), ([1 - f, 0, f], [0, 1 - f, f])):
                (x0, y0), (x1, y1) = xy(a), xy(b)
                ax.plot([x0, x1], [y0, y1], color="#e3e5ea", lw=0.4, zorder=0)
        ax.plot([0, 1, 0.5, 0], [0, 0, R3, 0], color="#3a3f4a", lw=0.7, zorder=1)
        xt, yt = xy(T)
        xp, yp = xy(S)
        ax.scatter(xt, yt, s=7, facecolor="white", edgecolor="#8a8f99", lw=0.5, zorder=2)
        for b_ in (3, 2, 1, 0):
            ax.scatter(xp[band == b_], yp[band == b_], s=10, color=BANDC[b_], edgecolor="none", zorder=3)
        ax.annotate("", xy=(xp.mean(), yp.mean()), xytext=(xt.mean(), yt.mean()),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0, mutation_scale=7), zorder=5)
        ax.text(0.5, R3 + 0.035, "THI", ha="center", va="bottom", color=COL["THI"], fontweight="bold")
        ax.text(-0.02, -0.1, "TBZ", ha="center", va="center", color=COL["TBZ"], fontweight="bold")
        ax.text(1.02, -0.1, "DQ", ha="center", va="center", color=COL["DQ"], fontweight="bold")
        ax.text(-0.12, R3 + 0.06, lab, ha="left", va="center", fontsize=8)
        ax.set(xlim=(-0.14, 1.12), ylim=(-0.15, R3 + 0.1), aspect="equal")
        ax.axis("off")
    ax = fig.add_subplot(gs[0, 2])
    pos = ax.get_position()
    ax.set_position([pos.x0 + 0.06, pos.y0 + 0.2, pos.width - 0.13, pos.height * 0.42])
    for i, (key, lab) in enumerate(methods):
        left = 0
        for b_ in range(4):
            n = counts[key][b_]
            ax.barh(i, n, left=left, color=BANDC[b_], height=0.62, edgecolor="white", lw=0.5)
            if n >= 3:
                ax.text(left + n / 2, i, str(n), ha="center", va="center", fontsize=6.5,
                        color="white" if b_ in (0, 3) else "black")
            left += n
        ax.text(66, i, f"{int(counts[key][:3].sum())}", ha="left", va="center", fontsize=7)
    ax.text(66, -0.75, "within 2×", ha="left", va="center", fontsize=6.5)
    ax.set(yticks=range(len(methods)), yticklabels=[m[1] for m in methods], xlim=(0, 64), ylim=(len(methods) - 0.5, -0.5),
           xticks=[0, 16, 32, 48, 64], xlabel="Mixtures (n = 64)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in BANDC]
    ax.legend(hs, BANDL, title="Fold error", ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.28),
              handlelength=0.9, handletextpad=0.4, columnspacing=0.9, fontsize=6.5, title_fontsize=6.5)
    if name:
        save(fig, name)
    else:
        plt.close(fig)
    if nn != MAIN_NNLS:
        return
    # 참고(그림에는 넣지 않음): 배포 MLP 의 같은 64 조건 저장 기록
    fold, acc = fold_of(d["MLP"][k], T)
    c = np.bincount(np.searchsorted([1.25, 1.5, 2.0], fold, side="left"), minlength=4)
    say(f"[b] (참고·그림 제외) 배포 MLP 저장 기록: fold 구간 {c.tolist()} · 2배 이내 {int(c[:3].sum())}/64 · 정확도 {acc.mean():.3f}")


# ------------------------------------------------------------------ 농도 단계 + 64 조건 표
def conc(d):
    """64 조건의 겉보기 µM 대 뿌린 µM. 대조 = 농도 맞춤에 쓴 맵들의 기하평균 농도 하나를 모든 성분에 적는 것."""
    g = d["grid"]
    T, C, test = d["T"][g], d["UM"][g], d["split"][g] == "test"
    fitted = (d["split"] == "loo") & np.isfinite(d["UM"]).all(1)
    const = 10 ** np.log10(d["T"][fitted]).mean()
    fold = np.maximum(C / T, T / C)
    fc = np.maximum(const / T, T / const)
    say(f"[conc] 64 조건 2배 이내 {100 * (fold <= 2).mean():.1f} % ({int((fold <= 2).sum())}/{fold.size}) · "
        f"1.5배 이내 {100 * (fold <= 1.5).mean():.1f} % · 중앙 {np.median(fold):.2f}× · "
        f"RMSE {np.sqrt(np.mean(np.log10(C / T) ** 2)):.3f} dex")
    say("[conc] 성분별 2배 이내 " + " · ".join(f"{s} {100 * (fold[:, j] <= 2).mean():.1f} %" for j, s in enumerate(COMP))
        + f" · 세 성분 모두 {int((fold <= 2).all(1).sum())}/{len(fold)}")
    say(f"[conc] 대조(상수 {const:.1f} µM, 맞춤 {int(fitted.sum())} 맵) 2배 이내 {100 * (fc <= 2).mean():.1f} % · "
        f"세 성분 모두 {int((fc <= 2).all(1).sum())}/{len(fc)}")
    say(f"[conc] 시험 격자 {int(test.sum())} 맵: 2배 이내 {100 * (fold[test] <= 2).mean():.1f} % · "
        f"1.5배 이내 {100 * (fold[test] <= 1.5).mean():.1f} % · 중앙 {np.median(fold[test]):.2f}× "
        f"(대조 {100 * (fc[test] <= 2).mean():.1f} %)")
    say("[conc] 성분별 1.5배 이내 " + " · ".join(f"{s} {100 * (fold[:, j] <= 1.5).mean():.1f} %" for j, s in enumerate(COMP))
        + f" · 대조 중앙 {np.median(fc):.2f}×")
    top = np.array([(t == t.max()).sum() == 1 for t in T])
    say(f"[conc] 가장 많은 성분을 맞힌 조건 {int((C[top].argmax(1) == T[top].argmax(1)).sum())}/{int(top.sum())}")
    lv = (3.0, 6.0, 12.0, 24.0)
    slope = lambda t, c: np.polyfit(np.log10(t).ravel(), np.log10(c).ravel(), 1)[0]  # noqa: E731
    say(f"[conc] log 판독 대 log 뿌린 농도 기울기 {slope(T, C):.2f} · 성분별 "
        + " · ".join(f"{s} {slope(T[:, j], C[:, j]):.2f}" for j, s in enumerate(COMP)))
    say("[conc] 뿌린 농도 수준별(세 성분 합쳐서) 2배 이내 / 판독 중앙 (10–90 %): "
        + " · ".join(f"{v:g} µM {100 * (fold[T == v] <= 2).mean():.1f} % / {np.median(C[T == v]):.1f} "
                     f"({np.percentile(C[T == v], 10):.1f}–{np.percentile(C[T == v], 90):.1f})" for v in lv))
    say("[conc] 성분별 판독 중앙(3 · 6 · 12 · 24): "
        + " | ".join(f"{s} " + " · ".join(f"{np.median(C[T[:, j] == v, j]):.1f}" for v in lv) for j, s in enumerate(COMP)))
    say("[conc] 성분별 수준별 2배 이내 %: "
        + " | ".join(f"{s} " + " · ".join(f"{100 * (fold[T[:, j] == v, j] <= 2).mean():.0f}" for v in lv)
                     for j, s in enumerate(COMP)))
    say("[conc] 등몰: " + " · ".join(f"{v:g}/{v:g}/{v:g} → " + " / ".join(f"{x:.1f}" for x in C[(T == v).all(1)][0]) for v in lv))
    return fold


def tables(d, fold_um):
    """그림을 직접 그릴 때 쓰는 64 조건 표 두 개."""
    g = d["grid"]
    T = d["T"][g]
    Tn = T / T.sum(1, keepdims=True)
    name, split = np.array(d["name"])[g], d["split"][g]
    kind = {"loo": "left out of the fit", "test": "test"}
    folds = {m: fold_of(d[m][g], T)[0] for m in ("BAND", "PLS")}
    with open(os.path.join(OUT, "fig4_composition_64.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "held_out_as"] + [f"{k}_applied_uM" for k in COMP] + [f"{k}_applied_share" for k in COMP]
                   + [f"{k}_band_share" for k in COMP] + [f"{k}_PLS_share" for k in COMP]
                   + ["band_error_pp", "PLS_error_pp", "band_fold", "PLS_fold", "band_fold_bin", "PLS_fold_bin",
                      "gate_px", "n_px"])
        for i in range(len(T)):
            e = [50 * np.abs(d[m][g][i] - Tn[i]).sum() for m in ("BAND", "PLS")]
            fo = [folds[m][i] for m in ("BAND", "PLS")]
            w.writerow([name[i], kind[split[i]], *T[i].astype(int), *np.round(Tn[i], 4), *np.round(d["BAND"][g][i], 4),
                        *np.round(d["PLS"][g][i], 4), *np.round(e, 2), *np.round(fo, 3),
                        *[BANDL[int(np.searchsorted([1.25, 1.5, 2.0], v, side="left"))] for v in fo],
                        d["gate"][g][i], d["npx"][g][i]])
    with open(os.path.join(OUT, "fig4_concentration_64.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "held_out_as"] + [f"{k}_applied_uM" for k in COMP] + [f"{k}_PLS_uM" for k in COMP]
                   + [f"{k}_fold" for k in COMP] + ["all_within_2x"])
        for i in range(len(T)):
            w.writerow([name[i], kind[split[i]], *T[i].astype(int), *np.round(d["UM"][g][i], 2), *np.round(fold_um[i], 3),
                        int((fold_um[i] <= 2).all())])


# ------------------------------------------------------------------ (c) lettering map
def grid_image(coords, values, fill=np.nan):
    xs, ys = np.unique(np.round(coords[:, 0], 3)), np.unique(np.round(coords[:, 1], 3))
    ix, iy = np.searchsorted(xs, np.round(coords[:, 0], 3)), np.searchsorted(ys, np.round(coords[:, 1], 3))
    v = np.asarray(values, float)
    img = np.full((len(ys), len(xs)) + v.shape[1:], fill, float)
    img[iy, ix] = v
    ext = [xs[0], xs[-1], ys[-1], ys[0]]
    return img, ext


def fig_c(z):
    coords, hit = z["coords"], z["hit"]
    band, pls, nn = z["band"], z["pls"], z["nnls"]
    order = ("THI", "TBZ", "DQ")
    idx = {s: list(z["comp"]).index(s) for s in order}
    rgb = np.array([matplotlib.colors.to_rgb(COL[s]) for s in COMP])
    vmax = 3000.0
    bn = np.clip(band / vmax, 0, 1)
    merge_i = np.clip(bn @ rgb, 0, 1)
    comp_hit = np.where(hit[:, None], pls, 0.0)
    merge_ii = np.clip(comp_hit @ rgb, 0, 1)
    img0, ext = grid_image(coords, merge_i, 0.0)
    h, w = img0.shape[:2]
    fw = 7.2
    fig, axes = plt.subplots(2, 4, figsize=(fw, fw / 4 * h / w * 2 + 0.75), gridspec_kw=dict(wspace=0.22, hspace=0.3))
    for r_, (merge, chan, vm, cbl) in enumerate(((merge_i, band, vmax, "Intensity (counts)"),
                                                (merge_ii, comp_hit, 1.0, "Fraction"))):
        ax = axes[r_, 0]
        ax.imshow(grid_image(coords, merge, 0.0)[0], interpolation="nearest", aspect="equal")
        ax.set_title("Merge", fontsize=7, pad=3)
        for c_, s in enumerate(order, start=1):
            ax = axes[r_, c_]
            cmap = LinearSegmentedColormap.from_list(s, ["black", COL[s]])
            im = ax.imshow(grid_image(coords, chan[:, idx[s]], 0.0)[0], cmap=cmap, vmin=0, vmax=vm,
                           interpolation="nearest", aspect="equal")
            ttl = f"{s}, {z['bands_cm'][idx[s]]:.0f} cm$^{{-1}}$" if r_ == 0 else s
            ax.set_title(ttl, fontsize=7, pad=3, color=COL[s])
            cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
            cb.set_ticks([0, vm])
            cb.set_ticklabels(["0", "3k" if vm > 1 else "1"])
            cb.ax.tick_params(length=1.5, width=0.4, labelsize=6)
            cb.outline.set_linewidth(0.4)
        for ax in axes[r_]:
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    axes[0, 0].set_ylabel("Band intensity", fontsize=7)
    axes[1, 0].set_ylabel("PLS model", fontsize=7)
    save(fig, "fig4c_lettering")
    say(f"[c] 글씨맵 {w}×{h} px = {len(hit)} · NNLS 게이트 0.15 통과 {int(hit.sum())} px")
    nm, nb = z["nnls_m"], z["nnls_b"]
    say(f"[c] 게이트 통과 픽셀 평균 DQ/TBZ/THI — NNLS(BLK) {np.round(nb[hit].mean(0) * 100, 1).tolist()} · "
        f"NNLS(배경 없음) {np.round(nm[hit].mean(0) * 100, 1).tolist()} · "
        f"NNLS(파이프라인, 본문) {np.round(nn[hit].mean(0) * 100, 1).tolist()} · "
        f"PLS {np.round(pls[hit].mean(0) * 100, 1).tolist()} · MLP(참고) {np.round(z['mlp'][hit].mean(0) * 100, 1).tolist()} (참 33.3 each)")
    say(f"[c] 맵 판독(격자 맵과 같이 픽셀 몫을 평균한 뒤 세 성분 합 1, 본문) DQ/TBZ/THI — "
        f"PLS {np.round(z['pls_map'] * 100, 1).tolist()} · MLP(참고) {np.round(z['mlp_map'] * 100, 1).tolist()}")
    say(f"[c] THI 가 최대인 픽셀 — NNLS(BLK) {int((nb[hit].argmax(1) == 2).sum())}/{int(hit.sum())} · "
        f"NNLS(배경 없음) {int((nm[hit].argmax(1) == 2).sum())}/{int(hit.sum())} · "
        f"NNLS(파이프라인, 본문) {int((nn[hit].argmax(1) == 2).sum())}/{int(hit.sum())} · "
        f"PLS 로 THI 가 최대인 픽셀 {int((pls[hit].argmax(1) == 2).sum())}/{int(hit.sum())}")
    say(f"[c] PLS 픽셀 최대 성분 분포 DQ/TBZ/THI {np.bincount(pls[hit].argmax(1), minlength=3).tolist()}")
    um = z["uM_pls"][hit]
    say(f"[c] PLS 픽셀 µM 중앙값 DQ/TBZ/THI {np.round(np.nanmedian(um, 0), 1).tolist()} (참 12/12/12) · "
        f"2배 이내 픽셀 {np.round(100 * np.nanmean((um >= 6) & (um <= 24), 0), 0).astype(int).tolist()} %")
    say(f"[c] 밴드 세기(counts) 게이트 통과 픽셀 중앙값 DQ/TBZ/THI {np.round(np.median(band[hit], 0)).astype(int).tolist()} · "
        f"나머지 픽셀 중앙값 {np.round(np.median(band[~hit], 0)).astype(int).tolist()}")


# ------------------------------------------------------------------ equimolar series
def fig_eq(d):
    T = d["T"]
    eq = (T[:, 0] == T[:, 1]) & (T[:, 1] == T[:, 2])
    conc = T[eq, 0]
    levels = np.unique(conc)
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.0), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, (key, lab) in zip(axes, ((MAIN_NNLS, "NNLS"), ("PLS", "PLS model (held-out)"))):
        S = d[key][eq] * 100
        ax.axvspan(3, 24, color="#f1f2f4", lw=0, zorder=0)
        ax.axhline(100 / 3, color="#3a3f4a", lw=0.6, ls=(0, (4, 3)), zorder=1)
        for i, s in enumerate(COMP):
            mean = [S[conc == c, i].mean() for c in levels]
            ax.plot(levels, mean, "-", color=COL[s], lw=0.9, zorder=2)
            ax.scatter(conc, S[:, i], s=13, color=COL[s], edgecolor="white", lw=0.4, zorder=3, label=s)
        ax.set_xscale("log")
        ax.set(xlim=(2.2, 450), ylim=(0, 100), xticks=[3, 10, 33, 100, 333], xticklabels=["3", "10", "33", "100", "333"],
               xlabel="Concentration per component (µM)")
        ax.minorticks_off()
        ax.set_title(lab, fontsize=7, pad=3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Read fraction (%)")
    axes[0].text(8.5, 96, "3–24 µM", ha="center", va="top", fontsize=6, color="#6b7280")
    axes[1].legend(frameon=False, loc="upper left", handletextpad=0.2, borderpad=0.1, fontsize=6.5)
    save(fig, "eq_series")
    say("[eq] 등몰 혼합물 판독 DQ/TBZ/THI % (조성 오차 %p)")
    for n, t, sp, a, m, k, b in zip(np.array(d["name"])[eq], conc, d["split"][eq], d["NNLS"][eq], d["NNLSm"][eq],
                                    d["NNLSb"][eq], d["PLS"][eq]):
        dv = lambda v: 50 * np.abs(v - 1 / 3).sum()  # noqa: E731
        say(f"   {t:>5.0f} µM  {sp:<4}  NNLS(BLK) {np.round(k * 100).astype(int).tolist()} ({dv(k):.1f})   "
            f"NNLS(배경 없음) {np.round(m * 100).astype(int).tolist()} ({dv(m):.1f})   "
            f"NNLS(파이프라인, 본문) {np.round(a * 100).astype(int).tolist()} ({dv(a):.1f})   "
            f"PLS {np.round(b * 100).astype(int).tolist()} ({dv(b):.1f})   {n}")


def main():
    d = load()
    band_readers(d)
    for m in ("BAND", "NNLSb", "NNLSm", "NNLS", "PLS"):         # 성분별 평균 편향과 고농도(격자 밖) 조성 오차
        Tn = d["T"] / d["T"].sum(1, keepdims=True)
        g = d["grid"]
        say(f"[bias] {m}: 64격자 편향 DQ/TBZ/THI {np.round((d[m][g] - Tn[g]).mean(0) * 100, 1).tolist()} %p · "
            f"격자 밖 {int((~g).sum())}맵 조성 오차 {50 * np.abs(d[m][~g] - Tn[~g]).sum(1).mean():.1f} %p")
    say(f"held-out 맵 {len(d['name'])} · 64격자 {int(d['grid'].sum())} (loo {int((d['grid'] & (d['split'] == 'loo')).sum())}"
        f" + test {int((d['grid'] & (d['split'] == 'test')).sum())})")
    gp = d["gate"] / d["npx"]
    eg = 50 * np.abs(d["PLS"] - d["T"] / d["T"].sum(1, keepdims=True)).sum(1)[d["grid"]]
    sg = d["split"][d["grid"]]
    say(f"[gate] 게이트 통과 픽셀: 64격자 {int(d['gate'][d['grid']].sum())}/{int(d['npx'][d['grid']].sum())}"
        f" ({100 * gp[d['grid']].mean():.1f} %, 가장 낮은 맵 {100 * gp[d['grid']].min():.0f} %) · "
        f"격자 밖 {100 * gp[~d['grid']].mean():.1f} % (가장 낮은 맵 {100 * gp[~d['grid']].min():.0f} %)")
    say(f"[PLS] 64 조건 조성 오차 {eg.mean():.1f} %p · 맞춤에서 뺀 {int((sg == 'loo').sum())} 조건 {eg[sg == 'loo'].mean():.1f} %p · "
        f"시험 {int((sg == 'test').sum())} 조건 {eg[sg == 'test'].mean():.1f} %p")
    tables(d, conc(d))
    fig_b(d, "BAND", "fig4b_ternary", "Band intensity")        # 본문: 신호 세기 + 단일성분 검량선 | PLS
    fig_b(d, MAIN_NNLS, "fig4b_ternary_nnls")                  # SI: 파이프라인 NNLS | PLS
    fig_b(d, "NNLSb", "fig4b_ternary_nnls_blk")                # SI: 순수 3성분 + BLK | PLS
    fig_b(d, "NNLSm", None)
    for key in ("lin50", "curve"):
        fig_b(d, f"BAND_{key}", None)
    fig_c(np.load(os.path.join(OUT, "lettering_pixels.npz")))
    fig_eq(d)
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
