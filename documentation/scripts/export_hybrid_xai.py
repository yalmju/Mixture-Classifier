"""Export a publication-ready hybrid-model schematic and XAI data/figure.

Trains the current full-data blank+Sips MLP, then explains that exact model with
Integrated Gradients, band permutation, and VIP-band ablation.

2026-08-14: 메인 클론에만 있고 커밋된 적이 없던 스크립트다 — 저장소에 넣으면서 수집을
`mixtures.py` 로 이관했다 (라벨 = `260814_mixture_final` 파일명, 최종 µM).
"""
from __future__ import annotations

import os as _os, sys as _sys                 # paths 부트스트랩 — 기계마다 마운트가 다르다
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, paths.REPO)

import dl_model
import mixtures as MX
from dl_explain import dl_explain

SUB = MX.SUB
COL = {"DQ": "#1687e0", "TBZ": "#23a36d", "THI": "#ef5275"}


def collect(db):
    del db  # 경로는 mixtures/paths 가 정한다 — 인자는 호환용으로만 남긴다
    high = MX.items(("high",))
    low_items = MX.items(("grid",), replicates=True)
    ncond = len(MX.groups(("grid",), replicates=True, verbose=False))
    return high, low_items, ncond


def calibration(db):
    del db
    return paths.calibration()


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def box(ax, xy, wh, text, fc, ec="#4a5565", fs=8.2):
    x, y = xy; w, h = wh
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.018,rounding_size=0.025",
                       linewidth=0.9, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10,
                                 linewidth=1.0, color="#667085"))


def plot_workflow(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, (.02, .69), (.19, .18), "Raman map\n(pixel spectra)", "#eef4fb")
    box(ax, (.28, .69), (.20, .18), "NNLS gate\nhit / background", "#f2f4f7")
    box(ax, (.55, .69), (.20, .18), "Full-spectrum\nMLP", "#fff1f4", ec=COL["THI"])
    box(ax, (.82, .69), (.16, .18), "Composition\nhead", "#f7f3ff")
    arrow(ax, (.21, .78), (.28, .78)); arrow(ax, (.48, .78), (.55, .78)); arrow(ax, (.75, .78), (.82, .78))
    box(ax, (.08, .28), (.22, .19), "Calibration\nK, gA, Sips m", "#eefaf4", ec=COL["TBZ"])
    box(ax, (.39, .28), (.25, .19), "Physics-simulated\nmixtures", "#eefaf4", ec=COL["TBZ"])
    box(ax, (.72, .28), (.22, .19), "Map-pooled\nlog-µM head", "#fff8e8")
    arrow(ax, (.30, .375), (.39, .375)); arrow(ax, (.64, .375), (.72, .375))
    arrow(ax, (.515, .47), (.63, .69))
    ax.text(.5, .08, "physics-guided pretraining + measured-mixture fine-tuning",
            ha="center", va="center", fontsize=8.5, weight="bold", color="#344054")
    ax.set_title("Hybrid spectral–physical learning", loc="left", fontsize=10, weight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=paths.DB)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pure = os.path.join(args.db, "Pure")
    cal = calibration(args.db)
    high, low, ncond = collect(args.db)
    items = high + low
    print(f"training: {len(high)} high maps + {len(low)} low maps ({ncond} low conditions)", flush=True)
    model = dl_model.train_model(
        pure, items, calib_path=cal, baseline=True, trim=None, method="mlp",
        epochs=args.epochs, seed=0, use_pretrain=True, nnls_screen=True,
        px_per_map=20, include_blank=True, sim_nuisance=False,
        sim_iso="sips_dq", progress=lambda s: print(s, flush=True))
    model_path = os.path.join(args.out, "hybrid_blank_sips_full_data.dlm")
    dl_model.save_model(model, model_path)
    print("explaining exact saved model", flush=True)
    ex = dl_explain(pure, low, calib_path=cal, baseline=True, model=model,
                    progress=lambda s: print(s, flush=True))

    wn = np.asarray(ex["wn"], float)
    attr_rows = []
    for s in ex["subs"]:
        vip = np.asarray(ex["vip"].get(s, []), float)
        for x, y in zip(wn, ex["attr"][s]):
            attr_rows.append([f"{x:.6g}", s, f"{float(y):.8g}",
                              int(np.any(np.abs(vip - x) <= 10))])
    write_csv(os.path.join(args.out, "xai_integrated_gradients.csv"),
              ["raman_shift_cm-1", "compound", "normalized_abs_IG", "within_10cm-1_of_VIP"], attr_rows)
    write_csv(os.path.join(args.out, "xai_band_permutation.csv"),
              ["band_center_cm-1", "delta_composition_error"], ex["perm"])
    write_csv(os.path.join(args.out, "xai_ligand_ablation.csv"),
              ["compound", "prediction_before", "prediction_after", "relative_drop_pct"],
              [[s, a, b, 100 * (a - b) / max(a, 1e-12)] for s, (a, b) in ex["abl"].items()])
    workflow = [
        [1, "Raman map", "measured pixel spectra", "real_data.load_map"],
        [2, "NNLS gate", "hit/background fixed before MLP", "dl_model._screened_map_spectra"],
        [3, "Physics simulator", "K, gA and DQ Sips exponent", "dl_quantify.simulate_mixtures"],
        [4, "Composition MLP", "full-spectrum softmax composition", "dl_model.train_model"],
        [5, "Concentration head", "NNLS ratio + intensity quantiles; map median loss", "dl_model._concentration_context_features"],
        [6, "Interpretability", "IG + band permutation + VIP ablation", "dl_explain.dl_explain"],
    ]
    write_csv(os.path.join(args.out, "hybrid_model_workflow.csv"),
              ["stage", "component", "role", "implementation"], workflow)
    write_csv(os.path.join(args.out, "provenance.csv"), ["key", "value"], [
        ["model", os.path.basename(model_path)], ["epochs", args.epochs], ["seed", 0],
        ["method", "MLP"], ["physics_pretraining", True], ["isotherm", "Sips for DQ; Langmuir for TBZ/THI"],
        ["NNLS_screen", True], ["include_blank", True], ["pixels_per_map", 20],
        ["low_conditions", ncond], ["low_maps", len(low)], ["high_maps", len(high)],
        ["evaluation_note", "full-data explanatory fit; predictive performance must use held-out results"],
        ["calibration", cal],
    ])

    fig = plt.figure(figsize=(10.6, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.25])
    ax0 = fig.add_subplot(gs[0, 0]); plot_workflow(ax0)
    ax1 = fig.add_subplot(gs[0, 1])
    for s in ex["subs"]:
        ax1.plot(wn, ex["attr"][s], lw=1.25, color=COL[s], label=s)
        for v in ex["vip"].get(s, []):
            ax1.axvline(v, color=COL[s], lw=.55, alpha=.28)
    ax1.set(xlabel="Raman shift (cm$^{-1}$)", ylabel="Normalized |IG|")
    ax1.set_title("Spectral attribution and VIP bands", loc="left", fontsize=10, weight="bold")
    ax1.legend(frameon=False, ncol=3, fontsize=8)
    ax2 = fig.add_subplot(gs[1, 0])
    perm = np.asarray(ex["perm"], float)
    ax2.bar(perm[:, 0], perm[:, 1], width=48, color="#7b8da6")
    ax2.axhline(0, color="#667085", lw=.7)
    ax2.set(xlabel="Band center (cm$^{-1}$)", ylabel="$\Delta$ composition error")
    ax2.set_title("Band-permutation importance", loc="left", fontsize=10, weight="bold")
    ax3 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(ex["subs"])); before = [ex["abl"][s][0] for s in ex["subs"]]; after = [ex["abl"][s][1] for s in ex["subs"]]
    w = .34
    ax3.bar(x - w/2, before, w, color=[COL[s] for s in ex["subs"]], alpha=.9, label="original")
    ax3.bar(x + w/2, after, w, color=[COL[s] for s in ex["subs"]], alpha=.35, hatch="//", label="VIP bands ablated")
    ax3.set_xticks(x, ex["subs"]); ax3.set_ylabel("Mean predicted fraction")
    ax3.set_title("Ligand-band ablation", loc="left", fontsize=10, weight="bold")
    ax3.legend(frameon=False, fontsize=8)
    for ax, lab in zip([ax0, ax1, ax2, ax3], "abcd"):
        ax.text(-.08, 1.04, f"({lab})", transform=ax.transAxes, fontsize=11, weight="bold")
    out = os.path.join(args.out, "hybrid_model_xai.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
