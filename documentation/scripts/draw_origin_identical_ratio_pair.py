"""Draw identical-composition, two-total-concentration comparison."""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

COL = {"DQ": "#2387d9", "TBZ": "#25a36f", "THI": "#ef4f75"}
METHODS = ["True", "NNLS", "Physics-guided MLP"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    conditions = [("DQ12-TBZ6-THI3", "21 µM total\n12/6/3 µM"),
                  ("DQ24-TBZ12-THI6", "42 µM total\n24/12/6 µM")]
    fig, ax = plt.subplots(figsize=(6.4, 3.25), dpi=300)
    centers = [1, 5]
    offsets = [-.78, 0, .78]
    width = .68
    for ci, (condition, label) in enumerate(conditions):
        group = {r["method"]: r for r in rows if r["condition"] == condition}
        for mi, method in enumerate(METHODS):
            x = centers[ci] + offsets[mi]
            bottom = 0.0
            for analyte in ("DQ", "TBZ", "THI"):
                value = float(group[method][f"{analyte}_pct"])
                ax.bar(x, value, bottom=bottom, width=width, color=COL[analyte],
                       edgecolor="white", linewidth=.7, zorder=3)
                if value >= 7:
                    ax.text(x, bottom + value/2, f"{value:.0f}", ha="center",
                            va="center", color="white", fontsize=7.3, weight="bold")
                bottom += value
        nnls_err = float(group["NNLS"]["composition_error_pct"])
        phys_err = float(group["Physics-guided MLP"]["composition_error_pct"])
        ax.annotate(f"{nnls_err:.1f}%  →  {phys_err:.1f}%",
                    xy=(centers[ci] + offsets[2], 104),
                    xytext=(centers[ci] + offsets[1], 112),
                    ha="center", va="center", fontsize=8, weight="bold",
                    arrowprops=dict(arrowstyle="-|>", color="#222831", lw=.85))
        ax.text(centers[ci], -13.0, label, ha="center", va="top",
                fontsize=8.2, weight="bold")
    xs = [c + o for c in centers for o in offsets]
    ax.set_xticks(xs, ["True", "NNLS", "Physics-\nguided MLP"] * 2)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Composition (%)", fontsize=9)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", color="#e7e9ee", linewidth=.55, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in ("DQ", "TBZ", "THI")]
    ax.legend(handles, ["DQ", "TBZ", "THI"], ncol=3, frameon=False,
              loc="upper center", bbox_to_anchor=(.5, 1.17), fontsize=8)
    ax.set_title("Physics-guided recovery at identical composition",
                 loc="left", pad=40, fontsize=11, weight="bold")
    ax.text(.99, 1.18, "True composition: 57.1/28.6/14.3%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.6,
            color="#667085")
    fig.text(.99, .02, "$L=L_{data}+0.03L_{surface}$  |  condition-held-out fold 2",
             ha="right", fontsize=7.1, color="#667085")
    fig.subplots_adjust(left=.10, right=.99, bottom=.25, top=.76)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(a.out)


if __name__ == "__main__":
    main()
