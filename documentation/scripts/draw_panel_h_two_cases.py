"""Draw the two-case physics-constrained composition panel."""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

COLORS = {"DQ": "#2387d9", "TBZ": "#25a36f", "THI": "#ef4f75"}
ORDER = ["True", "NNLS", "MLP", "MLP + Physics"]


def load_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def draw_case(ax, rows, title, subtitle, metric):
    by_method = {r["method"]: r for r in rows}
    x = np.arange(4)
    bottom = np.zeros(4)
    for analyte in ("DQ", "TBZ", "THI"):
        vals = np.array([float(by_method[m][f"{analyte}_pct"]) for m in ORDER])
        bars = ax.bar(x, vals, bottom=bottom, width=.70, color=COLORS[analyte],
                      edgecolor="white", linewidth=.7, label=analyte)
        for bar, val, base in zip(bars, vals, bottom):
            if val >= 7:
                ax.text(bar.get_x() + bar.get_width()/2, base + val/2,
                        f"{val:.0f}", ha="center", va="center",
                        color="white", fontsize=7.2, weight="bold")
        bottom += vals
    ax.set_title(title, loc="left", fontsize=9.2, weight="bold", pad=13)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=7.4,
            color="#667085", va="bottom")
    ax.set_xticks(x, ["True", "NNLS", "MLP", "MLP +\nPhysics"])
    ax.set_ylim(0, 116)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=7.4)
    ax.grid(axis="y", color="#e7e9ee", linewidth=.55, zorder=0)
    ax.set_axisbelow(True)
    if metric == "error":
        vals = [float(by_method[m]["composition_error_pct"]) for m in ORDER[1:]]
        ax.text(1, 106, f"{vals[0]:.1f}%", ha="center", fontsize=7.6, color="#667085")
        ax.annotate(f"{vals[1]:.1f}%  →  {vals[2]:.1f}%\n−{vals[1]-vals[2]:.1f}%p",
                    xy=(3, 103.0), xytext=(2, 112.0), ha="center", va="center",
                    fontsize=7.5, weight="bold",
                    arrowprops=dict(arrowstyle="-|>", lw=.8, color="#222831"))
        ax.text(.5, -.20, "Composition error", transform=ax.transAxes,
                ha="center", fontsize=7.1, color="#667085")
    else:
        vals = [float(by_method[m]["THI_bias_pct_point"]) for m in ORDER[1:]]
        ax.text(1, 106, f"THI +{vals[0]:.1f}%p", ha="center", fontsize=7.4,
                color=COLORS["THI"], weight="bold")
        ax.annotate(f"+{vals[1]:.1f}  →  +{vals[2]:.1f}%p",
                    xy=(3, 103.0), xytext=(2, 112.0), ha="center", va="center",
                    fontsize=7.5, color=COLORS["THI"], weight="bold",
                    arrowprops=dict(arrowstyle="-|>", lw=.8, color=COLORS["THI"]))
        ax.text(.5, -.20, "THI prediction bias", transform=ax.transAxes,
                ha="center", fontsize=7.1, color="#667085")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = load_rows(a.csv)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.15), dpi=300, sharey=True)
    case1 = [r for r in rows if r["case"] == "max_rescue"]
    case2 = [r for r in rows if r["case"] == "THI_overestimate"]
    draw_case(axes[0], case1, "Maximum held-out rescue",
              "DQ/TBZ/THI = 12/6/3 µM", "error")
    draw_case(axes[1], case2, "Correction of THI overestimation",
              "DQ/TBZ/THI = 24/3/12 µM", "bias")
    axes[0].set_ylabel("Composition (%)", fontsize=8.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in ("DQ", "TBZ", "THI")]
    fig.legend(handles, ["DQ", "TBZ", "THI"], loc="upper center",
               bbox_to_anchor=(.5, .985), ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Physics-constrained full-spectrum recovery", x=.08, y=1.035,
                 ha="left", fontsize=11, weight="bold")
    fig.text(.98, .985, "$L=L_{data}+0.03L_{surface}$", ha="right", va="top",
             fontsize=7.6, color="#667085")
    fig.subplots_adjust(left=.09, right=.99, bottom=.22, top=.77, wspace=.19)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(a.out)


if __name__ == "__main__":
    main()
