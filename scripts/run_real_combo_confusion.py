"""run_real_combo_confusion.py — treat each mixture's component SET as one class.

"Did the model recognise the THI+TBZ mixture as THI+TBZ?" — answer it with a
real square confusion matrix whose classes are the component COMBINATIONS
(rows = true combination, columns = predicted combination). The diagonal =
exactly-right composition.

    python run_real_combo_confusion.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap

from sers_mixture import preprocess, SERSMixtureClassifier, AugmentConfig
from run_real_pest import load_map, PEST, COMPS, MIXES

TEAL = "#0f9d6b"
CMAP = LinearSegmentedColormap.from_list(
    "teal", ["#f1f7f4", "#a9ddc7", "#3fb488", TEAL, "#0a6b49"])


def combo(names):
    """Canonical 'DQ+THI' label in fixed component order."""
    order = [c for c in COMPS if c in names]
    return "+".join(order) if order else "(none)"


def main():
    _, _, dq = load_map(os.path.join(PEST, "Reference", "DQ_corrected.csv"))
    _, _, thi = load_map(os.path.join(PEST, "Reference", "THI_corrected.csv"))
    _, _, tbz = load_map(os.path.join(PEST, "Reference", "TBZ_corrected.csv"))
    pures = preprocess(np.vstack([dq, thi, tbz]))

    clf = SERSMixtureClassifier(COMPS, prob_threshold=0.30, max_components=3,
                                augment=AugmentConfig(n_per_pure=200))
    clf.fit(pures)

    true_lbl, pred_lbl = [], []
    for k, nominal in MIXES.items():
        _, _, mean = load_map(os.path.join(PEST, "Ratio", k + "_corrected.csv"))
        p = clf.predict(preprocess(mean[None, :]))[0]
        true_lbl.append(combo([COMPS[i] for i, c in enumerate(nominal) if c > 0]))
        pred_lbl.append(combo(p))

    # classes: true combos first (rows), then any predicted-only combos (extra cols)
    row_classes = sorted(set(true_lbl), key=lambda s: (s.count("+"), s))
    extra = sorted(set(pred_lbl) - set(row_classes), key=lambda s: (s.count("+"), s))
    col_classes = row_classes + extra

    M = np.zeros((len(row_classes), len(col_classes)), int)
    ri = {c: i for i, c in enumerate(row_classes)}
    ci = {c: i for i, c in enumerate(col_classes)}
    for t, p in zip(true_lbl, pred_lbl):
        M[ri[t], ci[p]] += 1

    exact = sum(t == p for t, p in zip(true_lbl, pred_lbl)) / len(true_lbl)

    fig = Figure(figsize=(8.6, 5.4), dpi=120); fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.imshow(M, cmap=CMAP, aspect="auto", vmin=0)
    ax.set_xticks(range(len(col_classes))); ax.set_xticklabels(col_classes, fontsize=10)
    ax.set_yticks(range(len(row_classes))); ax.set_yticklabels(row_classes, fontsize=10)
    ax.set_xlabel("predicted combination", fontsize=11)
    ax.set_ylabel("true combination", fontsize=11)
    ax.set_title(f"combination confusion — exact match {exact:.0%}  (10 real mixtures)",
                 fontsize=12)
    ax.set_xticks(np.arange(-0.5, len(col_classes)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_classes)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    thr = M.max() / 2 if M.max() else 0.5
    for (r, c), v in np.ndenumerate(M):
        if v:
            ax.text(c, r, str(v), ha="center", va="center", fontsize=13,
                    color="white" if v > thr else "#1c2430", fontweight="bold")
    # mark the correct (diagonal) cells where col label == row label
    for r, rc in enumerate(row_classes):
        if rc in ci:
            ax.add_patch(matplotlib.patches.Rectangle(
                (ci[rc] - 0.5, r - 0.5), 1, 1, fill=False,
                edgecolor="#1c2430", lw=2))
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "real_pest_combo_confusion.png")
    fig.savefig(out, dpi=150, facecolor="white")
    print("saved ->", out)
    print(f"\nexact composition match: {exact:.0%}")
    for t, p in zip(true_lbl, pred_lbl):
        print(f"  true {t:14s} -> predicted {p}")


if __name__ == "__main__":
    import matplotlib.patches  # noqa
    main()
