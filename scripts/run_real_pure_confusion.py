"""run_real_pure_confusion.py — single-component 4-class confusion (DQ/THI/TBZ/BLK).

Shows that while MIXTURES are hard (competition), the PURE substances are cleanly
separable per pixel. Trains a 4-class classifier on the reference-map pixels and
reports a proper square confusion matrix + accuracy.

Caveat (honest): there is only ONE map per compound, so a per-pixel train/test
split shares a map between train and test (pixels are near-duplicates). The
numbers are in-domain and optimistic — they demonstrate *separability of the
classes*, not day-to-day generalisation (that needs independent replicate maps).

    python run_real_pure_confusion.py
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from sers_mixture import preprocess
from run_real_pest import load_map, PEST

TEAL = "#0f9d6b"
CMAP = LinearSegmentedColormap.from_list(
    "teal", ["#f1f7f4", "#a9ddc7", "#3fb488", TEAL, "#0a6b49"])
CLASSES = ["DQ", "THI", "TBZ", "BLK"]
FILES = {"DQ": "DQ_corrected.csv", "THI": "THI_corrected.csv",
         "TBZ": "TBZ_corrected.csv", "BLK": "blk_corrected.csv"}


def main():
    X, y = [], []
    for lab, fn in FILES.items():
        _, cube, _ = load_map(os.path.join(PEST, "Reference", fn))
        Xp = preprocess(cube)
        X.append(Xp); y += [CLASSES.index(lab)] * len(Xp)
        print(f"{lab}: {len(Xp)} pixels")
    X = np.vstack(X); y = np.array(y)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          stratify=y, random_state=0)
    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    acc = accuracy_score(yte, yp)
    cm = confusion_matrix(yte, yp, labels=range(4))
    print(f"\npixel accuracy (in-domain): {acc:.3f}")
    print("confusion:\n", cm)

    fig = Figure(figsize=(6.4, 5.6), dpi=130); fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.imshow(cm, cmap=CMAP, aspect="auto", vmin=0)
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES, fontsize=11)
    ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES, fontsize=11)
    ax.set_xlabel("predicted", fontsize=12); ax.set_ylabel("true", fontsize=12)
    ax.set_title(f"single-component confusion — accuracy {acc:.1%}\n"
                 f"(per-pixel, {len(yte)} held-out pixels)", fontsize=12)
    ax.set_xticks(np.arange(-0.5, 4), minor=True)
    ax.set_yticks(np.arange(-0.5, 4), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    thr = cm.max() / 2
    for (r, c), v in np.ndenumerate(cm):
        if v:
            ax.text(c, r, str(v), ha="center", va="center", fontsize=13,
                    color="white" if v > thr else "#1c2430", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "real_pest_pure_confusion.png")
    fig.savefig(out, dpi=150, facecolor="white")
    print("saved ->", out)


if __name__ == "__main__":
    main()
