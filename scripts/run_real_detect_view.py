"""run_real_detect_view.py — clearer detection view for the real pesticide data.

Replaces the confusing "green fill + red outline" grid with:
  (A) an outcome grid — every (sample, pesticide) cell is ONE flat colour:
        hit (present & found) · miss (present & not found) · false alarm
        (absent & found) · correct-absent — with a legend.
  (B) the proper multi-label "confusion": one 2x2 per pesticide
        (present/absent  ×  detected/not).

    python run_real_detect_view.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from sers_mixture import (preprocess, SERSMixtureClassifier, AugmentConfig,
                          names_to_indicator)
from run_real_pest import load_map, PEST, COMPS, MIXES

# light-theme outcome colours
TEAL = "#0f9d6b"   # hit  (TP)
RED = "#d64545"    # miss (FN)
AMBER = "#e0a020"  # false alarm (FP)
GRAY = "#eef1f4"   # correct-absent (TN)
INK = "#1c2430"


def main():
    _, _, dq = load_map(os.path.join(PEST, "Reference", "DQ_corrected.csv"))
    _, _, thi = load_map(os.path.join(PEST, "Reference", "THI_corrected.csv"))
    _, _, tbz = load_map(os.path.join(PEST, "Reference", "TBZ_corrected.csv"))
    pures = preprocess(np.vstack([dq, thi, tbz]))

    clf = SERSMixtureClassifier(COMPS, prob_threshold=0.30, max_components=3,
                                augment=AugmentConfig(n_per_pure=200))
    clf.fit(pures)

    names, means, true = [], [], []
    for k, nominal in MIXES.items():
        _, _, mean = load_map(os.path.join(PEST, "Ratio", k + "_corrected.csv"))
        names.append(k); means.append(mean)
        true.append([COMPS[i] for i, c in enumerate(nominal) if c > 0])
    means = preprocess(np.array(means))
    pred = clf.predict(means)
    yt = names_to_indicator(true, COMPS)
    yp = names_to_indicator(pred, COMPS)

    fig = Figure(figsize=(12, 5.2), dpi=120); fig.patch.set_facecolor("white")

    # ---- (A) outcome grid ----
    ax = fig.add_subplot(1, 2, 1)
    nS, nC = yt.shape
    for r in range(nS):
        for c in range(nC):
            t, p = yt[r, c], yp[r, c]
            if t and p:
                col, mark = TEAL, "O"
            elif t and not p:
                col, mark = RED, "X"
            elif not t and p:
                col, mark = AMBER, "!"
            else:
                col, mark = GRAY, ""
            ax.add_patch(matplotlib.patches.Rectangle((c, nS - 1 - r), 1, 1,
                         facecolor=col, edgecolor="white", linewidth=2))
            if mark:
                ax.text(c + 0.5, nS - 1 - r + 0.5, mark, ha="center", va="center",
                        color="white", fontsize=13, fontweight="bold")
    ax.set_xlim(0, nC); ax.set_ylim(0, nS)
    ax.set_xticks(np.arange(nC) + 0.5); ax.set_xticklabels(COMPS)
    ax.set_yticks(np.arange(nS) + 0.5); ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_title("detection outcome  (each cell = one pesticide in one sample)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(handles=[
        Patch(facecolor=TEAL, label="hit — present & found"),
        Patch(facecolor=RED, label="miss — present, not found"),
        Patch(facecolor=AMBER, label="false alarm — absent, found"),
        Patch(facecolor=GRAY, label="correct — absent & not found")],
        loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)

    # ---- (B) per-pesticide 2x2 confusion ----
    gs = fig.add_gridspec(3, 2)
    for i, nm in enumerate(COMPS):
        axc = fig.add_subplot(gs[i, 1])
        t, p = yt[:, i], yp[:, i]
        TP = int(((t == 1) & (p == 1)).sum()); FN = int(((t == 1) & (p == 0)).sum())
        FP = int(((t == 0) & (p == 1)).sum()); TN = int(((t == 0) & (p == 0)).sum())
        M = np.array([[TP, FN], [FP, TN]])
        axc.imshow([[0.9, 0.4], [0.4, 0.1]], cmap="Greens", vmin=0, vmax=1, aspect="auto")
        for (r, c), v in np.ndenumerate(M):
            axc.text(c, r, str(v), ha="center", va="center", fontsize=11,
                     color=INK, fontweight="bold")
        axc.set_xticks([0, 1]); axc.set_xticklabels(["detected", "not"], fontsize=8)
        axc.set_yticks([0, 1]); axc.set_yticklabels(["present", "absent"], fontsize=8)
        axc.set_title(f"{nm}", fontsize=10, loc="left")
        axc.tick_params(length=0)
    fig.text(0.74, 0.965, "multi-label confusion — one 2×2 per pesticide",
             fontsize=10, ha="left")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "real_pest_detection_view.png")
    fig.savefig(out, dpi=140, facecolor="white")
    print("saved ->", out)
    print("\nper-pesticide (present, detected/missed):")
    for i, nm in enumerate(COMPS):
        t, p = yt[:, i], yp[:, i]
        print(f"  {nm}: present in {int(t.sum())} samples, "
              f"found {int(((t==1)&(p==1)).sum())}, missed {int(((t==1)&(p==0)).sum())}")


if __name__ == "__main__":
    import matplotlib.patches  # noqa
    main()
