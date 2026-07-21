"""run_real_calibrated.py — response-factor calibration on the REAL pesticide data.

The raw NNLS signal ratio over-represents THI because thiram binds strongly and
has a large SERS response (B_i = R_i * C_i, R_THI >> R_DQ, R_TBZ). We recover the
per-compound response factor R_i from the 1:1 mixtures (equal concentration, so
R_a/R_b = B_a/B_b), then correct every mixture:  C_i  ∝  B_i / R_i.

Calibrate on the three 1:1 standards (DQ1TH1, TB1TH1, TBZ1DQ1); the 1:3 / 3:1
mixtures are NOT used for calibration, so they are an honest test.

    python run_real_calibrated.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from sers_mixture import preprocess
from competitive import fit_B
from run_real_pest import load_map, PEST, COMPS, SERIES, MIXES


def fit_all_B(names, pures):
    """Return {name: B vector over COMPS} for each ratio map (from its mean)."""
    means = []
    for k in names:
        _, _, mean = load_map(os.path.join(PEST, "Ratio", k + "_corrected.csv"))
        means.append(mean)
    means = preprocess(np.array(means))
    return {k: fit_B(m, pures)[0] for k, m in zip(names, means)}


def calibrate_R(B):
    """Least-squares response factors from the three 1:1 standards.

    log R (ref THI=0) from:
        DQ1TH1 : logR_DQ  - logR_THI = log(B_DQ /B_THI)
        TB1TH1 : logR_TBZ - logR_THI = log(B_TBZ/B_THI)
        TBZ1DQ1: logR_TBZ - logR_DQ  = log(B_TBZ/B_DQ)
    unknowns x = [logR_DQ, logR_TBZ] (logR_THI fixed at 0)."""
    iDQ, iTHI, iTBZ = 0, 1, 2
    A, y = [], []
    A.append([1, 0]); y.append(np.log(B["DQ1TH1"][iDQ] / B["DQ1TH1"][iTHI]))
    A.append([0, 1]); y.append(np.log(B["TB1TH1"][iTBZ] / B["TB1TH1"][iTHI]))
    A.append([-1, 1]); y.append(np.log(B["TBZ1DQ1"][iTBZ] / B["TBZ1DQ1"][iDQ]))
    x, *_ = np.linalg.lstsq(np.array(A, float), np.array(y, float), rcond=None)
    R = np.array([np.exp(x[0]), 1.0, np.exp(x[1])])   # [DQ, THI, TBZ]
    return R / np.median(R)


def ratio_over(present_idx, vec):
    v = np.array([vec[i] if i in present_idx else 0.0 for i in range(3)])
    return v / (v.sum() + 1e-12)


def main():
    _, _, dq = load_map(os.path.join(PEST, "Reference", "DQ_corrected.csv"))
    _, _, thi = load_map(os.path.join(PEST, "Reference", "THI_corrected.csv"))
    _, _, tbz = load_map(os.path.join(PEST, "Reference", "TBZ_corrected.csv"))
    pures = preprocess(np.vstack([dq, thi, tbz]))

    names = list(MIXES.keys())
    B = fit_all_B(names, pures)
    R = calibrate_R(B)
    print("response factors R (rel.):  "
          + "  ".join(f"{c}={R[i]:.2f}" for i, c in enumerate(COMPS)))
    print(f"  -> THI binds/emits ~{R[1]/R[0]:.0f}x DQ and ~{R[1]/R[2]:.0f}x TBZ "
          "per unit concentration\n")

    print(f"{'mixture':9s} {'comp':4s}  {'nominal':>7s} {'raw sig':>8s} {'calibrated':>11s}")
    err_raw, err_cal = [], []
    rows = []
    for k in names:
        nominal = np.array(MIXES[k], float)
        present = [i for i in range(3) if nominal[i] > 0]
        nom = ratio_over(present, nominal)
        raw = ratio_over(present, B[k])
        cal = ratio_over(present, B[k] / R)
        rows.append((k, present, nom, raw, cal))
        for i in present:
            print(f"{k:9s} {COMPS[i]:4s}  {nom[i]*100:6.0f}% {raw[i]*100:7.0f}% "
                  f"{cal[i]*100:10.0f}%")
            err_raw.append(abs(raw[i] - nom[i])); err_cal.append(abs(cal[i] - nom[i]))
        print()
    print(f"mean |recovered-nominal|:   raw {np.mean(err_raw)*100:.1f}%  ->  "
          f"calibrated {np.mean(err_cal)*100:.1f}%")

    # ---- before/after parity figure ----
    fig = Figure(figsize=(11, 4.6), dpi=120); fig.patch.set_facecolor("white")
    for ax, key, title in [(fig.add_subplot(1, 2, 1), "raw",
                            "before — raw signal ratio"),
                           (fig.add_subplot(1, 2, 2), "cal",
                            "after — response-calibrated")]:
        for k, present, nom, raw, cal in rows:
            val = raw if key == "raw" else cal
            for i in present:
                ax.scatter(nom[i], val[i], s=48, color=SERIES[COMPS[i]],
                           edgecolors="white", linewidths=0.5, zorder=3)
        ax.plot([0, 1], [0, 1], ls="--", color="#98a1ac", lw=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("nominal fraction"); ax.set_ylabel("recovered fraction")
        ax.set_title(title)
    handles = [matplotlib.lines.Line2D([], [], marker="o", ls="", color=SERIES[c],
               label=c) for c in COMPS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "real_pest_calibrated.png")
    fig.savefig(out, dpi=140, facecolor="white")
    print("\nsaved figure ->", out)


if __name__ == "__main__":
    import matplotlib.lines  # noqa
    main()
