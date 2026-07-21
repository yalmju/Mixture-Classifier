"""run_real_strategies.py — detection strategies that beat RF-on-the-mean.

RF/ResNet classify the MEAN spectrum, where the minor component is buried under
THI. Two physics-aware strategies recover it:

  (B) per-pixel NNLS + spatial aggregation — the mean throws away 400 pixels;
      a minor compound that is 4% in the mean can be 30% in a few pixels where
      local coverage favours it. Detect per pixel, then vote across the map.

  (C) matched-filter significance on the mean — instead of a fixed fraction
      threshold, keep a component only if adding its template drops the NNLS
      residual by more than the noise (an SNR/LOD test).

Compared against (A) the RandomForest baseline, on the real ratio maps.

    python run_real_strategies.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import numpy as np
from scipy.optimize import nnls

from sers_mixture import (preprocess, SERSMixtureClassifier, AugmentConfig,
                          names_to_indicator, multilabel_metrics)
from competitive import fit_B
from real_data import load_map, PEST_DEFAULT, COMPS, MIXES


def _combo(idxs):
    return "+".join(COMPS[i] for i in sorted(idxs)) if len(idxs) else "(none)"


def per_pixel_vote(cube, pures, frac_thr=0.15, pix_thr=0.15):
    """Detect a component if it is a meaningful fraction in >= pix_thr of pixels."""
    Xp = preprocess(cube)
    votes = np.zeros(len(COMPS))
    for y in Xp:
        B, _ = nnls(pures.T, y)
        frac = B / (B.sum() + 1e-12)
        votes += (frac >= frac_thr)
    frac_pix = votes / len(Xp)
    return set(np.where(frac_pix >= pix_thr)[0]), frac_pix


def matched_significance(mean, pures, k=3.0):
    """Keep component a if excluding it raises the NNLS residual by > k*noise."""
    y = preprocess(mean[None, :])[0]
    B_full, r_full = fit_B(y, pures)
    # noise proxy: robust std of the full-fit residual
    resid = y - B_full @ pures
    noise = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
    keep = set()
    for a in range(len(COMPS)):
        others = [i for i in range(len(COMPS)) if i != a]
        Bo, _ = nnls(pures[others].T, y)
        r_wo = np.linalg.norm(y - Bo @ pures[others])
        if (r_wo - r_full) / (noise * np.sqrt(len(y))) > k / np.sqrt(len(y)) * 3:
            keep.add(a)
    return keep


def main():
    ref = os.path.join(PEST_DEFAULT, "Reference")
    rat = os.path.join(PEST_DEFAULT, "Ratio")
    _, _, dq = load_map(os.path.join(ref, "DQ_corrected.csv"))
    _, _, thi = load_map(os.path.join(ref, "THI_corrected.csv"))
    _, _, tbz = load_map(os.path.join(ref, "TBZ_corrected.csv"))
    pures = preprocess(np.vstack([dq, thi, tbz]))

    names, cubes, means, true = [], [], [], []
    for k, nominal in MIXES.items():
        _, cube, mean = load_map(os.path.join(rat, k + "_corrected.csv"))
        names.append(k); cubes.append(cube); means.append(mean)
        true.append({i for i, c in enumerate(nominal) if c > 0})
    test_mean = preprocess(np.array(means))

    # (A) RandomForest on the mean
    rf = SERSMixtureClassifier(COMPS, prob_threshold=0.30, max_components=3,
                               augment=AugmentConfig(n_per_pure=200))
    rf.fit(pures)
    pred_rf = [set(np.where(names_to_indicator([p], COMPS)[0])[0])
               for p in rf.predict(test_mean)]

    # (B) per-pixel NNLS vote  &  (C) matched significance
    pred_pp = [per_pixel_vote(c, pures)[0] for c in cubes]
    pred_mf = [matched_significance(m, pures) for m in means]

    def scores(preds):
        yt = np.array([[1 if i in t else 0 for i in range(3)] for t in true])
        yp = np.array([[1 if i in p else 0 for i in range(3)] for p in preds])
        m = multilabel_metrics(yt, yp)
        exact = np.mean([t == p for t, p in zip(true, preds)])
        return m["micro_recall"], m["micro_precision"], m["micro_f1"], exact

    print(f"{'strategy':28s} {'recall':>7} {'prec':>6} {'F1':>6} {'exact':>7}")
    for lbl, preds in [("A) RandomForest (mean)", pred_rf),
                       ("B) per-pixel NNLS vote", pred_pp),
                       ("C) matched-filter (mean)", pred_mf)]:
        r, p, f, e = scores(preds)
        print(f"{lbl:28s} {r:7.2f} {p:6.2f} {f:6.2f} {e:7.2f}")

    print("\nper-mixture (true | RF | per-pixel | matched):")
    for nm, t, a, b, c in zip(names, true, pred_rf, pred_pp, pred_mf):
        print(f"  {nm:9s} {_combo(t):14s} {_combo(a):10s} {_combo(b):10s} {_combo(c):10s}")


if __name__ == "__main__":
    main()
