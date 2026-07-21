"""run_real_resnet.py — RandomForest vs ResNet1D on the REAL pesticide mixtures.

Both detectors are trained on the 3 pure MEAN spectra (+augmentation, no real
mixture seen) and asked to detect components in the 10 real ratio mixtures.
Honest head-to-head on whether the deep model helps the hard (mixture) case.

    python run_real_resnet.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import numpy as np

from sers_mixture import (preprocess, SERSMixtureClassifier, AugmentConfig,
                          multilabel_metrics, names_to_indicator)
from resnet1d import ResNet1DDetector
from real_data import load_map, PEST_DEFAULT, COMPS, MIXES


def _combo(names):
    order = [c for c in COMPS if c in names]
    return "+".join(order) if order else "(none)"


def main():
    ref = os.path.join(PEST_DEFAULT, "Reference")
    rat = os.path.join(PEST_DEFAULT, "Ratio")
    _, _, dq = load_map(os.path.join(ref, "DQ_corrected.csv"))
    _, _, thi = load_map(os.path.join(ref, "THI_corrected.csv"))
    _, _, tbz = load_map(os.path.join(ref, "TBZ_corrected.csv"))
    pures = preprocess(np.vstack([dq, thi, tbz]))

    names, means, true = [], [], []
    for k, nominal in MIXES.items():
        _, _, mean = load_map(os.path.join(rat, k + "_corrected.csv"))
        names.append(k); means.append(mean)
        true.append([COMPS[i] for i, c in enumerate(nominal) if c > 0])
    test = preprocess(np.array(means))
    yt = names_to_indicator(true, COMPS)

    aug = AugmentConfig(n_per_pure=200, noise_frac=0.03, shift_max=2,
                        baseline_amp=0.05, seed=0)

    # ---- RandomForest (current pipeline) ----
    rf = SERSMixtureClassifier(COMPS, prob_threshold=0.30, max_components=3, augment=aug)
    rf.fit(pures)
    pred_rf = rf.predict(test)

    # ---- ResNet1D ----
    print("[ResNet1D] training on 3 pure means + augmentation (CPU)…")
    rn = ResNet1DDetector(COMPS, prob_threshold=0.5, epochs=40, base=16, augment=aug)
    rn.fit(pures, verbose=False)
    pred_rn = rn.predict(test)

    m_rf = multilabel_metrics(yt, names_to_indicator(pred_rf, COMPS))
    m_rn = multilabel_metrics(yt, names_to_indicator(pred_rn, COMPS))
    ex_rf = np.mean([_combo(t) == _combo(p) for t, p in zip(true, pred_rf)])
    ex_rn = np.mean([_combo(t) == _combo(p) for t, p in zip(true, pred_rn)])

    print("\n" + "=" * 58)
    print(f"{'metric':22s} | {'RandomForest':>12} | {'ResNet1D':>10}")
    print("-" * 58)
    for k in m_rf:
        print(f"{k:22s} | {m_rf[k]:12.3f} | {m_rn[k]:10.3f}")
    print(f"{'exact composition':22s} | {ex_rf:12.3f} | {ex_rn:10.3f}")
    print("=" * 58)

    print("\nper-mixture detection:")
    print(f"{'mixture':9s} {'true':14s} {'RF':14s} {'ResNet1D':14s}")
    for nm, t, prf, prn in zip(names, true, pred_rf, pred_rn):
        print(f"{nm:9s} {_combo(t):14s} {_combo(prf):14s} {_combo(prn):14s}")


if __name__ == "__main__":
    main()
