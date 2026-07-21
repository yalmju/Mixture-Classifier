"""
run_on_my_data.py
=================
Template for running the detector on YOUR measured SERS spectra.

Expected CSV format  (comma-separated, one wavenumber axis shared by all):

  pure.csv
  --------
  wavenumber, Thiram, Paraquat, Ferbam, ...
  400.0,      0.01,   0.00,     0.02, ...
  402.8,      0.03,   0.01,     0.05, ...
  ...                                        (one column per PURE compound)

  mixtures.csv   (spectra to classify; component columns after wavenumber)
  ------------
  wavenumber, mix1, mix2, mix3, ...
  400.0,      0.04, 0.02, 0.07, ...
  ...

If your pure and mixture files use different wavenumber axes, set
RESAMPLE=True and everything is interpolated onto the pure-file axis.

Optionally provide ground-truth for the mixtures to get accuracy numbers:
edit TRUE_LABELS below (list of lists of compound names), or leave it None.
"""
import numpy as np
from sers_mixture import (
    SERSMixtureClassifier, AugmentConfig, preprocess,
    resample_to_axis, multilabel_metrics, names_to_indicator,
)

PURE_CSV = "pure.csv"
MIX_CSV = "mixtures.csv"
RESAMPLE = False
TRUE_LABELS = None        # e.g. [["Thiram", "Paraquat"], ["Ferbam"], ...]


def load_csv(path):
    import csv
    with open(path) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    names = [h.strip() for h in header[1:]]
    arr = np.array([[float(v) for v in r] for r in rows[1:]])
    axis = arr[:, 0]
    spectra = arr[:, 1:].T          # (n_spectra, n_feat)
    return axis, names, spectra


def main():
    ax_p, comp_names, pure_raw = load_csv(PURE_CSV)
    ax_m, mix_names, mix_raw = load_csv(MIX_CSV)

    if RESAMPLE:
        mix_raw = np.array([resample_to_axis(ax_m, m, ax_p) for m in mix_raw])

    pure = preprocess(pure_raw)
    mix = preprocess(mix_raw)

    clf = SERSMixtureClassifier(
        component_names=comp_names,
        prob_threshold=0.18,       # lower -> more sensitive (more recall)
        max_components=3,
        nnls_rel_threshold=0.06,   # lower -> keeps weaker components
        augment=AugmentConfig(n_per_pure=200, noise_frac=0.03,
                              shift_max=2, baseline_amp=0.05),
    )
    clf.fit(pure)

    details = clf.predict(mix, return_details=True)
    for name, d in zip(mix_names, details):
        prop = {k: round(v, 2) for k, v in d["proportions"].items()}
        print(f"{name:12s} -> {d['components']}   proportions={prop}")

    if TRUE_LABELS is not None:
        pred = [d["components"] for d in details]
        Yt = names_to_indicator(TRUE_LABELS, comp_names)
        Yp = names_to_indicator(pred, comp_names)
        print("\nmetrics:", multilabel_metrics(Yt, Yp))


if __name__ == "__main__":
    main()
