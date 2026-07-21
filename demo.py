"""
demo.py
=======
End-to-end demo: train the SERS mixture detector on PURE spectra only,
then evaluate it on synthetic binary/ternary mixtures built with a
competitive-adsorption (non-additive) model.

Run:  python demo.py
"""
import numpy as np
from synthetic import build_dataset
from sers_mixture import (
    SERSMixtureClassifier, AugmentConfig, preprocess,
    multilabel_metrics, names_to_indicator,
)


def main(seed=0):
    data = build_dataset(n_components=6, n_feat=500, seed=seed)
    names = data["names"]

    # ---- preprocess (same pipeline for pure + test) ----
    pure = preprocess(data["pure_raw"])
    test = preprocess(data["test_specs"])

    # ---- train on pure only ----
    clf = SERSMixtureClassifier(
        component_names=names,
        prob_threshold=0.18,
        max_components=3,
        nnls_rel_threshold=0.06,
        augment=AugmentConfig(n_per_pure=200, noise_frac=0.03,
                              shift_max=2, baseline_amp=0.05, seed=seed),
    )
    clf.fit(pure)

    # ---- predict on mixtures ----
    details = clf.predict(test, return_details=True)
    pred_names = [d["components"] for d in details]

    # ---- metrics ----
    true_names = [[names[i] for i in t] for t in data["test_true"]]
    Y_true = names_to_indicator(true_names, names)
    Y_pred = names_to_indicator(pred_names, names)
    m = multilabel_metrics(Y_true, Y_pred)

    print("=" * 62)
    print("SERS mixture component detection  (trained on PURE spectra only)")
    print("=" * 62)
    print(f"components         : {names}")
    print(f"test mixtures      : {len(test)}  (sizes 1,2,3)")
    print("-" * 62)
    for k, v in m.items():
        print(f"{k:20s}: {v:.3f}")
    print("-" * 62)

    # per-mixture-size exact-match breakdown
    sizes = np.array([len(t) for t in data["test_true"]])
    for sz in (1, 2, 3):
        mask = sizes == sz
        em = np.mean([set(pred_names[i]) == set(true_names[i])
                      for i in np.where(mask)[0]])
        print(f"exact-match | {sz}-component mixtures : {em:.3f}")
    print("-" * 62)

    # a few example predictions
    print("examples (true  ->  predicted [proportions]):")
    for i in range(0, len(test), max(1, len(test) // 8)):
        prop = {k: round(v, 2) for k, v in details[i]["proportions"].items()}
        print(f"  {true_names[i]}  ->  {pred_names[i]}   {prop}")
    print("=" * 62)


if __name__ == "__main__":
    main()
