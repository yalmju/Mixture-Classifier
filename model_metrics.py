"""model_metrics.py
================
UI-agnostic training + evaluation for the Mixture classifier, so the GUI just
draws the numbers. Trains ``SERSMixtureClassifier`` on synthetic pure spectra
(zero real data needed) and returns everything a metrics dashboard needs:

    - micro precision / recall / F1, exact-match ratio, hamming loss
    - per-component precision / recall / F1
    - an N×N confusion matrix (single-component test spectra, argmax head)
    - a 2-D PCA embedding of the spectra, labelled, with the pure templates
      projected through the same PCA

Everything here is plain numpy / scikit-learn — no Qt, no tkinter — so it is
trivially testable and reusable across any front-end.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from synthetic import build_dataset
from sers_mixture import (
    preprocess, SERSMixtureClassifier, AugmentConfig,
    multilabel_metrics, names_to_indicator,
)


@dataclass
class MetricsResult:
    names: list[str]
    micro: dict                     # multilabel_metrics(...) output
    per_component: dict             # name -> (precision, recall, f1, support)
    confusion: np.ndarray           # (K, K) int, rows = true, cols = predicted
    pca_points: np.ndarray          # (n_single, 2)
    pca_labels: np.ndarray          # (n_single,) int component index
    pca_pure: np.ndarray            # (K, 2) projected pure templates
    n_train: int
    n_test: int


def compute_metrics(n_components: int = 6, threshold: float = 0.30,
                    max_components: int = 3, n_per_pure: int = 120,
                    seed: int = 0) -> MetricsResult:
    data = build_dataset(n_components=n_components, seed=seed)
    names = data["names"]
    K = len(names)

    pures = preprocess(data["pure_raw"])
    clf = SERSMixtureClassifier(
        names, prob_threshold=threshold, max_components=max_components,
        augment=AugmentConfig(n_per_pure=n_per_pure), random_state=seed)
    clf.fit(pures)

    test = preprocess(data["test_specs"])
    true_names = [[names[i] for i in t] for t in data["test_true"]]
    pred_names = clf.predict(test)

    y_true = names_to_indicator(true_names, names)
    y_pred = names_to_indicator(pred_names, names)
    micro = multilabel_metrics(y_true, y_pred)

    per = {}
    for i, nm in enumerate(names):
        tp = int(((y_true[:, i] == 1) & (y_pred[:, i] == 1)).sum())
        fp = int(((y_true[:, i] == 0) & (y_pred[:, i] == 1)).sum())
        fn = int(((y_true[:, i] == 1) & (y_pred[:, i] == 0)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        per[nm] = (p, r, f, tp + fn)

    # confusion matrix on the single-component test spectra (argmax head)
    probs = clf.predict_proba(test)
    single = [k for k, t in enumerate(data["test_true"]) if len(t) == 1]
    cm = np.zeros((K, K), dtype=int)
    for k in single:
        cm[data["test_true"][k][0], int(np.argmax(probs[k]))] += 1

    # PCA of the single-component spectra (clean clusters) + pure templates
    Xs = test[single]
    labels = np.array([data["test_true"][k][0] for k in single], dtype=int)
    pca = PCA(n_components=2, random_state=seed)
    emb = pca.fit_transform(np.vstack([Xs, pures]))

    return MetricsResult(
        names=names, micro=micro, per_component=per, confusion=cm,
        pca_points=emb[:len(Xs)], pca_labels=labels, pca_pure=emb[len(Xs):],
        n_train=n_components * n_per_pure, n_test=len(test))


if __name__ == "__main__":
    r = compute_metrics()
    print("micro:", {k: round(v, 3) for k, v in r.micro.items()})
    print("components:", r.names)
    print("confusion:\n", r.confusion)
    print("PCA points:", r.pca_points.shape, "labels:", r.pca_labels.shape)
