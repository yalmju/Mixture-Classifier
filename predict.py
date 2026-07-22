"""predict.py — apply the reference set to an unknown sample and read its
component ratio.

The simplified "Map tool": you already organised references (Samples) and can
train on them (Model); here you just load ONE unknown sample map and get the
composition ratio of the known substances in it — mean-spectrum NNLS unmixing for
the ratio, plus a per-pixel vote for robust presence detection.

UI-agnostic (numpy / scikit-learn), so the Qt tab just draws the numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from real_data import load_map, per_pixel_vote
from dataset import discover_dataset, is_blank
from sers_mixture import preprocess, SERSMixtureClassifier, AugmentConfig


@dataclass
class PredictResult:
    comps: list                 # reference substances (non-blank)
    ratio: dict                 # name -> proportion (detected components, ~sums to 1)
    detected: list              # component names present (classifier + per-pixel vote)
    proba: dict                 # name -> stage-1 evidence probability
    wn: np.ndarray              # wavenumber axis
    mean_spectrum: np.ndarray   # preprocessed unknown mean spectrum
    templates: np.ndarray       # (K, n_feat) preprocessed reference templates
    n_pixels: int


def predict_sample(data_dir, sample_path, threshold=0.30, baseline=True,
                   trim=None) -> PredictResult:
    """Load the reference substances from ``data_dir`` (Samples grouping) and the
    unknown map at ``sample_path``, and return the estimated component ratio."""
    groups = discover_dataset(data_dir)
    comps = [c for c, _ in groups if not is_blank(c)]
    if not comps:
        raise FileNotFoundError(
            f"no substance references found in {data_dir} (need at least one "
            "non-blank class).")
    path_of = {c: [p for _b, p in maps] for c, maps in groups}

    # reference templates: mean spectrum per substance (across its batch maps)
    means, wn = [], None
    for c in comps:
        cbs = []
        for p in path_of[c]:
            wn, cube, _m, _c = load_map(p)
            cbs.append(cube)
        means.append(np.vstack(cbs).mean(axis=0))
    means = np.array(means)

    # unknown sample
    wn_u, cube_u, mean_u, _coord = load_map(sample_path)

    if trim is not None:
        lo, hi = trim
        m = (wn >= lo) & (wn <= hi)
        if m.sum() >= 10:
            means = means[:, m]; mean_u = mean_u[m]; cube_u = cube_u[:, m]; wn = wn[m]

    pures = preprocess(means, do_baseline=baseline)
    mean_f = preprocess(mean_u[None, :], do_baseline=baseline)[0]

    clf = SERSMixtureClassifier(comps, prob_threshold=threshold,
                                max_components=len(comps),
                                augment=AugmentConfig(n_per_pure=200))
    clf.fit(pures)
    detail = clf.predict(mean_f[None, :], return_details=True)[0]

    voted = per_pixel_vote(cube_u, pures, comps)          # robust presence
    detected = sorted(set(detail["components"]) | {comps[i] for i in voted})

    return PredictResult(
        comps=comps, ratio=detail["proportions"], detected=detected,
        proba=detail["proba"], wn=wn, mean_spectrum=mean_f, templates=pures,
        n_pixels=len(cube_u))


if __name__ == "__main__":
    import sys
    r = predict_sample(sys.argv[1], sys.argv[2])
    print("detected:", r.detected)
    print("ratio:", {k: round(v, 3) for k, v in r.ratio.items()})
