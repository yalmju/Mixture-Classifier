"""classify.py — apply a model TRAINED in the Model tab (saved as
unmixr_model.joblib) to a new test map: classify every pixel and return a
per-pixel class map + confidence, in the same shape the Real-data tab already
draws (so the composition pie, per-pixel glyphs and pixel spectrum all work).

The class-probability vector per pixel plays the role of the composition vector;
the max probability is the confidence (used as the reliability channel).
"""
from __future__ import annotations

import numpy as np

from real_data import load_map
from dataset import is_blank
from model_training import _featurize
from unmix import UnmixResult, _baseline_removed


def classify_map(model_path, test_path, min_conf=0.0, progress=None) -> UnmixResult:
    """Load the trained-model bundle at ``model_path`` and classify each pixel of
    ``test_path``. Preprocessing matches what the model was trained with (stored in
    the bundle). ``min_conf`` is the confidence a pixel needs to count as a hit."""
    import joblib
    bundle = joblib.load(model_path)
    model = bundle.get("model")
    classes = list(bundle.get("classes", []))
    if model is None or not classes:
        raise ValueError(f"{model_path} is not a UNMIXR model bundle.")
    if not hasattr(model, "predict"):
        raise ValueError("this saved model can't be applied here — export an "
                         "sklearn / PLS-DA model (the ResNet backend isn't supported).")
    pp = bundle.get("preprocessing") or {}
    baseline = pp.get("baseline", True); trim = pp.get("trim")
    deriv = pp.get("deriv", 0); norm = pp.get("norm", "l2")

    if progress:
        progress("loading test map")
    wn, cube, _mean, coord = load_map(test_path)
    if trim is not None:
        lo, hi = trim
        m = (wn >= lo) & (wn <= hi)
        if m.sum() >= 10:
            wn = wn[m]; cube = cube[:, m]

    if progress:
        progress("classifying pixels")
    X = _featurize(cube, baseline=baseline, deriv=deriv, norm=norm)
    K = len(classes)
    A = np.zeros((len(X), K))
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        for j, lbl in enumerate(np.asarray(model.classes_)):
            A[:, int(lbl)] = proba[:, j]
    else:                                               # hard labels → one-hot
        pred = np.asarray(model.predict(X), int)
        A[np.arange(len(X)), pred] = 1.0

    bg_mask = np.array([is_blank(c) for c in classes])
    nonbg = [i for i in range(K) if not bg_mask[i]]
    spectra = _baseline_removed(cube, baseline)

    conf = A.max(axis=1)                                # max class probability
    dom = A.argmax(axis=1)
    hit = np.array([(not bg_mask[dom[i]]) and conf[i] >= min_conf
                    for i in range(len(X))])
    Anb = A[:, nonbg]
    nb_tot = Anb.sum(axis=1, keepdims=True)
    ratio_nb = np.divide(Anb, nb_tot, out=np.zeros_like(Anb), where=nb_tot > 0)
    hit_frac = float(hit.mean())
    mean_ratio = ratio_nb[hit].mean(axis=0) if hit.any() else ratio_nb.mean(axis=0)
    dominant = [classes[i] for i in nonbg][int(mean_ratio.argmax())] if nonbg else classes[0]

    return UnmixResult(
        comps=classes, bg_mask=bg_mask, nonbg=nonbg, method="model", wn=wn,
        coords=coord, spectra=spectra.astype(np.float32), templates=None,
        A=A, ratio_nb=ratio_nb, hit=hit, reliab=conf, n_pixels=len(X),
        hit_frac=hit_frac, mean_ratio=mean_ratio, dominant=dominant,
        mean_r2=float(conf.mean()))
