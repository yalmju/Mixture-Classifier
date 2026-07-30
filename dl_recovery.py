"""dl_recovery.py — physics-informed DL composition prediction for the Recovery panel.

Leave-one-map-out over the loaded known-ratio mixtures: for each mixture, train the
full-spectrum composition model on all the OTHERS (plus a physics-simulated pretrain
from the references + calibration) and predict the held-out one. Returns per-mixture
``{name, nominal, mean}`` (composition in SUBSTANCES order), so the result feeds the
SAME drift-triangle / recovery / relative-drift plots as the classical NNLS path.

UI-agnostic (numpy; torch is imported lazily inside dl_quantify).
"""
from __future__ import annotations

import os
import numpy as np


def dl_recovery(data_dir, items, calib_path=None, baseline=True, trim=None, progress=None):
    """items: list of (map_path, true_ratio_dict[, true_conc]). Returns a list of
    {name, nominal, mean} in composition.SUBSTANCES order (leave-one-map-out DL)."""
    from unmix import _templates, _baseline_removed, _l2
    from real_data import load_map
    from dataset import is_blank
    from sers_mixture import als_baseline
    from composition import SUBSTANCES
    from dl_quantify import (simulate_mixtures, train_composition, predict_composition, _ratio)

    names, wn, means = _templates(data_dir, baseline, None)
    lo, hi = trim if trim else (300.0, 1800.0)
    mask = (wn >= lo) & (wn <= hi)
    if mask.sum() < 10:
        mask = np.ones(len(wn), bool)
    idx = {n: i for i, n in enumerate(names)}
    subs = [s for s in SUBSTANCES if s in idx and not is_blank(s)]
    if len(subs) < 2:
        raise ValueError("DL recovery needs ≥2 reference substances matching the mixtures.")
    P = _l2(_baseline_removed(means[[idx[s] for s in subs]][:, mask], baseline))

    def mix_spec(path):
        _w, cube, _m, _c = load_map(path)
        cube = np.asarray(cube, float)[:, mask]
        w = cube.sum(1); w = w / (w.sum() + 1e-12)
        mean = w @ cube                                   # intensity-weighted mean pixel
        y = np.clip(mean - als_baseline(mean), 0, None)
        return y / (np.linalg.norm(y) + 1e-12)

    X, Y, labels = [], [], []
    for it in items:
        path, ratio = it[0], it[1]
        vec = _ratio([float(ratio.get(s, 0.0)) for s in subs])
        if vec.sum() <= 0:
            continue
        X.append(mix_spec(path)); Y.append(vec)
        labels.append(os.path.basename(path).replace("_corrected", "").replace(".csv", ""))
    X = np.array(X); Y = np.array(Y); N = len(X)
    if N < 2:
        raise ValueError("DL recovery needs ≥2 mixtures (leave-one-out).")

    # physics pretrain from the calibration (K, gA) — skipped gracefully if unavailable
    pre = None
    if calib_path:
        try:
            from io_utils import load_calibration_csv
            from calibration import calibrate
            ax_c, names_c, dils = load_calibration_csv(calib_path)
            mc = (ax_c >= lo) & (ax_c <= hi)
            dil = [(dils[names_c.index(s)][0], np.asarray(dils[names_c.index(s)][1])[:, mc])
                   for s in subs if s in names_c]
            if len(dil) == len(subs):
                cal = calibrate(dil, P, subs)
                rng = np.random.default_rng(0)
                Xs, Cs = simulate_mixtures(P, cal.K, cal.gA, 5000, rng,
                                           noise=0.015, baseline=0.03, gain_lo=0.8, gain_hi=1.25)
                Xs = np.array([r / (np.linalg.norm(r) + 1e-12) for r in Xs]).astype(np.float32)
                pre = (Xs, np.array([_ratio(c) for c in Cs]).astype(np.float32))
        except Exception as e:                            # bad/mismatched calibration → real-only
            if progress:
                progress(f"DL: calibration pretrain skipped ({e})")

    order = [SUBSTANCES.index(s) for s in subs]
    out = []
    for i in range(N):
        if progress:
            progress(f"DL leave-one-out — mixture {i + 1}/{N}  ({N - i - 1} left)")
        tr = [j for j in range(N) if j != i]
        model = train_composition(X[tr], Y[tr], len(subs), pretrain=pre, seed=i)
        pred = predict_composition(model, X[i])[0]
        nom = np.zeros(len(SUBSTANCES)); mn = np.zeros(len(SUBSTANCES))
        for k, o in enumerate(order):
            nom[o] = Y[i][k]; mn[o] = pred[k]
        out.append({"name": labels[i], "nominal": nom, "mean": mn})
    return out
