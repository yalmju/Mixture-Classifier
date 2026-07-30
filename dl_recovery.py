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


def dl_recovery(data_dir, items, calib_path=None, baseline=True, trim=None, progress=None,
                method="mlp", epochs=350, seed=0, use_pretrain=True,
                n_components=8, n_trees=300):
    """items: list of (map_path, true_ratio_dict[, true_conc]). Returns a list of
    {name, nominal, mean[, uM_true, uM_pred]} in composition.SUBSTANCES order
    (leave-one-map-out DL). uM_* only when absolute concentrations (µM) were supplied for
    ≥3 mixtures — the DL then also predicts an ORDER-OF-MAGNITUDE µM per component.

    Training knobs: ``epochs`` (fine-tune iterations), ``seed`` (base RNG seed; each
    left-out mixture uses seed+i), ``use_pretrain`` (physics simulator warm-up; needs a
    calibration — ignored when none is loaded)."""
    import torch
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

    def mix_mean(path):
        _w, cube, _m, _c = load_map(path)
        cube = np.asarray(cube, float)[:, mask]
        w = cube.sum(1); w = w / (w.sum() + 1e-12)
        mean = w @ cube                                   # intensity-weighted mean pixel
        return np.clip(mean - als_baseline(mean), 0, None)   # un-normalised (keeps magnitude)

    X, Xabs, Y, Cabs, labels = [], [], [], [], []
    for it in items:
        path, ratio = it[0], it[1]
        conc = it[2] if len(it) > 2 else None
        vec = _ratio([float(ratio.get(s, 0.0)) for s in subs])
        if vec.sum() <= 0:
            continue
        ya = mix_mean(path)
        X.append(ya / (np.linalg.norm(ya) + 1e-12)); Xabs.append(ya); Y.append(vec)
        Cabs.append([float(conc.get(s, 0.0)) for s in subs] if conc else None)
        labels.append(os.path.basename(path).replace("_corrected", "").replace(".csv", ""))
    X = np.array(X); Xabs = np.array(Xabs); Y = np.array(Y); N = len(X)
    if N < 2:
        raise ValueError("DL recovery needs ≥2 mixtures (leave-one-out).")

    # physics pretrain from the calibration (K, gA) — skipped gracefully if unavailable
    pre = None
    if calib_path and use_pretrain:
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

    # optional: order-of-magnitude µM per component (needs absolute concentrations, ≥3 mixtures)
    have = [i for i in range(N) if Cabs[i] is not None and any(c > 0 for c in Cabs[i])]
    uM = None
    if len(have) >= 3:
        import torch.nn as nn
        C = np.array([Cabs[i] if Cabs[i] is not None else [0.0] * len(subs) for i in range(N)], float)
        uM = np.full((N, len(subs)), np.nan)
        hv = np.array(have)
        for n_i, i in enumerate(have):
            if progress:
                progress(f"DL µM leave-one-out — {n_i + 1}/{len(have)}")
            tr = hv[hv != i]
            mu = Xabs[tr].mean(0); sd = Xabs[tr].std(0) + 1e-8
            Xt = ((Xabs[tr] - mu) / sd).astype(np.float32); Xe = ((Xabs[i] - mu) / sd).astype(np.float32)
            Yt = (np.log10(np.clip(C[tr], 1e-8, None)) + 6.0).astype(np.float32)   # log10 µM
            torch.manual_seed(seed + i)
            net = nn.Sequential(nn.Linear(Xt.shape[1], 256), nn.BatchNorm1d(256), nn.ReLU(),
                                nn.Dropout(0.15), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, len(subs)))
            op = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-3)
            Xtt = torch.tensor(Xt); Ytt = torch.tensor(Yt)
            for _ in range(epochs):
                net.train(); op.zero_grad(); ((net(Xtt) - Ytt) ** 2).mean().backward(); op.step()
            net.eval()
            with torch.no_grad():
                logv = net(torch.tensor(Xe[None, :])).numpy()[0]
            uM[i] = 10.0 ** np.clip(logv, -3.0, 6.0)      # clip log10 µM → avoid overflow

    order = [SUBSTANCES.index(s) for s in subs]
    out = []
    for i in range(N):
        if progress:
            progress(f"DL leave-one-out — mixture {i + 1}/{N}  ({N - i - 1} left)")
        tr = [j for j in range(N) if j != i]
        if method == "pls":
            from sklearn.cross_decomposition import PLSRegression
            nc = max(1, min(int(n_components), len(tr) - 1, X.shape[1]))
            p = np.clip(PLSRegression(n_components=nc).fit(X[tr], Y[tr]).predict(X[i][None, :])[0], 0, None)
            pred = p / (p.sum() + 1e-12)
        elif method == "rf":
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=int(n_trees), random_state=int(seed)).fit(X[tr], Y[tr])
            p = np.clip(rf.predict(X[i][None, :])[0], 0, None); pred = p / (p.sum() + 1e-12)
        elif method == "cnn":
            from dl_model import _cnn
            torch.manual_seed(seed + i); net = _cnn(X.shape[1], len(subs))
            sm = torch.nn.LogSoftmax(dim=1)
            op = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-3)
            Xt = torch.tensor(X[tr].astype(np.float32)); Yt = torch.tensor(Y[tr].astype(np.float32))
            wt = 1.0 + 2.0 * (1.0 - Yt)
            for _ in range(int(epochs)):
                net.train(); op.zero_grad()
                (wt * (sm(net(Xt)).exp() - Yt).abs()).sum(1).mean().backward(); op.step()
            net.eval()
            with torch.no_grad():
                pred = torch.softmax(net(torch.tensor(X[i][None, :].astype(np.float32))), 1).numpy()[0]
        else:
            model = train_composition(X[tr], Y[tr], len(subs), pretrain=pre, seed=seed + i, epochs_ft=epochs)
            pred = predict_composition(model, X[i])[0]
        nom = np.zeros(len(SUBSTANCES)); mn = np.zeros(len(SUBSTANCES))
        for k, o in enumerate(order):
            nom[o] = Y[i][k]; mn[o] = pred[k]
        row = {"name": labels[i], "nominal": nom, "mean": mn}
        if uM is not None and not np.isnan(uM[i]).all():
            row["uM_true"] = {subs[k]: Cabs[i][k] * 1e6 for k in range(len(subs)) if Cabs[i][k] > 0}
            row["uM_pred"] = {subs[k]: float(uM[i, k]) for k in range(len(subs)) if Cabs[i][k] > 0}
        out.append(row)
    return out
