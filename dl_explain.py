"""dl_explain.py — interpretability of the Recovery panel's DL composition model.

Trains the full-spectrum composition MLP on the loaded known-ratio mixtures, then returns
three consistency checks (no figure — the UI plots the returned data):
  attr  : Integrated-Gradients per-wavenumber attribution per compound (SHAP-style)
  perm  : band permutation importance (Δ composition error when a window is shuffled)
  abl   : ligand ablation (predicted fraction before/after zeroing a compound's VIP bands)

UI-agnostic (numpy; torch lazily inside dl_quantify).
"""
from __future__ import annotations

import os
import numpy as np


def dl_explain(data_dir, items, calib_path=None, baseline=True, trim=None, progress=None):
    from unmix import _templates, _baseline_removed, _l2, compute_vip_bands
    from real_data import load_map
    from dataset import is_blank
    from sers_mixture import als_baseline
    from composition import SUBSTANCES
    from dl_quantify import simulate_mixtures, train_composition, _spec_net, _ratio
    import torch

    names, wn, means = _templates(data_dir, baseline, None)
    lo, hi = trim if trim else (300.0, 1800.0)
    mask = (wn >= lo) & (wn <= hi)
    if mask.sum() < 10:
        mask = np.ones(len(wn), bool)
    wnv = wn[mask]
    idx = {n: i for i, n in enumerate(names)}
    subs = [s for s in SUBSTANCES if s in idx and not is_blank(s)]
    if len(subs) < 2:
        raise ValueError("DL explain needs ≥2 reference substances matching the mixtures.")
    P = _l2(_baseline_removed(means[[idx[s] for s in subs]][:, mask], baseline))
    nfeat = P.shape[1]
    vip = compute_vip_bands(data_dir, baseline=baseline, trim=(lo, hi))[1]

    def mix_spec(path):
        _w, cube, _m, _c = load_map(path)
        cube = np.asarray(cube, float)[:, mask]
        w = cube.sum(1); w = w / (w.sum() + 1e-12); mean = w @ cube
        y = np.clip(mean - als_baseline(mean), 0, None)
        return y / (np.linalg.norm(y) + 1e-12)

    if progress:
        progress("DL explain — reading spectra")
    X, Y = [], []
    for it in items:
        path, ratio = it[0], it[1]
        vec = _ratio([float(ratio.get(s, 0.0)) for s in subs])
        if vec.sum() > 0:
            X.append(mix_spec(path)); Y.append(vec)
    X = np.array(X, np.float32); Y = np.array(Y, np.float32); N = len(X)
    if N < 3:
        raise ValueError("DL explain needs ≥3 mixtures.")

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
        except Exception:
            pre = None

    if progress:
        progress("DL explain — training model")
    model = train_composition(X, Y, len(subs), pretrain=pre, seed=0)
    net = _spec_net(model["n_feat"], model["n_comp"]); net.load_state_dict(model["state"]); net.eval()

    def predict(Xa):
        with torch.no_grad():
            return torch.softmax(net(torch.tensor(np.atleast_2d(Xa).astype(np.float32))), 1).numpy()

    # (1) Integrated Gradients
    def ig(x, c, steps=40):
        xt = torch.tensor(x, dtype=torch.float32); base = torch.zeros_like(xt); tot = torch.zeros_like(xt)
        for k in range(1, steps + 1):
            s = (base + (k / steps) * (xt - base)).unsqueeze(0).requires_grad_(True)
            out = torch.softmax(net(s), 1)[0, c]
            g, = torch.autograd.grad(out, s); tot += g[0]
        return ((xt - base) * (tot / steps)).detach().numpy()
    attr = {}
    for c, name in enumerate(subs):
        if progress:
            progress(f"DL explain — attribution {name}")
        pres = np.where(Y[:, c] > 0.05)[0]
        A = np.mean([np.abs(ig(X[i], c)) for i in pres], axis=0)
        attr[name] = A / (A.max() + 1e-12)

    # (2) band permutation importance
    if progress:
        progress("DL explain — permutation importance")
    def cerr(Pm): return float(np.mean([0.5 * np.abs(Pm[i] - Y[i]).sum() for i in range(N)]))
    E0 = cerr(predict(X)); edges = np.arange(lo, hi + 1, 60.0); perm = []
    rngp = np.random.default_rng(1)
    for a, b in zip(edges[:-1], edges[1:]):
        reg = (wnv >= a) & (wnv < b)
        if reg.sum() == 0:
            continue
        d = [cerr(predict(np.where(reg, X[rngp.permutation(N)], X))) - E0 for _ in range(4)]
        perm.append(((a + b) / 2, float(np.mean(d))))

    # (3) ligand ablation
    abl = {}
    base_pred = predict(X)
    for c, name in enumerate(subs):
        reg = np.zeros(nfeat, bool)
        for wv in vip.get(name, []):
            reg |= (np.abs(wnv - wv) <= 10)
        Xa = X.copy(); Xa[:, reg] = 0.0
        pres = np.where(Y[:, c] > 0.05)[0]
        abl[name] = (float(base_pred[pres, c].mean()), float(predict(Xa)[pres, c].mean()))

    return {"wn": wnv, "subs": subs, "attr": attr, "vip": vip,
            "perm": np.array(perm), "abl": abl}
