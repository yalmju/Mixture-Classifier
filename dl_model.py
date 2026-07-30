"""dl_model.py — a portable physics-informed DL model that Recovery TRAINS (from
known-ratio mixtures) and Real-data APPLIES (to an unknown test map).

Holds a spectrum→composition head and, when absolute concentrations were supplied, a
spectrum→log10 µM head. Everything needed to score a new map is baked in (unit templates
P, the spectral window, per-feature standardisation), so Real-data needs no labels.

Save/load via pickle (torch states stored as numpy). UI-agnostic (torch lazy).
"""
from __future__ import annotations

import os
import pickle
import numpy as np


def _refs(data_dir, baseline, trim):
    from unmix import _templates, _baseline_removed, _l2
    from dataset import is_blank
    from composition import SUBSTANCES
    names, wn, means = _templates(data_dir, baseline, None)
    lo, hi = trim if trim else (300.0, 1800.0)
    mask = (wn >= lo) & (wn <= hi)
    if mask.sum() < 10:
        mask = np.ones(len(wn), bool)
    idx = {n: i for i, n in enumerate(names)}
    subs = [s for s in SUBSTANCES if s in idx and not is_blank(s)]
    P = _l2(_baseline_removed(means[[idx[s] for s in subs]][:, mask], baseline))
    return subs, wn, mask, P, lo, hi


def _mean_spectrum(cube, mask):
    cube = np.asarray(cube, float)[:, mask]
    w = cube.sum(1); w = w / (w.sum() + 1e-12)
    from sers_mixture import als_baseline
    mean = w @ cube
    return np.clip(mean - als_baseline(mean), 0, None)


def train_model(data_dir, items, calib_path=None, baseline=True, trim=None, progress=None,
                method="mlp", epochs=350, seed=0, use_pretrain=True,
                n_components=8, n_trees=300):
    """Train a composition model (+ µM if absolute concentrations given) on ALL mixtures.
    items: (path, ratio_dict[, conc_dict in M]). Returns a portable model dict.

    ``method`` picks the composition head:
      - "mlp"  physics-informed deep net — knobs: epochs, seed, use_pretrain
      - "pls"  PLS regression           — knob: n_components
      - "rf"   random forest            — knobs: n_trees, seed
    The µM head (order-of-magnitude concentration) is the same small MLP regardless."""
    import torch, torch.nn as nn
    from real_data import load_map
    from dl_quantify import simulate_mixtures, train_composition, _spec_net, _ratio
    subs, wn, mask, P, lo, hi = _refs(data_dir, baseline, trim)
    if len(subs) < 2:
        raise ValueError("need ≥2 reference substances.")
    X, Xabs, Y, Cabs = [], [], [], []
    for it in items:
        ratio = it[1]; conc = it[2] if len(it) > 2 else None
        vec = _ratio([float(ratio.get(s, 0.0)) for s in subs])
        if vec.sum() <= 0:
            continue
        _w, cube, _m, _c = load_map(it[0])
        ya = _mean_spectrum(cube, mask)
        X.append(ya / (np.linalg.norm(ya) + 1e-12)); Xabs.append(ya); Y.append(vec)
        Cabs.append([float(conc.get(s, 0.0)) for s in subs] if conc else None)
    X = np.array(X, np.float32); Xabs = np.array(Xabs); Y = np.array(Y, np.float32)
    if len(X) < 3:
        raise ValueError("need ≥3 mixtures to train.")

    pre = None
    if calib_path and use_pretrain:
        try:
            from io_utils import load_calibration_csv
            from calibration import calibrate
            ax_c, nc, dils = load_calibration_csv(calib_path); mc = (ax_c >= lo) & (ax_c <= hi)
            dil = [(dils[nc.index(s)][0], np.asarray(dils[nc.index(s)][1])[:, mc]) for s in subs if s in nc]
            if len(dil) == len(subs):
                cal = calibrate(dil, P, subs); rng = np.random.default_rng(0)
                Xs, Cs = simulate_mixtures(P, cal.K, cal.gA, 5000, rng, noise=0.015, baseline=0.03, gain_lo=0.8, gain_hi=1.25)
                Xs = np.array([r / (np.linalg.norm(r) + 1e-12) for r in Xs]).astype(np.float32)
                pre = (Xs, np.array([_ratio(c) for c in Cs]).astype(np.float32))
        except Exception:
            pre = None

    method = (method or "mlp").lower()
    if progress:
        progress(f"training composition head ({method})")
    if method == "pls":
        from sklearn.cross_decomposition import PLSRegression
        nc = max(1, min(int(n_components), len(X) - 1, X.shape[1]))
        comp_store = {"method": "pls", "sk": PLSRegression(n_components=nc).fit(X, Y)}
    elif method == "rf":
        from sklearn.ensemble import RandomForestRegressor
        comp_store = {"method": "rf",
                      "sk": RandomForestRegressor(n_estimators=int(n_trees),
                                                  random_state=int(seed)).fit(X, Y)}
    else:
        method = "mlp"
        comp = train_composition(X, Y, len(subs), pretrain=pre, seed=seed, epochs_ft=epochs)
        comp_store = {"method": "mlp",
                      "comp_state": {k: v.cpu().numpy() for k, v in comp["state"].items()},
                      "comp_hidden": (256, 64)}

    uM = None
    have = [i for i in range(len(X)) if Cabs[i] is not None and any(c > 0 for c in Cabs[i])]
    if len(have) >= 3:
        if progress:
            progress("training concentration head")
        hv = np.array(have); C = np.array([Cabs[i] if Cabs[i] is not None else [0.0] * len(subs)
                                           for i in range(len(X))], float)
        mu = Xabs[hv].mean(0); sd = Xabs[hv].std(0) + 1e-8
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(Xabs.shape[1], 256), nn.BatchNorm1d(256), nn.ReLU(),
                            nn.Dropout(0.15), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, len(subs)))
        op = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-3)
        Xt = torch.tensor(((Xabs[hv] - mu) / sd).astype(np.float32))
        Yt = torch.tensor((np.log10(np.clip(C[hv], 1e-8, None)) + 6.0).astype(np.float32))
        for _ in range(epochs):
            net.train(); op.zero_grad(); ((net(Xt) - Yt) ** 2).mean().backward(); op.step()
        net.eval()
        uM = {"state": {k: v.detach().numpy() for k, v in net.state_dict().items()},
              "mu": mu, "sd": sd}

    return {"subs": subs, "lo": lo, "hi": hi, "n_feat": int(mask.sum()), "P": P,
            "uM": uM, "n_train": int(len(X)), "has_uM": uM is not None, **comp_store}


def apply_model(model, wn, cube):
    """Score a test map (or single spectrum): returns {composition: {name: frac},
    uM: {name: µM} or None}. Handles the model's own window + standardisation."""
    lo, hi = model["lo"], model["hi"]
    mask = (np.asarray(wn) >= lo) & (np.asarray(wn) <= hi)
    ya = _mean_spectrum(cube, mask)
    subs = model["subs"]
    xl2 = (ya / (np.linalg.norm(ya) + 1e-12)).astype(np.float32)
    method = model.get("method", "mlp")
    if method in ("pls", "rf"):                                # sklearn composition head
        comp = np.clip(np.asarray(model["sk"].predict(xl2[None, :])[0], float), 0, None)
        comp = comp / (comp.sum() + 1e-12)
    else:                                                      # MLP (softmax) head
        import torch
        from dl_quantify import _spec_net
        net = _spec_net(model["n_feat"], len(subs), model["comp_hidden"])
        net.load_state_dict({k: torch.tensor(v) for k, v in model["comp_state"].items()}); net.eval()
        with torch.no_grad():
            comp = torch.softmax(net(torch.tensor(xl2[None, :])), 1).numpy()[0]
    out = {"composition": {subs[k]: float(comp[k]) for k in range(len(subs))}, "uM": None}
    if model.get("uM"):
        import torch.nn as nn
        u = model["uM"]
        net2 = nn.Sequential(nn.Linear(model["n_feat"], 256), nn.BatchNorm1d(256), nn.ReLU(),
                             nn.Dropout(0.15), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, len(subs)))
        net2.load_state_dict({k: torch.tensor(v) for k, v in u["state"].items()}); net2.eval()
        xe = ((ya - u["mu"]) / u["sd"]).astype(np.float32)
        with torch.no_grad():
            logv = net2(torch.tensor(xe[None, :])).numpy()[0]
        out["uM"] = {subs[k]: float(10.0 ** np.clip(logv[k], -3.0, 6.0)) for k in range(len(subs))}
    return out


def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)
