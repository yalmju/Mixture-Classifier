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


def _cnn(n_feat, n_comp):
    """1-D CNN over the spectrum → composition logits. Same builder for train & apply so
    a saved state_dict reloads. (torch imported lazily to keep this module UI-agnostic.)"""
    import torch.nn as nn
    class C(nn.Module):
        def __init__(s):
            super().__init__()
            s.b = nn.Sequential(nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
                                nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
                                nn.Flatten(), nn.Linear(32 * 8, 64), nn.ReLU(), nn.Linear(64, n_comp))
        def forward(s, x):
            return s.b(x[:, None, :])
    return C()


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
      - "cnn"  1-D CNN over the spectrum — knobs: epochs, seed
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
    elif method == "cnn":
        import torch
        torch.manual_seed(int(seed)); net = _cnn(X.shape[1], len(subs))
        sm = torch.nn.LogSoftmax(dim=1)
        op = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-3)
        Xt = torch.tensor(X); Yt = torch.tensor(Y); w = 1.0 + 2.0 * (1.0 - Yt)   # up-weight buried
        for _ in range(int(epochs)):
            net.train(); op.zero_grad()
            (w * (sm(net(Xt)).exp() - Yt).abs()).sum(1).mean().backward(); op.step()
        net.eval()
        comp_store = {"method": "cnn",
                      "comp_state": {k: v.detach().numpy() for k, v in net.state_dict().items()}}
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
    else:                                                      # torch softmax head (MLP or CNN)
        import torch
        if method == "cnn":
            net = _cnn(model["n_feat"], len(subs))
        else:
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


def apply_recovery(model, items, progress=None):
    """Apply an already-trained composition model to each known-ratio mixture (NO training)
    → list of {name, nominal, mean[, uM_pred, uM_true]} in composition.SUBSTANCES order,
    the same shape dl_recovery returns, so the Recovery plots don't change. This is how
    Recovery 'inherits' the model trained once in the Model tab."""
    import os
    from real_data import load_map
    from composition import SUBSTANCES
    subs = model["subs"]
    out = []
    for k, it in enumerate(items):
        if progress:
            progress(f"applying model — {k + 1}/{len(items)}")
        path, ratio = it[0], it[1]
        conc = it[2] if len(it) > 2 else None
        wn, cube, _m, _c = load_map(path)
        res = apply_model(model, wn, cube); comp = res["composition"]
        s = sum(float(ratio.get(sn, 0)) for sn in subs)
        nom = np.zeros(len(SUBSTANCES)); mn = np.zeros(len(SUBSTANCES))
        for sn in subs:
            o = SUBSTANCES.index(sn)
            nom[o] = float(ratio.get(sn, 0)) / s if s > 0 else 0.0
            mn[o] = comp.get(sn, 0.0)
        row = {"name": os.path.basename(path).replace("_corrected", "").replace(".csv", ""),
               "nominal": nom, "mean": mn}
        if res.get("uM"):
            row["uM_pred"] = {sn: res["uM"][sn] for sn in subs}
            if conc:
                row["uM_true"] = {sn: conc[sn] * 1e6 for sn in subs if conc.get(sn, 0) > 0}
        out.append(row)
    return out


def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)
