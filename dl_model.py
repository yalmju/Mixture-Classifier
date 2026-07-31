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


def _fit_predict(method, Xtr, Ytr, Xte, *, pre=None, epochs=350, seed=0,
                 n_components=8, n_trees=300, P_ref=None):
    """Fit ``method`` on (Xtr, Ytr) and return composition predictions for Xte, rows
    summing to 1. Shared by the full-data fit and each leave-one-out fold so both use
    exactly the same estimator."""
    n_comp = Ytr.shape[1]
    if method == "nnls":                       # classical baseline: no training at all
        from dl_quantify import surface_composition
        p = surface_composition(np.atleast_2d(Xte), P_ref)
    elif method == "pls":
        from sklearn.cross_decomposition import PLSRegression
        nc = max(1, min(int(n_components), len(Xtr) - 1, Xtr.shape[1]))
        p = PLSRegression(n_components=nc).fit(Xtr, Ytr).predict(np.atleast_2d(Xte))
    elif method == "rf":
        from sklearn.ensemble import RandomForestRegressor
        p = RandomForestRegressor(n_estimators=int(n_trees),
                                  random_state=int(seed)).fit(Xtr, Ytr).predict(np.atleast_2d(Xte))
    elif method == "cnn":
        import torch
        torch.manual_seed(int(seed)); net = _cnn(Xtr.shape[1], n_comp)
        sm = torch.nn.LogSoftmax(dim=1)
        op = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-3)
        Xt = torch.tensor(Xtr); Yt = torch.tensor(Ytr); w = 1.0 + 2.0 * (1.0 - Yt)
        for _ in range(int(epochs)):
            net.train(); op.zero_grad()
            (w * (sm(net(Xt)).exp() - Yt).abs()).sum(1).mean().backward(); op.step()
        net.eval()
        with torch.no_grad():
            return torch.softmax(net(torch.tensor(np.atleast_2d(Xte).astype(np.float32))), 1).numpy()
    else:
        from dl_quantify import train_composition, predict_composition
        m = train_composition(Xtr, Ytr, n_comp, pretrain=pre, seed=seed, epochs_ft=epochs)
        return predict_composition(m, np.atleast_2d(Xte))
    p = np.clip(np.asarray(p, float), 0, None)
    return p / (p.sum(1, keepdims=True) + 1e-12)


def _map_spectra(cube, mask, n_px=0):
    """Spectra to train on for ONE map. ``n_px``=0 keeps the historic behaviour (a single
    intensity-weighted mean). ``n_px``>0 also returns that many individual pixels — the
    brightest ones, which carry the SERS signal rather than blank substrate — each
    baseline-corrected. The map's known ratio labels all of them, so a 400-pixel map
    contributes hundreds of training examples instead of one; splits must stay GROUPED BY
    MAP or pixels of the same map would leak across the split."""
    from sers_mixture import als_baseline
    cube = np.asarray(cube, float)[:, mask]
    w = cube.sum(1); wn_ = w / (w.sum() + 1e-12)
    mean = wn_ @ cube
    out = [np.clip(mean - als_baseline(mean), 0, None)]
    if n_px and len(cube) > 1:
        order = np.argsort(-w)                       # brightest first
        for i in order[:int(n_px)]:
            y = cube[i]
            out.append(np.clip(y - als_baseline(y), 0, None))
    return out


def _mean_spectrum(cube, mask):
    cube = np.asarray(cube, float)[:, mask]
    w = cube.sum(1); w = w / (w.sum() + 1e-12)
    from sers_mixture import als_baseline
    mean = w @ cube
    return np.clip(mean - als_baseline(mean), 0, None)


def train_model(data_dir, items, calib_path=None, baseline=True, trim=None, progress=None,
                method="mlp", epochs=350, seed=0, use_pretrain=True,
                n_components=8, n_trees=300, loo=False, test_items=None, px_per_map=0):
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
    X, Xabs, Y, Cabs, paths = [], [], [], [], []
    for k, it in enumerate(items):
        if progress and (k % 3 == 0 or k == len(items) - 1):   # breathe during map loading
            progress(f"loading maps {k + 1}/{len(items)}")
        ratio = it[1]; conc = it[2] if len(it) > 2 else None
        vec = _ratio([float(ratio.get(s, 0.0)) for s in subs])
        if vec.sum() <= 0:
            continue
        _w, cube, _m, _c = load_map(it[0])
        for j, ya in enumerate(_map_spectra(cube, mask, px_per_map)):
            X.append(ya / (np.linalg.norm(ya) + 1e-12)); Y.append(vec)
            paths.append(it[0])                       # group key: the MAP, not the row
            if j == 0:                                # µM head stays on the map mean
                Xabs.append(ya)
                Cabs.append([float(conc.get(s, 0.0)) for s in subs] if conc else None)
    X = np.array(X, np.float32); Xabs = np.array(Xabs); Y = np.array(Y, np.float32)
    paths = np.array(paths, object)
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
    def _norm_rows(p):
        p = np.clip(np.asarray(p, float), 0, None); return p / (p.sum(1, keepdims=True) + 1e-12)
    loss_curve = []
    if method == "nnls":                       # classical baseline: nothing is trained
        from dl_quantify import surface_composition
        comp_store = {"method": "nnls"}
        tp = _norm_rows(surface_composition(X, P))
    elif method == "pls":
        from sklearn.cross_decomposition import PLSRegression
        nc = max(1, min(int(n_components), len(X) - 1, X.shape[1]))
        sk = PLSRegression(n_components=nc).fit(X, Y)
        comp_store = {"method": "pls", "sk": sk}; tp = _norm_rows(sk.predict(X))
    elif method == "rf":
        from sklearn.ensemble import RandomForestRegressor
        sk = RandomForestRegressor(n_estimators=int(n_trees), random_state=int(seed)).fit(X, Y)
        comp_store = {"method": "rf", "sk": sk}; tp = _norm_rows(sk.predict(X))
    elif method == "cnn":
        import torch
        torch.manual_seed(int(seed)); net = _cnn(X.shape[1], len(subs))
        sm = torch.nn.LogSoftmax(dim=1)
        op = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-3)
        Xt = torch.tensor(X); Yt = torch.tensor(Y); w = 1.0 + 2.0 * (1.0 - Yt)   # up-weight buried
        for ep in range(int(epochs)):
            net.train(); op.zero_grad()
            l = (w * (sm(net(Xt)).exp() - Yt).abs()).sum(1).mean(); l.backward(); op.step()
            loss_curve.append(float(l.detach()))
            if progress and (ep % 10 == 0 or ep == int(epochs) - 1):
                progress(f"epoch {ep + 1}/{int(epochs)}  loss {loss_curve[-1]:.3f}")
        net.eval()
        with torch.no_grad():
            tp = torch.softmax(net(Xt), 1).numpy()
        comp_store = {"method": "cnn",
                      "comp_state": {k: v.detach().numpy() for k, v in net.state_dict().items()}}
    else:
        method = "mlp"
        comp = train_composition(X, Y, len(subs), pretrain=pre, seed=seed, epochs_ft=epochs,
                                 progress=progress)
        comp_store = {"method": "mlp",
                      "comp_state": {k: v.cpu().numpy() for k, v in comp["state"].items()},
                      "comp_hidden": (256, 64)}
        from dl_quantify import predict_composition
        tp = predict_composition(comp, X); loss_curve = comp.get("hist", [])
    train_eval = {"true": np.asarray(Y, float).tolist(), "pred": np.asarray(tp, float).tolist(),
                  "loss": loss_curve}

    # An INDEPENDENT batch (Role = test in Samples) beats leave-one-out: those maps were
    # measured separately and never touched training, so scoring them is the real check.
    test_eval = None
    if test_items:
        Xt_, Yt_, tp_paths = [], [], []
        for k, it in enumerate(test_items):
            if progress:
                progress(f"scoring held-out batch {k + 1}/{len(test_items)}")
            vec = _ratio([float(it[1].get(s_, 0.0)) for s_ in subs])
            if vec.sum() <= 0:
                continue
            _w, cube, _m, _c = load_map(it[0])
            ya = _mean_spectrum(cube, mask)
            Xt_.append(ya / (np.linalg.norm(ya) + 1e-12)); Yt_.append(vec); tp_paths.append(it[0])
        if Xt_:
            Pt = _fit_predict(method, X, Y, np.array(Xt_, np.float32), pre=pre, epochs=epochs,
                              seed=seed, n_components=n_components, n_trees=n_trees, P_ref=P)
            test_eval = {"true": np.asarray(Yt_, float).tolist(),
                         "pred": np.asarray(Pt, float).tolist(), "paths": tp_paths}

    loo_eval = None
    if loo and test_eval is None and len(X) >= 3:                     # honest metrics: predict each held-out mixture
        uniq = list(dict.fromkeys(paths.tolist()))     # leave one MAP out, not one row
        tv, pv, pl = [], [], []
        for i, mp in enumerate(uniq):
            if progress:
                progress(f"leave-one-map-out {i + 1}/{len(uniq)}")
            te = np.where(paths == mp)[0]; tr = np.where(paths != mp)[0]
            pred = _fit_predict(method, X[tr], Y[tr], X[te], pre=pre, epochs=epochs,
                                seed=seed + i, n_components=n_components,
                                n_trees=n_trees, P_ref=P)
            tv.append(Y[te[0]].tolist()); pv.append(np.asarray(pred, float).mean(0).tolist())
            pl.append(mp)
        loo_eval = {"true": tv, "pred": pv, "paths": pl}

    uM = None
    # Cabs/Xabs are per MAP (one row each); X may hold many pixel rows per map
    have = [i for i in range(len(Xabs)) if Cabs[i] is not None and any(c > 0 for c in Cabs[i])]
    if len(have) >= 3:
        if progress:
            progress("training concentration head")
        hv = np.array(have); C = np.array([Cabs[i] if Cabs[i] is not None else [0.0] * len(subs)
                                           for i in range(len(Xabs))], float)
        mu = Xabs[hv].mean(0); sd = Xabs[hv].std(0) + 1e-8
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(Xabs.shape[1], 256), nn.BatchNorm1d(256), nn.ReLU(),
                            nn.Dropout(0.15), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, len(subs)))
        op = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-3)
        Xt = torch.tensor(((Xabs[hv] - mu) / sd).astype(np.float32))
        Yt = torch.tensor((np.log10(np.clip(C[hv], 1e-8, None)) + 6.0).astype(np.float32))
        for ep in range(epochs):
            net.train(); op.zero_grad(); ((net(Xt) - Yt) ** 2).mean().backward(); op.step()
            if progress and ep % 15 == 0:
                progress(f"concentration head — epoch {ep + 1}/{epochs}")
        net.eval()
        uM = {"state": {k: v.detach().numpy() for k, v in net.state_dict().items()},
              "mu": mu, "sd": sd}

    return {"subs": subs, "lo": lo, "hi": hi, "n_feat": int(mask.sum()), "P": P,
            "uM": uM, "n_train": int(len(X)), "has_uM": uM is not None,
            "train_eval": train_eval, "loo_eval": loo_eval, "test_eval": test_eval,
            **comp_store}


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


def kfold_stability(data_dir, items, method="mlp", folds=5, progress=None, **kw):
    """Repeat the held-out check over every fold of an even 1-in-``folds`` split (start
    offsets 0..folds-1) so a single lucky/unlucky test set cannot set the headline number.
    Returns {"errors": [...per fold], "mean", "sd"} of the composition error."""
    order = sorted(range(len(items)), key=lambda i: os.path.basename(items[i][0]))
    errs = []
    for f in range(int(folds)):
        te = [items[i] for pos, i in enumerate(order) if pos % folds == f]
        tr = [items[i] for pos, i in enumerate(order) if pos % folds != f]
        if len(tr) < 3 or not te:
            continue
        if progress:
            progress(f"fold {f + 1}/{folds} — {len(tr)} train / {len(te)} test")
        m = train_model(data_dir, tr, method=method, test_items=te, progress=None, **kw)
        ev = m.get("test_eval")
        if not ev:
            continue
        T = np.asarray(ev["true"], float); Pd = np.asarray(ev["pred"], float)
        errs.append(float((0.5 * np.abs(Pd - T).sum(1)).mean()))
    return {"errors": errs,
            "mean": float(np.mean(errs)) if errs else float("nan"),
            "sd": float(np.std(errs)) if errs else float("nan")}


def benchmark_loo(data_dir, items, calib_path=None, baseline=True, trim=None, progress=None,
                  methods=("nnls", "pls", "rf", "cnn", "mlp"), epochs=350, seed=0,
                  use_pretrain=True,
                  n_components=8, n_trees=300):
    """Leave-one-out comparison of the composition methods on the SAME mixtures — the
    honest counterpart to the train-set numbers. Maps are loaded once, then every method
    is refit per fold. Returns {method: {"true", "pred"}} plus "subs"."""
    from real_data import load_map
    from dl_quantify import simulate_mixtures, _ratio
    subs, wn, mask, P, lo, hi = _refs(data_dir, baseline, trim)
    X, Y = [], []
    for k, it in enumerate(items):
        if progress:
            progress(f"loading maps {k + 1}/{len(items)}")
        vec = _ratio([float(it[1].get(s, 0.0)) for s in subs])
        if vec.sum() <= 0:
            continue
        _w, cube, _m, _c = load_map(it[0])
        ya = _mean_spectrum(cube, mask)
        X.append(ya / (np.linalg.norm(ya) + 1e-12)); Y.append(vec)
    X = np.array(X, np.float32); Y = np.array(Y, np.float32)
    if len(X) < 3:
        raise ValueError("need ≥3 mixtures for a leave-one-out benchmark.")

    pre = None
    if calib_path and use_pretrain:
        try:
            from io_utils import load_calibration_csv
            from calibration import calibrate
            ax_c, nc, dils = load_calibration_csv(calib_path); mc = (ax_c >= lo) & (ax_c <= hi)
            dil = [(dils[nc.index(s)][0], np.asarray(dils[nc.index(s)][1])[:, mc])
                   for s in subs if s in nc]
            if len(dil) == len(subs):
                cal = calibrate(dil, P, subs); rng = np.random.default_rng(0)
                Xs, Cs = simulate_mixtures(P, cal.K, cal.gA, 5000, rng, noise=0.015,
                                           baseline=0.03, gain_lo=0.8, gain_hi=1.25)
                Xs = np.array([r / (np.linalg.norm(r) + 1e-12) for r in Xs]).astype(np.float32)
                pre = (Xs, np.array([_ratio(c) for c in Cs]).astype(np.float32))
        except Exception:
            pre = None

    out = {"subs": subs}
    for mi, meth in enumerate(methods):
        Pm = np.zeros_like(Y, float)
        for i in range(len(X)):
            if progress:
                progress(f"{meth.upper()} leave-one-out {i + 1}/{len(X)}  "
                         f"[{mi + 1}/{len(methods)} methods]")
            tr = [j for j in range(len(X)) if j != i]
            Pm[i] = _fit_predict(meth, X[tr], Y[tr], X[i], pre=pre, epochs=epochs,
                                 seed=seed + i, n_components=n_components,
                                 n_trees=n_trees, P_ref=P)[0]
        out[meth] = {"true": Y.tolist(), "pred": Pm.tolist()}
    return out


def apply_model_pixels(model, wn, spectra):
    """Composition for EVERY pixel spectrum (n_px, n_wn) → (n_px, n_subs), rows summing
    to 1. Same heads as apply_model, just batched, so the Real-data tab can draw a
    per-pixel composition map from the trained model instead of only a map-level answer."""
    lo, hi = model["lo"], model["hi"]
    wn = np.asarray(wn); mask = (wn >= lo) & (wn <= hi)
    X = np.asarray(spectra, float)
    if X.shape[1] == len(wn):
        X = X[:, mask]
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    method = model.get("method", "mlp")
    if method in ("pls", "rf"):
        p = np.clip(np.asarray(model["sk"].predict(X.astype(np.float64)), float), 0, None)
        return p / (p.sum(1, keepdims=True) + 1e-12)
    import torch
    if method == "cnn":
        net = _cnn(model["n_feat"], len(model["subs"]))
    else:
        from dl_quantify import _spec_net
        net = _spec_net(model["n_feat"], len(model["subs"]), model["comp_hidden"])
    net.load_state_dict({k: torch.tensor(v) for k, v in model["comp_state"].items()}); net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 512):                 # batch so a big map stays in memory
            out.append(torch.softmax(net(torch.tensor(X[i:i + 512].astype(np.float32))), 1).numpy())
    return np.vstack(out)


def apply_recovery(model, items, progress=None):
    """Apply an already-trained composition model to each known-ratio mixture (NO training)
    → list of {name, nominal, mean[, uM_pred, uM_true]} in composition.SUBSTANCES order,
    the same shape dl_recovery returns, so the Recovery plots don't change. This is how
    Recovery 'inherits' the model trained once in the Model tab."""
    import os
    from real_data import load_map
    from composition import SUBSTANCES
    subs = model["subs"]
    # A mixture the model TRAINED on must not be scored by that model — it would just
    # recite its own answer (recovery collapses to ~100%). Training already computed a
    # held-out prediction for each of those maps, so reuse it and only run the model on
    # mixtures it has genuinely never seen.
    lo_ = model.get("loo_eval") or {}
    held = {}
    if lo_.get("paths"):
        for pth, pred in zip(lo_["paths"], lo_["pred"]):
            held[os.path.normcase(os.path.normpath(pth))] = pred
    out = []
    for k, it in enumerate(items):
        if progress:
            progress(f"applying model — {k + 1}/{len(items)}")
        path, ratio = it[0], it[1]
        conc = it[2] if len(it) > 2 else None
        key = os.path.normcase(os.path.normpath(path))
        if key in held:                                  # held-out prediction from training
            comp = {subs[j]: float(held[key][j]) for j in range(len(subs))}
            res = {"uM": None}
        else:
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
