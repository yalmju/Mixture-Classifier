"""Pixel-level surface-composition model for SERS maps.

The model is deliberately separate from the map-level Recovery model. It learns on
synthetic pixels made from measured pure-reference pixels, where the surface spectral
contribution is known exactly. Known-mixture nominal ratios are never copied onto every
pixel.
"""
from __future__ import annotations

import os
import pickle

import numpy as np

MODEL_KIND = "pixel_surface_v2"
MODEL_VERSION = 2


def _net(n_feat, n_out, hidden=(256, 64)):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(n_feat, hidden[0]), nn.BatchNorm1d(hidden[0]), nn.ReLU(),
                         nn.Dropout(0.12), nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
                         nn.Linear(hidden[1], n_out))


def _trim(wn, lo, hi):
    wn = np.asarray(wn, float)
    return (wn >= float(lo)) & (wn <= float(hi))


def _l2(x):
    x = np.asarray(x, float)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _preprocess_pixels(cube, mask, baseline=True):
    from unmix import _baseline_removed
    x = np.asarray(cube, float)
    if x.shape[1] != int(mask.sum()):
        x = x[:, mask]
    return _l2(_baseline_removed(x, baseline)).astype(np.float32)


def _reference_pool(data_dir, subs, lo, hi, per_map=120, baseline=True, progress=None):
    from dataset import discover_dataset, is_blank
    from real_data import load_map
    grouped = {s: [] for s in subs}
    axis = None
    for name, paths in discover_dataset(data_dir):
        if name not in grouped or is_blank(name):
            continue
        for _batch, path, role in paths:
            wn, cube, _m, _coord = load_map(path)
            axis = np.asarray(wn, float) if axis is None else axis
            mask = _trim(wn, lo, hi)
            cube = np.asarray(cube, float)
            order = np.argsort(-cube[:, mask].sum(1))[:int(per_map)]
            if progress:
                progress(f"reference pixels: {name} / {os.path.basename(path)}")
            grouped[name].append(_preprocess_pixels(cube[order], mask, baseline))
    missing = [s for s in subs if not grouped[s]]
    if missing:
        raise ValueError(f"missing pure reference pixels for {missing}")
    return axis, {s: np.vstack(grouped[s]) for s in subs}


def _synthetic(pool, subs, n, rng, alpha=0.45, noise=0.018):
    """Mix unit pure spectra; labels are their surface spectral contributions."""
    k = len(subs); nf = pool[subs[0]].shape[1]
    frac = rng.dirichlet(np.full(k, float(alpha)), size=int(n)).astype(np.float32)
    # Include exact pure anchors and balanced mixtures in every training set.
    anchors = np.vstack([np.eye(k, dtype=np.float32),
                         np.full((max(8, k), k), 1.0 / k, dtype=np.float32)])
    frac[:len(anchors)] = anchors[:min(len(anchors), len(frac))]
    x = np.zeros((len(frac), nf), np.float32)
    for j, s in enumerate(subs):
        p = pool[s]
        pick = p[rng.integers(0, len(p), size=len(frac))]
        x += frac[:, j:j + 1] * pick
    gain = rng.lognormal(mean=0.0, sigma=0.28, size=(len(x), 1)).astype(np.float32)
    x *= gain
    x += rng.normal(0.0, float(noise), size=x.shape).astype(np.float32)
    x = np.clip(x, 0, None)
    return _l2(x).astype(np.float32), frac


def train_pixel_surface(data_dir, *, subs=None, lo=400.0, hi=1800.0, per_map=120,
                        n_synth=12000, epochs=180, seed=0, baseline=True, progress=None):
    """Train a per-pixel surface-contribution MLP from measured pure pixels."""
    import torch
    from dataset import discover_dataset, is_blank
    if subs is None:
        subs = [n for n, _ in discover_dataset(data_dir) if not is_blank(n)]
    subs = list(subs)
    axis, pool = _reference_pool(data_dir, subs, lo, hi, per_map, baseline, progress)
    rng = np.random.default_rng(int(seed))
    x, y = _synthetic(pool, subs, int(n_synth), rng)
    order = rng.permutation(len(x)); cut = max(1, int(len(x) * 0.85))
    tr, va = order[:cut], order[cut:]
    torch.manual_seed(int(seed)); net = _net(x.shape[1], len(subs))
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-3)
    xt, yt = torch.tensor(x[tr]), torch.tensor(y[tr])
    xv, yv = torch.tensor(x[va]), torch.tensor(y[va])
    history = []
    for ep in range(int(epochs)):
        net.train(); opt.zero_grad()
        pred = torch.softmax(net(xt), 1)
        # Minor components matter, but do not let near-zero labels dominate the loss.
        w = 1.0 + 1.5 * (1.0 - yt)
        loss = (w * (pred - yt).abs()).sum(1).mean()
        loss.backward(); opt.step()
        if ep % 10 == 0 or ep == int(epochs) - 1:
            net.eval()
            with torch.no_grad():
                pv = torch.softmax(net(xv), 1)
                tv = 0.5 * (pv - yv).abs().sum(1).mean().item()
            history.append((ep + 1, float(loss.detach()), float(tv)))
            if progress:
                progress(f"pixel surface epoch {ep + 1}/{epochs} - synthetic TV {tv:.3f}")
    return {"kind": MODEL_KIND, "version": MODEL_VERSION, "subs": subs,
            "lo": float(lo), "hi": float(hi), "n_feat": int(x.shape[1]),
            "hidden": (256, 64), "state": {k: v.detach().cpu().numpy()
                                            for k, v in net.state_dict().items()},
            "baseline": bool(baseline), "training_level": "synthetic_measured_pure_pixels",
            "px_per_map": int(per_map), "n_train": int(len(tr)),
            "n_maps": int(sum(len(v) // max(1, int(per_map)) for v in pool.values())),
            "synthetic_val_tv": float(history[-1][2]), "history": history}


def apply_pixel_surface(model, wn, spectra, *, preprocessed=False, batch_size=512):
    import torch
    x = np.asarray(spectra, float)
    mask = _trim(wn, model["lo"], model["hi"])
    if preprocessed:
        if x.shape[1] == len(mask):
            x = x[:, mask]
        x = _l2(np.clip(x, 0, None)).astype(np.float32)
    else:
        x = _preprocess_pixels(x, mask, model.get("baseline", True))
    if x.shape[1] != int(model["n_feat"]):
        raise ValueError(f"Pixel surface model expects {model['n_feat']} features in "
                         f"{model['lo']:.0f}-{model['hi']:.0f} cm-1, got {x.shape[1]}; "
                         "retrain it with the Samples preprocessing window.")
    net = _net(model["n_feat"], len(model["subs"]), tuple(model.get("hidden", (256, 64))))
    net.load_state_dict({k: torch.tensor(v) for k, v in model["state"].items()}); net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), int(batch_size)):
            out.append(torch.softmax(net(torch.tensor(x[i:i + batch_size])), 1).numpy())
    return np.vstack(out)


def train_concentration_head(model, calibration_path, pure_data_dir, *, n_train=20000,
                             n_val=4000, epochs=70, seed=0, progress=None):
    """Add a physics-informed per-pixel log10-concentration head to v2."""
    from calibration import calibrate
    from dl_model import _refs
    from dl_quantify import simulate_mixtures, train_quantifier, predict_concentration
    from io_utils import load_calibration_csv
    from unmix import _baseline_removed

    subs = list(model["subs"]); baseline = bool(model.get("baseline", False))
    lo, hi = float(model["lo"]), float(model["hi"])
    rsubs, rwn, rmask, P, _lo, _hi = _refs(pure_data_dir, baseline, (lo, hi))
    target_wn = np.asarray(rwn, float)[rmask]
    P = np.asarray(P, float)[[rsubs.index(s) for s in subs]]
    cwn, cnames, dils = load_calibration_csv(calibration_path)
    cidx = {n: i for i, n in enumerate(cnames)}
    missing = [s for s in subs if s not in cidx]
    if missing:
        raise ValueError(f"calibration is missing {missing}")
    cmask = _trim(cwn, lo, hi); aligned = []; ranges = []
    for s in subs:
        C, Y = dils[cidx[s]]; C = np.asarray(C, float)
        Y = np.asarray(Y, float)[:, cmask]
        if Y.shape[1] != len(target_wn) or not np.allclose(cwn[cmask], target_wn):
            Y = np.vstack([np.interp(target_wn, cwn[cmask], row) for row in Y])
        aligned.append((C, _baseline_removed(Y, baseline)))
        ranges.append((float(C.min()), float(C.max())))
    calib = calibrate(aligned, P, subs)
    sim = {"c_lo": min(v[0] for v in ranges), "c_hi": max(v[1] for v in ranges),
           "p_present": 0.78, "gain_lo": 0.35, "gain_hi": 2.8,
           "noise": 0.02, "baseline": 0.0, "min_present": 1}
    q, hist = train_quantifier(P, calib.K, calib.gA, subs, n_train=int(n_train),
                               n_val=int(n_val), epochs=int(epochs), seed=int(seed),
                               sim=sim, progress=progress)
    Xv, Cv = simulate_mixtures(P, calib.K, calib.gA, int(n_val),
                               np.random.default_rng(int(seed) + 991), **sim)
    score = np.sqrt(np.mean(((Xv - q.x_mean) / q.x_std) ** 2, axis=1))
    Pv = predict_concentration(q, Xv); present = Cv > 0
    log_mae = float(np.mean(np.abs(np.log10(np.clip(Pv[present], 1e-12, None)) -
                                    np.log10(Cv[present]))))
    model["uM"] = {"kind": "pixel_concentration_v2", "subs": subs,
        "state": {k: v.detach().cpu().numpy() for k, v in q.state.items()},
        "mu": np.asarray(q.x_mean, np.float32), "sd": np.asarray(q.x_std, np.float32),
        "hidden": tuple(q.hidden), "ranges_M": np.asarray(ranges, float),
        "ood_rms_threshold": float(np.quantile(score, 0.995)),
        "calibration_path": str(calibration_path), "K": calib.K, "gA": calib.gA,
        "training": {"n_train": int(n_train), "n_val": int(n_val),
            "epochs": int(epochs), "seed": int(seed), "sim": sim,
            "val_present_log_mae": log_mae,
            "scope": "order-of-magnitude, semi-quantitative"}}
    model["version"] = max(3, int(model.get("version", 2)))
    return model


def apply_pixel_concentration(model, wn, spectra, *, batch_size=512):
    """Return (uM, names, meta); invalid estimates are NaN and marked OOD."""
    import torch
    from dl_quantify import _build_net
    from unmix import _baseline_removed
    u = model.get("uM")
    if not u or u.get("kind") != "pixel_concentration_v2":
        return None, [], {}
    x = np.asarray(spectra, float); mask = _trim(wn, model["lo"], model["hi"])
    if x.shape[1] == len(mask): x = x[:, mask]
    x = _baseline_removed(x, bool(model.get("baseline", False))).astype(np.float32)
    z = ((x - u["mu"]) / u["sd"]).astype(np.float32)
    score = np.sqrt(np.mean(z * z, axis=1))
    spectral_ood = score > float(u["ood_rms_threshold"])
    net = _build_net(model["n_feat"], len(u["subs"]), tuple(u.get("hidden", (256, 128))))
    net.load_state_dict({k: torch.tensor(v) for k, v in u["state"].items()}); net.eval()
    pred = []
    with torch.no_grad():
        for i in range(0, len(z), int(batch_size)):
            pred.append(net(torch.tensor(z[i:i + batch_size])).numpy())
    M = 10.0 ** np.vstack(pred); ranges = np.asarray(u["ranges_M"], float)
    invalid = (M < ranges[:, 0]) | (M > ranges[:, 1]) | spectral_ood[:, None]
    out = M * 1e6; out[invalid] = np.nan
    return out, list(u["subs"]), {"spectral_ood": spectral_ood,
        "component_ood": invalid, "ranges_M": ranges, "score": score}
def save_pixel_surface(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_pixel_surface(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    if model.get("kind") != MODEL_KIND:
        raise ValueError("not a Pixel surface v2 model")
    return model