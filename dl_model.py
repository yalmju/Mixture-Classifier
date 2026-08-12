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
    names, wn, means = _templates(data_dir, baseline, None)
    lo, hi = trim if trim else (float(np.min(wn)), float(np.max(wn)))
    mask = (wn >= lo) & (wn <= hi)
    if mask.sum() < 10:
        mask = np.ones(len(wn), bool)
    idx = {n: i for i, n in enumerate(names)}
    subs = [s for s in names if not is_blank(s)]
    P = _l2(_baseline_removed(means[[idx[s] for s in subs]][:, mask], baseline))
    return subs, wn, mask, P, lo, hi


def _nuisance_refs(data_dir, baseline, mask):
    """Unit templates for what sits on the substrate but is NOT an analyte — INK, BLK.

    `dataset.BLANK_ALIASES` counts "ink" as blank, so these never enter the unmixing
    basis. The printed ink is real and large, though: its NNLS coefficient runs 0.06–1.5×
    the analyte total (median 0.40) and its template has cosine 0.73 with DQ — closer than
    DQ is to TBZ. Handing them to the simulator as nuisance lets the physics pre-training
    see ink-shaped intensity that is NOT DQ. Returns (n_nu, n_feat) or None."""
    from unmix import _templates, _baseline_removed, _l2
    from dataset import is_blank
    try:
        names, wn, means = _templates(data_dir, baseline, None)
    except Exception:
        return None
    idx = [i for i, n in enumerate(names) if is_blank(n)]
    if not idx:
        return None
    return _l2(_baseline_removed(means[idx][:, mask], baseline))


def _composition_features(x, mode="log1p_raw"):
    """Composition features that preserve between-pixel intensity; no row-wise L2."""
    x = np.clip(np.asarray(x, float), 0, None)
    if mode == "legacy_l2":
        if x.ndim == 1:
            return x / (np.linalg.norm(x) + 1e-12)
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    return np.log1p(x)


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
                 n_components=8, n_trees=300, P_ref=None, rf_max_features=None):
    """Fit ``method`` on (Xtr, Ytr) and return composition predictions for Xte, rows
    summing to 1. Shared by the full-data fit and each leave-one-out fold so both use
    exactly the same estimator."""
    n_comp = Ytr.shape[1]
    if method == "nnls":                       # classical baseline: no training at all
        from dl_quantify import surface_composition
        raw = np.expm1(np.clip(np.atleast_2d(Xte), 0, None))
        p = surface_composition(_composition_features(raw, "legacy_l2"), P_ref)
    elif method == "pls":
        from sklearn.cross_decomposition import PLSRegression
        nc = max(1, min(int(n_components), len(Xtr) - 1, Xtr.shape[1]))
        p = PLSRegression(n_components=nc).fit(Xtr, Ytr).predict(np.atleast_2d(Xte))
    elif method == "rf":
        # max_features: sklearn's REGRESSOR default is 1.0 — every split reads all ~2000
        # spectral channels, which is minutes per fit here. 'sqrt' is the usual forest
        # recipe and is what the benchmark passes; None keeps the library default so an
        # already-trained model scores the same as before.
        from sklearn.ensemble import RandomForestRegressor
        kw = {} if rf_max_features is None else {"max_features": rf_max_features}
        p = RandomForestRegressor(n_estimators=int(n_trees), random_state=int(seed),
                                  **kw).fit(Xtr, Ytr).predict(np.atleast_2d(Xte))
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


def _map_spectra(cube, mask, n_px=0, spread=False, baseline_correct=True):
    """Return representative spectra from one map.

    ``n_px=0`` returns one intensity-weighted mean spectrum. ``n_px>0`` returns
    individual pixels, brightest first unless ``spread`` is requested. Pixels from the
    same map must remain in the same validation split because they share one map label.
    """
    from sers_mixture import als_baseline
    cube = np.asarray(cube, float)[:, mask]
    w = cube.sum(1); wn_ = w / (w.sum() + 1e-12)
    mean = wn_ @ cube
    out = ([] if n_px else
           [np.clip(mean - als_baseline(mean), 0, None) if baseline_correct
            else np.clip(mean, 0, None)])
    if n_px and len(cube) > 1:
        order = np.argsort(-w)                       # brightest first
        if spread:                                   # background: sample the WHOLE intensity
            k = max(1, len(order) // int(n_px))       # range, or the model only ever sees
            order = order[::k][:int(n_px)]            # bright substrate and never dim pixels
        for i in order[:int(n_px)]:
            y = cube[i]
            out.append(np.clip(y - als_baseline(y), 0, None) if baseline_correct
                       else np.clip(y, 0, None))
    return out


def _mean_spectrum(cube, mask, baseline_correct=True):
    cube = np.asarray(cube, float)[:, mask]
    w = cube.sum(1); w = w / (w.sum() + 1e-12)
    from sers_mixture import als_baseline
    mean = w @ cube
    return (np.clip(mean - als_baseline(mean), 0, None) if baseline_correct
            else np.clip(mean, 0, None))


def nnls_hit_spectra(data_dir, path, baseline=True, trim=None, min_frac=0.15,
                     hit_mode="threshold", progress=None):
    """Return only pixels accepted by the same NNLS gate used by Real data.

    NNLS decides where SERS ink produced a substance signal; the learned model
    subsequently estimates composition/concentration and must not redefine this mask.
    """
    from unmix import unmix_map
    r = unmix_map(data_dir, path, method="nnls", baseline=baseline, trim=trim,
                  min_frac=float(min_frac), hit_mode=hit_mode, progress=progress)
    keep = np.asarray(r.hit, bool)
    if not keep.any():
        raise ValueError(f"NNLS found no hit pixels in {os.path.basename(path)}")
    return r.wn, np.asarray(r.spectra, float)[keep], {
        "hit_fraction": float(keep.mean()), "hit_count": int(keep.sum()),
        "n_pixels": int(len(keep)), "hit_rule": r.hit_rule,
    }


def _noisy_copies(y, k, rng, lo=0.25, hi=1.0):
    """``k`` low-SNR versions of one training spectrum, same label.

    Mixture pixels are sampled brightest-first (a dim pixel in a mixture map may be bare
    substrate, so labelling it with the map's ratio would be a lie), while blank pixels
    are sampled across the whole intensity range. The model therefore only ever sees
    BRIGHT substance and BOTH bright and dim blank — it learns "dim means background" and
    calls a faint but real substance pixel blank. Scaling a genuine substance pixel down
    and adding noise gives a dim substance example whose label is still true."""
    out = []
    ref = float(np.std(y)) or 1.0
    for _ in range(int(k)):
        g = rng.uniform(lo, hi)                       # weaker signal…
        out.append(np.clip(g * y + rng.normal(0, (1.0 - g) * ref, y.shape), 0, None))
    return out


def _concentration_context_features(X, groups=None, ratios=None):
    """Low-dimensional competitive-adsorption features for each hit pixel.

    The concentration inverse sees the local NNLS surface composition and signal size,
    plus P10/median/P90 summaries of those quantities over the same experimental map.
    It therefore learns the applied-concentration relation without memorising thousands
    of wavelength channels from only a few dozen maps.
    """
    X = np.clip(np.asarray(X, float), 0, None)
    if ratios is None:
        raise ValueError("NNLS composition ratios are required for concentration context")
    R = np.asarray(ratios, float)
    if groups is None:
        groups = np.zeros(len(X), int)
    groups = np.asarray(groups, object)
    log_total = np.log1p(X.sum(axis=1, keepdims=True))
    # local ratio + local log-intensity + map ratio quantiles + map intensity quantiles
    out = np.empty((len(X), R.shape[1] + 1 + R.shape[1] * 3 + 3), float)
    for g in dict.fromkeys(groups.tolist()):
        idx = np.where(groups == g)[0]
        rq = np.percentile(R[idx], [10, 50, 90], axis=0).reshape(-1)
        iq = np.percentile(log_total[idx, 0], [10, 50, 90])
        ctx = np.concatenate([rq, iq])
        out[idx] = np.concatenate([R[idx], log_total[idx],
                                   np.repeat(ctx[None, :], len(idx), axis=0)], axis=1)
    return out


def _group_validation_indices(groups, fraction=0.2, seed=0):
    """Deterministic row indices with entire maps assigned to validation."""
    groups = np.asarray(groups, object)
    unique = np.array(list(dict.fromkeys(groups.tolist())), object)
    if len(unique) < 5:
        return np.arange(len(groups)), np.array([], int)
    rng = np.random.default_rng(int(seed)); order = rng.permutation(len(unique))
    n_val = max(1, int(round(len(unique) * float(fraction))))
    val_maps = set(unique[order[:n_val]].tolist())
    is_val = np.array([g in val_maps for g in groups], bool)
    return np.where(~is_val)[0], np.where(is_val)[0]


def train_model(data_dir, items, calib_path=None, baseline=True, trim=None, progress=None,
                method="mlp", epochs=300, seed=0, use_pretrain=True, epoch_diagnostics=False,
                n_components=8, n_trees=300, loo=False, test_items=None, px_per_map=0,
                include_blank=False, noise_aug=0, nnls_screen=True,
                screen_min_frac=0.15, equal_volume_mix=False, sim_nuisance=True,
                sim_iso=None):
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
    aug_rng = np.random.default_rng(int(seed))
    X, Xabs, Y, Cabs, paths, abs_paths = [], [], [], [], [], []
    # Splitting groups by COMPOSITION, not by map. The same mixture measured twice is two
    # maps but one condition; grouping by map lets a repeat of the held-out condition stay
    # in training, and the held-out number comes back flattered. `paths` stays the map (it
    # is what pooling, the map count and the per-path annotation need) and `conds` is the
    # parallel key used wherever a train/test boundary is drawn.
    conds, abs_conds = [], []
    screen_stats = []
    for k, it in enumerate(items):
        if progress and (k % 3 == 0 or k == len(items) - 1):   # breathe during map loading
            progress(f"loading maps {k + 1}/{len(items)}")
        ratio = it[1]; conc = it[2] if len(it) > 2 else None
        vec = _ratio([float(ratio.get(s, 0.0)) for s in subs])
        if vec.sum() <= 0:
            continue
        if nnls_screen:
            _w, cube, smeta = nnls_hit_spectra(
                data_dir, it[0], baseline=baseline, trim=trim,
                min_frac=screen_min_frac, progress=None)
            local_mask = np.ones(cube.shape[1], bool)
            screen_stats.append({"path": it[0], **smeta})
        else:
            _w, cube, _m, _c = load_map(it[0]); local_mask = mask
        # The CONDITION key. A string, not a tuple: these live in a numpy object array and
        # `arr == key` on tuples broadcasts element-wise instead of comparing whole keys.
        ckey = "r:" + ",".join(f"{v:.6f}" for v in vec)
        # The concentration head splits on ratio AND absolute µM — two maps at the same
        # ratio but different concentration are genuinely different conditions for it,
        # while for composition they are the same one.
        akey = ckey + "|c:" + (",".join(f"{float(conc.get(s, 0.0)):.12g}" for s in subs)
                               if conc else "none")
        for j, ya in enumerate(_map_spectra(cube, local_mask, px_per_map, baseline_correct=not nnls_screen)):
            X.append(_composition_features(ya)); Y.append(vec)
            paths.append(it[0]); conds.append(ckey)   # map for pooling, condition for splits
            for yn in (_noisy_copies(ya, noise_aug, aug_rng) if (noise_aug and j) else ()):
                X.append(_composition_features(yn)); Y.append(vec)
                paths.append(it[0]); conds.append(ckey)   # same map → same split group
            # The µM head trains on the SAME rows as the composition head. It used to
            # see one mean spectrum per map — a few dozen examples, none of them a
            # pixel — and was then asked for a per-pixel concentration, which is the
            # out-of-distribution case that returned ~0 µM for a substance the
            # composition head was calling 61%.
            Xabs.append(ya)
            abs_paths.append(it[0]); abs_conds.append(akey)
            if conc:
                stock = [float(conc.get(s, 0.0)) for s in subs]
                # Stored values are source-solution concentrations. Components were
                # mixed at equal volumes, so final substrate concentration is divided
                # by the number of non-zero component solutions (2 or 3).
                dilution = (max(1, sum(v > 0 for v in stock))
                            if equal_volume_mix else 1)
                Cabs.append([v / dilution for v in stock])
            else:
                Cabs.append(None)
    # Optionally learn BACKGROUND as its own class: pixels from the blank reference map
    # labelled "100% blank". Without it the model must spread every spectrum — even bare
    # substrate — across the substances, so it can never say "nothing here".
    blank_name = None
    if include_blank:
        from dataset import discover_references, is_blank, base_and_batch
        blanks = [(base_and_batch(c)[0], pth)          # 'BLK-1' → class 'BLK'
                  for c, pth in discover_references(data_dir)
                  if is_blank(base_and_batch(c)[0])]
        if blanks:
            blank_name = blanks[0][0]
            for c, pth in blanks:
                try:
                    _w, cube, _m, _c = load_map(pth)
                except Exception:
                    continue
                # Blanks keep the MAP as their split group. Grouping every blank into one
                # "blank condition" would make a fold that holds it out train with no
                # background examples at all.
                for ya in _map_spectra(cube, mask, px_per_map or 60, spread=True):
                    X.append(_composition_features(ya))
                    Y.append([0.0] * len(subs) + [1.0])
                    paths.append(pth); conds.append(pth)
                    for yn in _noisy_copies(ya, noise_aug, aug_rng):   # blank at low SNR too
                        X.append(_composition_features(yn))
                        Y.append([0.0] * len(subs) + [1.0])
                        paths.append(pth); conds.append(pth)
            for i in range(len(Y)):                      # widen the mixture labels
                if len(Y[i]) == len(subs):
                    Y[i] = list(Y[i]) + [0.0]
            subs = list(subs) + [blank_name]
    X = np.array(X, np.float32); Xabs = np.array(Xabs); Y = np.array(Y, np.float32)
    paths = np.array(paths, object); conds = np.array(conds, object)
    if len(X) < 3:
        raise ValueError("need ≥3 mixtures to train.")

    pre = None
    if calib_path and use_pretrain:
        try:
            from io_utils import load_calibration_csv
            from calibration import calibrate
            ax_c, nc, dils = load_calibration_csv(calib_path); mc = (ax_c >= lo) & (ax_c <= hi)
            # `subs` may already carry the blank class appended above, but the calibration
            # only has the ANALYTES and P only has analyte rows. Gating on len(subs) made
            # 3 != 4 whenever include_blank was on, so the whole physics pre-training was
            # skipped — silently, no exception, nothing in the log. Compare against the
            # analyte count instead, and pad the pre-training targets with a zero blank
            # column so their width matches Y.
            asubs = subs[:-1] if blank_name else list(subs)
            dil = [(dils[nc.index(s)][0], np.asarray(dils[nc.index(s)][1])[:, mc])
                   for s in asubs if s in nc]
            if len(dil) == len(asubs):
                cal = calibrate(dil, P, asubs); rng = np.random.default_rng(0)
                # `sim_iso="sips_dq"` 는 DQ 만 Sips 지수로 시뮬레이션한다. DQ 희석계열은
                # Freundlich 를 뚜렷이 선호하고(R² 0.98 vs 0.89) THI 는 Langmuir 를
                # 선호하므로, 전부 바꾸지 않고 성분별로 둔다. m=1 은 Langmuir 와 동일.
                iso_m = None
                if sim_iso == "sips_dq":
                    from calibration import fit_sips
                    iso_m = np.ones(len(asubs))
                    for _i, _s in enumerate(asubs):
                        if _s.upper().startswith("DQ"):
                            iso_m[_i] = fit_sips(cal.C_series[_i], cal.B_series[_i])[2]
                    if progress:
                        progress("sim isotherm m = " + np.array2string(iso_m, precision=3))
                Xs, Cs = simulate_mixtures(P, cal.K, cal.gA, 5000, rng, noise=0.015,
                                           baseline=0.03, gain_lo=0.8, gain_hi=1.25,
                                           iso_m=iso_m,
                                           nuisance=(_nuisance_refs(data_dir, baseline, mask)
                                                     if sim_nuisance else None))
                Xs = _composition_features(Xs).astype(np.float32)
                Yp = np.array([_ratio(c) for c in Cs]).astype(np.float32)
                if blank_name:                     # simulated mixtures always hold analytes
                    Yp = np.hstack([Yp, np.zeros((len(Yp), 1), np.float32)])
                pre = (Xs, Yp)
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
        # Train for a FIXED number of epochs. Choosing the epoch on held-out maps was
        # tried and does not work at this dataset size: with 34 maps the 20% group split
        # leaves 7 validation maps, and the validation curve is noise rather than a
        # descent-then-rise. Three seeds picked epoch 75, 2 and 61, and after the "best"
        # epoch the curve keeps oscillating (seed 1: 0.53 at ep2, 1.17 at ep10, 0.75 at
        # ep100). Leave-one-map-out confirms the selection was costing accuracy --
        # 19.1% composition error training the full 350 against 22.8% at the selected 6.
        # A fixed budget is both better and reproducible; revisit when there are enough
        # maps for a validation split to mean something.
        selection = None
        selected_epochs = int(epochs)
        if len(va_sel_diag := _group_validation_indices(conds, seed=seed)[1]) and epoch_diagnostics:
            if progress:
                progress("recording the held-out epoch curve (diagnostic only)")
            tr_sel = _group_validation_indices(conds, seed=seed)[0]
            selection = train_composition(
                X[tr_sel], Y[tr_sel], len(subs), pretrain=pre, seed=seed,
                epochs_ft=epochs, X_val=X[va_sel_diag], Y_val=Y[va_sel_diag], progress=progress)
        comp = train_composition(X, Y, len(subs), pretrain=pre, seed=seed,
                                 epochs_ft=selected_epochs, progress=progress)
        comp_store = {"method": "mlp",
                      "comp_state": {k: v.cpu().numpy() for k, v in comp["state"].items()},
                      "comp_hidden": (256, 64),
                      "selected_epochs": selected_epochs,
                      "epoch_rule": "fixed",
                      "selection_val_loss": (selection.get("best_val") if selection else None),
                      "selection_val_history": (selection.get("val_hist", []) if selection else []),
                      "selection_level": "diagnostic" if selection else None}
        from dl_quantify import predict_composition
        tp = predict_composition(comp, X); loss_curve = comp.get("hist", [])
    train_eval = {"true": np.asarray(Y, float).tolist(), "pred": np.asarray(tp, float).tolist(),
                  "loss": loss_curve}

    # An INDEPENDENT batch (Role = test in Samples) beats leave-one-out: those maps were
    # measured separately and never touched training, so scoring them is the real check.
    test_eval = None
    if test_items:
        Xt_, row_paths, truth_by_map = [], [], {}
        for k, it in enumerate(test_items):
            if progress:
                progress(f"scoring held-out batch {k + 1}/{len(test_items)}")
            vec = _ratio([float(it[1].get(s_, 0.0)) for s_ in subs])
            if vec.sum() <= 0:
                continue
            if nnls_screen:
                _w, cube, _sm = nnls_hit_spectra(
                    data_dir, it[0], baseline=baseline, trim=trim,
                    min_frac=screen_min_frac, progress=None)
                local_mask = np.ones(cube.shape[1], bool)
            else:
                _w, cube, _m, _c = load_map(it[0]); local_mask = mask
            specs = _map_spectra(cube, local_mask, px_per_map,
                                 baseline_correct=not nnls_screen)
            truth_by_map[it[0]] = vec
            for ya in specs:
                Xt_.append(_composition_features(ya))
                row_paths.append(it[0])
        if Xt_:
            Pt = _fit_predict(method, X, Y, np.array(Xt_, np.float32), pre=pre,
                              epochs=epochs, seed=seed, n_components=n_components,
                              n_trees=n_trees, P_ref=P)
            ordered = list(dict.fromkeys(row_paths))
            test_eval = {
                "true": [truth_by_map[p].tolist() for p in ordered],
                "pred": [np.asarray(Pt)[np.asarray(row_paths) == p].mean(0).tolist()
                         for p in ordered],
                "paths": ordered,
                "rows_per_map": int(px_per_map) if px_per_map else 1,
            }
    loo_eval = None
    if loo and len(X) >= 3:                     # honest metrics: predict each held-out mixture
        # Leave one CONDITION out: every map of that composition leaves together, so a
        # repeat measurement cannot sit in training while its twin is being scored.
        uniq = list(dict.fromkeys(conds.tolist()))
        tv, pv, pl = [], [], []
        for i, ck in enumerate(uniq):
            if progress:
                progress(f"leave-one-condition-out {i + 1}/{len(uniq)}")
            te = np.where(conds == ck)[0]; tr = np.where(conds != ck)[0]
            if not len(tr):
                continue
            pred = _fit_predict(method, X[tr], Y[tr], X[te], pre=pre, epochs=epochs,
                                seed=seed + i, n_components=n_components,
                                n_trees=n_trees, P_ref=P)
            pred = np.asarray(pred, float)
            # Report one row per held-out MAP — downstream (_annotate, Recovery) looks
            # these up by file path, and a per-map number is also what the user reads.
            te_paths = paths[te]
            for mp in dict.fromkeys(te_paths.tolist()):
                sel = te_paths == mp
                tv.append(Y[te[0]].tolist()); pv.append(pred[sel].mean(0).tolist())
                pl.append(mp)
        loo_eval = {"true": tv, "pred": pv, "paths": pl, "level": "condition"}

    uM = None
    # The concentration head covers the COMPOUNDS only. A blank class has no
    # concentration, but appending it to `subs` widened the head to 4 outputs while the
    # labels in Cabs stayed 3 wide, so enabling "learn background" made training die with
    # "size of tensor a (4) must match the size of tensor b (3)".
    conc_subs = [s_ for s_ in subs if s_ != blank_name] if blank_name else list(subs)
    # Concentration truth belongs to the experimental MAP, not to each local hotspot.
    # Pixels are encoded separately, but one robust median loss is evaluated per map.
    have = [i for i in range(len(Xabs)) if Cabs[i] is not None and any(c > 0 for c in Cabs[i])]
    if len(have) >= 3:
        if progress:
            progress("training concentration head (map-pooled)")
        hv = np.array(have)
        C = np.array([list(Cabs[i])[:len(conc_subs)] if Cabs[i] is not None
                      else [0.0] * len(conc_subs) for i in range(len(Xabs))], float)
        # `gp` stays the MAP — it is the POOLING key (the label is one number per map, so
        # the loss compares the per-map median) and the context-feature key. `gcond` is the
        # SPLITTING key. Conflating the two would pool repeats of one condition into a
        # single pseudo-map and change what the head is trained to predict.
        gp = np.asarray(abs_paths, object)[hv]
        gcond = np.asarray(abs_conds, object)[hv]
        from dl_quantify import surface_composition
        Rabs = surface_composition(_composition_features(Xabs[hv], 'legacy_l2'), P)
        Xctx = _concentration_context_features(Xabs[hv], gp, Rabs)
        target = (np.log10(np.clip(C[hv], 1e-8, None)) + 6.0).astype(np.float32)

        def make_conc_net():
            return nn.Sequential(nn.Linear(Xctx.shape[1], 128), nn.BatchNorm1d(128), nn.ReLU(),
                                 nn.Dropout(0.25), nn.Linear(128, 32), nn.ReLU(),
                                 nn.Linear(32, len(conc_subs)))

        def pooled_loss(pred, truth, group_values, consistency_weight=0.05):
            terms = []
            for g in dict.fromkeys(group_values.tolist()):
                idx = np.where(group_values == g)[0]
                ii = torch.as_tensor(idx, dtype=torch.long)
                terms.append(torch.nn.functional.smooth_l1_loss(
                    torch.quantile(pred[ii], 0.5, dim=0), truth[ii[0]], beta=0.25))
            map_term = torch.stack(terms).mean()
            if consistency_weight:
                map_term = map_term + float(consistency_weight) * torch.nn.functional.smooth_l1_loss(
                    pred, truth, beta=0.5)
            return map_term

        # Select capacity using entire held-out CONDITIONS, never random pixels from a map.
        # Fixed budget here too, for the same reason as the composition head above.
        tr_sel, va_sel = _group_validation_indices(gcond, seed=seed + 17)
        selected_uM_epochs = int(epochs); selection_uM_val = []; best_val = float("inf"); stale = 0
        if len(va_sel) and epoch_diagnostics:
            smu = Xctx[tr_sel].mean(0); ssd = Xctx[tr_sel].std(0) + 1e-8
            sXtr = torch.tensor(((Xctx[tr_sel] - smu) / ssd).astype(np.float32))
            sXva = torch.tensor(((Xctx[va_sel] - smu) / ssd).astype(np.float32))
            sYtr = torch.tensor(target[tr_sel]); sYva = torch.tensor(target[va_sel])
            torch.manual_seed(seed + 17); snet = make_conc_net()
            sop = torch.optim.Adam(snet.parameters(), lr=1e-3, weight_decay=3e-3)
            for ep in range(int(epochs)):
                snet.train(); sop.zero_grad(); sp = snet(sXtr)
                sl = pooled_loss(sp, sYtr, gp[tr_sel]); sl.backward(); sop.step()
                snet.eval()
                with torch.no_grad():
                    sv = float(pooled_loss(snet(sXva), sYva, gp[va_sel], 0.0))
                selection_uM_val.append(sv)
                if sv < best_val - 1e-4:          # recorded, but no longer chooses the epoch
                    best_val = sv; stale = 0
                else:
                    stale += 1
                if progress and (ep % 15 == 0 or ep == int(epochs) - 1):
                    progress(f"selecting concentration epoch {ep + 1}/{epochs}  val {sv:.3f}")
                if stale >= 40:
                    break

        # Refit on every map, but only for the validation-selected number of epochs.
        mu = Xctx.mean(0); sd = Xctx.std(0) + 1e-8
        Xe = ((Xctx - mu) / sd).astype(np.float32)
        Xt = torch.tensor(Xe); Yt = torch.tensor(target)
        torch.manual_seed(seed); net = make_conc_net()
        op = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=3e-3)
        loss_curve_uM = []
        for ep in range(selected_uM_epochs):
            net.train(); op.zero_grad(); loss = pooled_loss(net(Xt), Yt, gp)
            loss.backward(); op.step(); loss_curve_uM.append(float(loss.detach()))
            if progress and (ep % 15 == 0 or ep == selected_uM_epochs - 1):
                progress(f"concentration refit {ep + 1}/{selected_uM_epochs}")
        net.eval()
        dist = np.sqrt(np.mean(Xe ** 2, axis=1))
        ranges = []
        for j in range(len(conc_subs)):
            positive = C[hv, j][C[hv, j] > 0]
            ranges.append([float(positive.min()), float(positive.max())]
                          if len(positive) else [float("nan"), float("nan")])
        uM = {"kind": "map_pooled_pixel_concentration_v1",
              "state": {k: v.detach().numpy() for k, v in net.state_dict().items()},
              "mu": mu, "sd": sd, "subs": list(conc_subs), "pool": "median_log10",
              "input_mode": "nnls_ratio+intensity_quantiles_v1", "hidden": (128, 32),
              "raw_n_feat": Xabs.shape[1], "ratio_n_feat": len(conc_subs),
              "ood_threshold": float(np.quantile(dist, 0.99)),
              "ranges_M": np.asarray(ranges, float), "train_loss": loss_curve_uM,
              "selected_epochs": selected_uM_epochs,
              "selection_val_loss": (best_val if selection_uM_val else None),
              "selection_val_history": selection_uM_val,
              "selection_level": "condition",
              "n_maps": len(dict.fromkeys(gp.tolist())), "n_pixels": len(hv)}

        # Concentration validation follows the same leave-one-CONDITION-out boundary as
        # composition. Each held-out map is pooled from its pixel predictions.
        if loo:
            loo_true, loo_pred, loo_paths = [], [], []
            unique_conds = list(dict.fromkeys(gcond.tolist()))
            for fold_i, held in enumerate(unique_conds):
                te = np.where(gcond == held)[0]; tr = np.where(gcond != held)[0]
                if len(te) == 0 or len(tr) < 3:
                    continue
                if progress:
                    progress(f"concentration leave-one-condition-out {fold_i + 1}/{len(unique_conds)}")
                ftrain = _concentration_context_features(Xabs[hv][tr], gp[tr], Rabs[tr])
                fmu = ftrain.mean(0); fsd = ftrain.std(0) + 1e-8
                fX = torch.tensor(((ftrain - fmu) / fsd).astype(np.float32))
                fY = torch.tensor((np.log10(np.clip(C[hv][tr], 1e-8, None)) + 6.0).astype(np.float32))
                fgp = gp[tr]
                fgroups = [np.where(fgp == g)[0] for g in dict.fromkeys(fgp.tolist())]
                torch.manual_seed(seed + 1000 + fold_i)
                fnet = nn.Sequential(nn.Linear(ftrain.shape[1], 256), nn.BatchNorm1d(256), nn.ReLU(),
                                     nn.Dropout(0.15), nn.Linear(256, 64), nn.ReLU(),
                                     nn.Linear(64, len(conc_subs)))
                fop = torch.optim.Adam(fnet.parameters(), lr=1e-3, weight_decay=1e-3)
                for _ in range(epochs):
                    fnet.train(); fop.zero_grad(); fp = fnet(fX); fl = []
                    for fi in fgroups:
                        fii = torch.as_tensor(fi, dtype=torch.long)
                        fl.append(torch.nn.functional.smooth_l1_loss(
                            torch.quantile(fp[fii], 0.5, dim=0), fY[fii[0]], beta=0.25))
                    fmap = torch.stack(fl).mean()
                    fcons = torch.nn.functional.smooth_l1_loss(fp, fY, beta=0.5)
                    (fmap + 0.05 * fcons).backward(); fop.step()
                fnet.eval()
                ftest = _concentration_context_features(Xabs[hv][te], ratios=Rabs[te])
                test_x = torch.tensor(((ftest - fmu) / fsd).astype(np.float32))
                with torch.no_grad():
                    logits = fnet(test_x).numpy()
                # One row per held-out MAP, pooled over that map's own pixels — the
                # downstream lookup is by file path, and a condition can hold several maps.
                te_paths = gp[te]
                for mp in dict.fromkeys(te_paths.tolist()):
                    sel = np.where(te_paths == mp)[0]
                    loo_true.append((C[hv][te[sel[0]]] * 1e6).tolist())
                    loo_pred.append((10.0 ** np.clip(np.median(logits[sel], axis=0),
                                                     -3.0, 6.0)).tolist())
                    loo_paths.append(mp)
            uM["loo_eval"] = {"true_uM": loo_true, "pred_uM": loo_pred,
                              "paths": loo_paths, "level": "condition"}

    return {"subs": subs, "lo": lo, "hi": hi, "n_feat": int(mask.sum()), "P": P,
            "feature_mode": "log1p_raw",
            "uM": uM, "n_train": int(len(X)), "has_uM": uM is not None,
            "n_maps": int(len(set(paths.tolist()))),
            "px_per_map": int(px_per_map),
            "training_level": "map_mean" if int(px_per_map) == 0 else "pixels",
            "nnls_screen": bool(nnls_screen), "screen_min_frac": float(screen_min_frac),
            "equal_volume_mix": bool(equal_volume_mix),
            "concentration_basis": ("equal-volume final mixture"
                                    if equal_volume_mix else "values supplied in Samples"),
            "screen_stats": screen_stats, "data_dir": str(data_dir),
            "baseline": bool(baseline), "trim": trim,
            "train_eval": train_eval, "loo_eval": loo_eval, "test_eval": test_eval,
            "blank": blank_name, **comp_store}


def apply_model(model, wn, cube):
    """Score a test map (or single spectrum): returns {composition: {name: frac},
    uM: {name: µM} or None}. Handles the model's own window + standardisation."""
    if model.get("kind") == "pixel_surface_v2":
        from pixel_surface import apply_pixel_surface
        p = apply_pixel_surface(model, wn, np.atleast_2d(cube),
                                preprocessed=False).mean(axis=0)
        return {"composition": {s: float(p[i]) for i, s in enumerate(model["subs"])},
                "uM": None}
    lo, hi = model["lo"], model["hi"]
    mask = (np.asarray(wn) >= lo) & (np.asarray(wn) <= hi)
    # Follow the model's OWN preprocessing. This used to force ALS unconditionally, so a
    # model trained with baseline=False (references already corrected upstream) had the
    # baseline removed a second time here — the test map went through a preprocessing the
    # training spectra never saw. Measured cost on the 260806 model: composition error
    # 41.1% double-corrected against 25.5% honouring the flag.
    ya = _mean_spectrum(cube, mask, baseline_correct=bool(model.get("baseline", True)))
    subs = model["subs"]
    xfeat = _composition_features(ya, model.get("feature_mode", "legacy_l2")).astype(np.float32)
    method = model.get("method", "mlp")
    if method in ("pls", "rf"):                                # sklearn composition head
        comp = np.clip(np.asarray(model["sk"].predict(xfeat[None, :])[0], float), 0, None)
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
            comp = torch.softmax(net(torch.tensor(xfeat[None, :])), 1).numpy()[0]
    out = {"composition": {subs[k]: float(comp[k]) for k in range(len(subs))}, "uM": None}
    if model.get("uM"):
        # Delegate to the per-pixel head and pool, rather than rebuilding the net here.
        # The old code hardcoded Linear(n_feat, 256)/(256, 64) and fed the mean spectrum,
        # but the concentration head takes 16 context features through (128, 32) — so it
        # raised a size-mismatch on every model trained with the current head, i.e. the
        # map-level µM readout was dead. Pooling the pixels is also what the head was
        # trained for: its loss compares the per-map MEDIAN prediction with the label.
        per_px, usubs = apply_uM_pixels(model, wn, cube)
        if per_px is not None and len(per_px):
            med = np.median(np.asarray(per_px, float), axis=0)
            out["uM"] = {usubs[k]: float(med[k]) for k in range(len(usubs))}
    return out


def kfold_stability(data_dir, items, method="mlp", folds=5, progress=None, seed=0, **kw):
    """Repeat the held-out check over every fold of an even 1-in-``folds`` split (start
    offsets 0..folds-1) so a single lucky/unlucky test set cannot set the headline number.
    Returns {"errors": [...per fold], "mean", "sd"} of the composition error.

    Folds are drawn over CONDITIONS, not maps. The old version sorted maps by basename and
    strided ``pos % folds``; repeats of one mixture sort adjacently ('1-1', '1-2', '1-3'),
    so that stride guaranteed they landed in different folds — the held-out map always had
    a twin in training. Grouping by composition and shuffling the groups removes both the
    leak and the systematic ordering."""
    from dl_quantify import _ratio as _r
    subs0 = _refs(data_dir, kw.get("baseline", True), kw.get("trim"))[0]

    def _ck(it):
        v = _r([float(it[1].get(s, 0.0)) for s in subs0])
        return "r:" + ",".join(f"{x:.6f}" for x in v)

    groups = {}
    for i, it in enumerate(items):
        groups.setdefault(_ck(it), []).append(i)
    keys = sorted(groups)                       # sorted first so the shuffle is reproducible
    order = np.random.default_rng(int(seed)).permutation(len(keys))
    errs = []
    for f in range(int(folds)):
        te_keys = {keys[i] for pos, i in enumerate(order) if pos % folds == f}
        te = [items[i] for k in te_keys for i in groups[k]]
        tr = [items[i] for k in keys if k not in te_keys for i in groups[k]]
        if len(tr) < 3 or not te:
            continue
        if progress:
            progress(f"fold {f + 1}/{folds} — {len(tr)} train / {len(te)} test")
        m = train_model(data_dir, tr, method=method, test_items=te, progress=None,
                        seed=seed, **kw)
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
                  n_components=8, n_trees=300, px_per_map=0, rf_max_features=None):
    """Leave-one-out comparison of the composition methods on the SAME mixtures — the
    honest counterpart to the train-set numbers. Maps are loaded once, then every method
    is refit per fold. Returns {method: {"true", "pred"}} plus "subs"."""
    from real_data import load_map
    from dl_quantify import simulate_mixtures, _ratio
    subs, wn, mask, P, lo, hi = _refs(data_dir, baseline, trim)
    X, Y, gkey = [], [], []
    for k, it in enumerate(items):
        if progress:
            progress(f"loading maps {k + 1}/{len(items)}")
        vec = _ratio([float(it[1].get(s, 0.0)) for s in subs])
        if vec.sum() <= 0:
            continue
        ckey = "r:" + ",".join(f"{v:.6f}" for v in vec)
        _w, cube, _m, _c = load_map(it[0])
        for ya in _map_spectra(cube, mask, px_per_map):
            X.append(_composition_features(ya)); Y.append(vec)
            gkey.append(ckey)                         # group = the CONDITION, not the map
    X = np.array(X, np.float32); Y = np.array(Y, np.float32)
    gkey = np.array(gkey, object)
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
                                           baseline=0.03, gain_lo=0.8, gain_hi=1.25,
                                           nuisance=_nuisance_refs(data_dir, baseline, mask))
                Xs = _composition_features(Xs).astype(np.float32)
                pre = (Xs, np.array([_ratio(c) for c in Cs]).astype(np.float32))
        except Exception:
            pre = None

    out = {"subs": subs}
    for mi, meth in enumerate(methods):
        uniq = list(dict.fromkeys(gkey.tolist()))
        tv, pv = [], []
        for i, mp in enumerate(uniq):
            if progress:
                progress(f"{meth.upper()} leave-one-condition-out {i + 1}/{len(uniq)}  "
                         f"[{mi + 1}/{len(methods)} methods]")
            te = np.where(gkey == mp)[0]; tr = np.where(gkey != mp)[0]
            pred = _fit_predict(meth, X[tr], Y[tr], X[te], pre=pre, epochs=epochs,
                                seed=seed + i, n_components=n_components,
                                n_trees=n_trees, P_ref=P,
                                rf_max_features=rf_max_features)
            tv.append(Y[te[0]].tolist()); pv.append(np.asarray(pred, float).mean(0).tolist())
        out[meth] = {"true": tv, "pred": pv}
    return out


def apply_model_pixels(model, wn, spectra):
    """Composition for EVERY pixel spectrum (n_px, n_wn) → (n_px, n_subs), rows summing
    to 1. Same heads as apply_model, just batched, so the Real-data tab can draw a
    per-pixel composition map from the trained model instead of only a map-level answer."""
    if model.get("kind") == "pixel_surface_v2":
        from pixel_surface import apply_pixel_surface
        return apply_pixel_surface(model, wn, spectra, preprocessed=True)
    lo, hi = model["lo"], model["hi"]
    wn = np.asarray(wn); mask = (wn >= lo) & (wn <= hi)
    X = np.asarray(spectra, float)
    if X.shape[1] == len(wn):
        X = X[:, mask]
    X = _composition_features(X, model.get("feature_mode", "legacy_l2"))
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


def apply_uM_pixels(model, wn, spectra, return_meta=False):
    """Per-pixel absolute concentration from the model's µM head: (n_px, n_conc) in µM,
    plus the substance names that head covers. The head is trained on log10 µM labels,
    so it is an order-of-magnitude estimate — but it is the SAME estimator the map-level
    readout uses, batched, which is what lets the concentration maps come from the model
    instead of the Langmuir isotherm."""
    if model.get("kind") == "pixel_surface_v2" and model.get("uM", {}).get("kind") == "pixel_concentration_v2":
        from pixel_surface import apply_pixel_concentration
        out = apply_pixel_concentration(model, wn, spectra)
        return out if return_meta else out[:2]
    import torch, torch.nn as nn
    u = model.get("uM")
    if not u:
        return None, []
    usubs = u.get("subs") or model["subs"]
    wn = np.asarray(wn); mask = (wn >= model["lo"]) & (wn <= model["hi"])
    X = np.asarray(spectra, float)
    if X.shape[1] == len(wn):
        X = X[:, mask]
    if u.get("input_mode") == "nnls_ratio+intensity_quantiles_v1":
        from dl_quantify import surface_composition
        ratios = surface_composition(_composition_features(X, "legacy_l2"), model["P"])
        X = _concentration_context_features(X, ratios=ratios)
    elif u.get("input_mode") == "pixel_log1p+map_median+p90+nnls_ratio":
        from dl_quantify import surface_composition
        ratios = surface_composition(_composition_features(X, "legacy_l2"), model["P"])
        X = _concentration_context_features(X, ratios=ratios)
    elif u.get("input_mode") == "pixel_log1p+map_median+p90":
        X = _concentration_context_features(X)
    hidden = tuple(u.get("hidden", (256, 64)))
    net = nn.Sequential(nn.Linear(len(u["mu"]), hidden[0]), nn.BatchNorm1d(hidden[0]), nn.ReLU(),
                        nn.Dropout(0.25 if u.get("hidden") else 0.15),
                        nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
                        nn.Linear(hidden[1], len(usubs)))
    net.load_state_dict({k: torch.tensor(v) for k, v in u["state"].items()}); net.eval()
    Xe = ((X - u["mu"]) / u["sd"]).astype(np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(Xe), 512):
            out.append(net(torch.tensor(Xe[i:i + 512])).numpy())
    logv = np.vstack(out)
    result = (10.0 ** np.clip(logv, -3.0, 6.0), list(usubs))
    if not return_meta:
        return result
    meta = {}
    if u.get("kind") == "map_pooled_pixel_concentration_v1":
        dist = np.sqrt(np.mean(Xe ** 2, axis=1))
        ood = dist > float(u.get("ood_threshold", np.inf))
        meta = {"feature_ood": ood,
                "component_ood": np.repeat(ood[:, None], len(usubs), axis=1),
                "ranges_M": u.get("ranges_M")}
    return (*result, meta)


def apply_recovery(model, items, progress=None):
    """Apply an already-trained composition model to each known-ratio mixture (NO training)
    → list of {name, nominal, mean[, uM_pred, uM_true]} in composition.SUBSTANCES order,
    the same shape dl_recovery returns, so the Recovery plots don't change. This is how
    Recovery 'inherits' the model trained once in the Model tab."""
    import os
    from real_data import load_map
    from dataset import is_blank
    all_subs = list(model["subs"])
    subs = [s for s in all_subs if not is_blank(s)]
    # A mixture the model TRAINED on must not be scored by that model — it would just
    # recite its own answer (recovery collapses to ~100%). Training already computed a
    # held-out prediction for each of those maps, so reuse it and only run the model on
    # mixtures it has genuinely never seen.
    held = {}
    train_held = set()
    for ev in (model.get("test_eval") or {}, model.get("loo_eval") or {}):
        if ev.get("paths"):
            for pth, pred in zip(ev["paths"], ev["pred"]):
                held[os.path.normcase(os.path.normpath(pth))] = pred
    for pth in (model.get("loo_eval") or {}).get("paths", []):
        train_held.add(os.path.normcase(os.path.normpath(pth)))
    held_uM = {}
    uev = (model.get("uM") or {}).get("loo_eval") or {}
    for pth, pred in zip(uev.get("paths", []), uev.get("pred_uM", [])):
        held_uM[os.path.normcase(os.path.normpath(pth))] = pred
    out = []
    for k, it in enumerate(items):
        if progress:
            progress(f"applying model — {k + 1}/{len(items)}")
        path, ratio = it[0], it[1]
        conc = it[2] if len(it) > 2 else None
        key = os.path.normcase(os.path.normpath(path))
        if key in held:                                  # held-out composition from training
            comp = {sn: float(held[key][all_subs.index(sn)]) for sn in subs}
            res = {"uM": None}
            # Never score a training map with the full concentration head. Use the
            # prediction cached from a head fitted without that map.
            if key in held_uM:
                vals = held_uM[key]
                unames = (model.get("uM") or {}).get("subs", subs)
                res["uM"] = {sn: float(vals[j]) for j, sn in enumerate(unames)}
            elif model.get("nnls_screen") and model.get("uM") and key not in train_held:
                wn, hit_cube, smeta = nnls_hit_spectra(
                    model.get("data_dir") or os.path.dirname(path), path,
                    baseline=model.get("baseline", True), trim=model.get("trim"),
                    min_frac=model.get("screen_min_frac", 0.15), progress=None)
                um, unames = apply_uM_pixels(model, wn, hit_cube)
                if um is not None:
                    med = np.median(np.asarray(um, float), axis=0)
                    res["uM"] = {sn: float(med[j]) for j, sn in enumerate(unames)}
                    res["uM_p10"] = {sn: float(np.percentile(um[:, j], 10))
                                       for j, sn in enumerate(unames)}
                    res["uM_p90"] = {sn: float(np.percentile(um[:, j], 90))
                                       for j, sn in enumerate(unames)}
        elif model.get("nnls_screen"):
            wn, hit_cube, smeta = nnls_hit_spectra(
                model.get("data_dir") or os.path.dirname(path), path,
                baseline=model.get("baseline", True), trim=model.get("trim"),
                min_frac=model.get("screen_min_frac", 0.15), progress=None)
            pp = apply_model_pixels(model, wn, hit_cube)
            pm = np.asarray(pp, float).mean(axis=0)
            comp = {subs[j]: float(pm[j]) for j in range(len(subs))}
            um, unames = apply_uM_pixels(model, wn, hit_cube)
            res = {"uM": None, "hit_fraction": smeta["hit_fraction"]}
            if um is not None:
                med = np.median(np.asarray(um, float), axis=0)
                res["uM"] = {sn: float(med[j]) for j, sn in enumerate(unames)}
                res["uM_p10"] = {sn: float(np.percentile(um[:, j], 10))
                                   for j, sn in enumerate(unames)}
                res["uM_p90"] = {sn: float(np.percentile(um[:, j], 90))
                                   for j, sn in enumerate(unames)}
        else:
            wn, cube, _m, _c = load_map(path)
            res = apply_model(model, wn, cube); comp = res["composition"]
        s = sum(float(ratio.get(sn, 0)) for sn in subs)
        nom = np.zeros(len(subs)); mn = np.zeros(len(subs))
        for sn in subs:
            o = subs.index(sn)
            nom[o] = float(ratio.get(sn, 0)) / s if s > 0 else 0.0
            mn[o] = comp.get(sn, 0.0)
        row = {"name": os.path.basename(path).replace("_corrected", "").replace(".csv", ""),
               "nominal": nom, "mean": mn}
        if res.get("uM"):
            row["uM_pred"] = {sn: res["uM"][sn] for sn in subs}
            row["uM_p10"] = res.get("uM_p10")
            row["uM_p90"] = res.get("uM_p90")
            if conc:
                dilution = (max(1, sum(float(conc.get(sn, 0)) > 0 for sn in subs))
                            if model.get("equal_volume_mix", False) else 1)
                row["uM_true"] = {sn: conc[sn] * 1e6 / dilution
                                  for sn in subs if conc.get(sn, 0) > 0}
        out.append(row)
    return out


def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)
