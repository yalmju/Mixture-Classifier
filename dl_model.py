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


def _fit_torch_bag(method, Xtr, Ytr, train_maps, Xte, *, pre=None, epochs=350,
                   seed=0, predict_batch=256, pixels_per_map_step=32,
                   return_net=False, progress=None):
    """Train a spectrum head with map-pooled supervision and predict test pixels."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from dl_quantify import _spec_net
    Xtr = np.asarray(Xtr, np.float32); Ytr = np.asarray(Ytr, np.float32)
    Xte = np.atleast_2d(np.asarray(Xte, np.float32)); maps = np.asarray(train_maps, object)
    n_comp = Ytr.shape[1]; torch.manual_seed(int(seed))
    net = _cnn(Xtr.shape[1], n_comp) if method == "cnn" else _spec_net(Xtr.shape[1], n_comp)

    def weighted_l1(pred, target):
        w = 1.0 + 2.0 * (1.0 - target)
        return (w * (pred - target).abs()).sum(1).mean()

    if pre is not None and len(pre[0]):
        xp = torch.tensor(np.asarray(pre[0], np.float32)); yp = torch.tensor(np.asarray(pre[1], np.float32))
        op = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        dl = DataLoader(TensorDataset(xp, yp), batch_size=256, shuffle=True)
        for _ in range(25):
            net.train()
            for xb, yb in dl:
                op.zero_grad(); weighted_l1(torch.softmax(net(xb), 1), yb).backward(); op.step()

    by_map = [np.where(maps == mp)[0] for mp in dict.fromkeys(maps.tolist())]
    rng = np.random.default_rng(int(seed)); op = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-3)
    map_batch = 1 if method == "cnn" else 4
    for _ep in range(int(epochs)):
        order = rng.permutation(len(by_map)); net.train()
        for start in range(0, len(order), map_batch):
            groups = []
            for i in order[start:start + map_batch]:
                g = by_map[i]
                # Rotate through a large map instead of forwarding all 400 spectra at
                # every gradient step. Across epochs every pixel participates; held-out
                # scoring below still pools every available test pixel.
                if len(g) > int(pixels_per_map_step):
                    g = rng.choice(g, size=int(pixels_per_map_step), replace=False)
                groups.append(g)
            idx = np.concatenate(groups)
            prob = torch.softmax(net(torch.tensor(Xtr[idx])), 1); pooled = []; targets = []; pos = 0
            for g in groups:
                pooled.append(prob[pos:pos + len(g)].mean(0)); targets.append(Ytr[g[0]]); pos += len(g)
            pm = torch.stack(pooled); yt = torch.tensor(np.asarray(targets, np.float32))
            loss = weighted_l1(pm, yt) + 0.05 * weighted_l1(prob, torch.tensor(Ytr[idx]))
            op.zero_grad(); loss.backward(); op.step()
        if progress and (_ep % 10 == 0 or _ep == int(epochs) - 1):
            progress(f"map-pooled epoch {_ep + 1}/{int(epochs)}  "
                     f"loss {float(loss.detach()):.3f}")

    net.eval(); out = []; bs = 64 if method == "cnn" else int(predict_batch)
    with torch.no_grad():
        for start in range(0, len(Xte), bs):
            out.append(torch.softmax(net(torch.tensor(Xte[start:start + bs])), 1).numpy())
    result = np.vstack(out)
    return (result, net) if return_net else result


def _fit_predict(method, Xtr, Ytr, Xte, *, pre=None, epochs=350, seed=0,
                 n_components=8, n_trees=300, P_ref=None, rf_max_features=None,
                 band_mask=None, mcr_iter=6, train_maps=None):
    """Fit ``method`` on (Xtr, Ytr) and return composition predictions for Xte, rows
    summing to 1. Shared by the full-data fit and each leave-one-out fold so both use
    exactly the same estimator."""
    n_comp = Ytr.shape[1]
    if method == "nnls":                       # classical baseline: no training at all
        from dl_quantify import surface_composition
        raw = np.expm1(np.clip(np.atleast_2d(Xte), 0, None))
        p = surface_composition(_composition_features(raw, "legacy_l2"), P_ref)
    elif method == "band":                     # VIP-band NNLS: no training either — the
        # same NNLS, but fit only on each compound's least-cross-talk marker windows
        # (``band_mask``, computed once by the caller from the pure templates). This is
        # the Validate tab's VIP-band decomposition, scored on the benchmark's terms.
        from dl_quantify import surface_composition
        raw = np.expm1(np.clip(np.atleast_2d(Xte), 0, None))
        if band_mask is not None and np.asarray(band_mask, bool).any():
            bm = np.asarray(band_mask, bool)
            raw = raw[:, bm]
            Pm = np.asarray(P_ref, float)[:, bm]
            Pm = Pm / (np.linalg.norm(Pm, axis=1, keepdims=True) + 1e-12)
        else:                                  # no usable bands → plain NNLS, honestly
            Pm = P_ref
        p = surface_composition(_composition_features(raw, "legacy_l2"), Pm)
    elif method == "null":
        # The floor: always answer with the mean composition of the TRAINING conditions.
        # A method that cannot beat this has learned nothing from the spectrum. Honest by
        # construction here — Ytr already excludes the held-out condition.
        if train_maps is None:
            centre = np.asarray(Ytr, float).mean(axis=0, keepdims=True)
        else:
            tm = np.asarray(train_maps, object)
            centre = np.asarray([Ytr[np.where(tm == mk)[0][0]]
                                 for mk in dict.fromkeys(tm.tolist())], float).mean(
                                     axis=0, keepdims=True)
        p = np.repeat(centre, len(np.atleast_2d(Xte)), axis=0)
    elif method == "nnls_rf":
        # NNLS, then ONE response factor per substance (dl_quantify.fit_response_factors),
        # fitted on the training conditions only. The control for the paper's claim: if
        # three numbers close most of the NNLS-to-learned gap, that gap was response
        # factors, not spectral learning.
        from dl_quantify import (surface_composition, fit_response_factors,
                                 apply_response_factors)
        raw_tr = np.expm1(np.clip(Xtr, 0, None))
        raw_te = np.expm1(np.clip(np.atleast_2d(Xte), 0, None))
        s_tr = surface_composition(_composition_features(raw_tr, "legacy_l2"), P_ref)
        r = fit_response_factors(s_tr, np.asarray(Ytr, float))
        s_te = surface_composition(_composition_features(raw_te, "legacy_l2"), P_ref)
        p = apply_response_factors(s_te, r)
    elif method == "mcr":
        # MCR-ALS refines the component SPECTRA from the data (seeded by the pure
        # templates), then decomposes the held-out spectrum on the refined ones — the
        # answer to "your templates are not the real surface spectra". Uses no labels at
        # all, and refines on the training rows only, so nothing leaks.
        from unmix import _mcr_als, _l2
        from dl_quantify import surface_composition
        raw_tr = np.expm1(np.clip(Xtr, 0, None))
        raw_te = np.expm1(np.clip(np.atleast_2d(Xte), 0, None))
        _C, S = _mcr_als(_l2(raw_tr), np.asarray(P_ref, float), n_iter=int(mcr_iter))
        p = surface_composition(_l2(raw_te), S)
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
    elif method in ("cnn", "mlp") and train_maps is not None:
        return _fit_torch_bag(method, Xtr, Ytr, train_maps, Xte, pre=pre,
                              epochs=epochs, seed=seed)
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


def _representative_indices(cube, n_px, seed=0):
    """Choose measured medoids from shape + log-intensity clusters."""
    cube = np.asarray(cube, float)
    n = len(cube); k = min(max(1, int(n_px)), n)
    if k >= n:
        return np.arange(n, dtype=int)
    bins = np.array_split(np.arange(cube.shape[1]), min(32, cube.shape[1]))
    shape = np.column_stack([cube[:, b].mean(axis=1) for b in bins])
    total = np.log1p(np.clip(cube.sum(axis=1), 0, None))
    shape /= np.linalg.norm(shape, axis=1, keepdims=True) + 1e-12
    iz = (total - np.median(total)) / (np.std(total) + 1e-12)
    feat = np.column_stack([shape, 0.35 * iz])
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=k, random_state=int(seed), n_init=3,
                         batch_size=min(512, n), max_iter=100,
                         reassignment_ratio=0.0)
    labels = km.fit_predict(feat)
    chosen = []
    for j in range(k):
        idx = np.where(labels == j)[0]
        if not len(idx):
            continue
        d2 = np.sum((feat[idx] - km.cluster_centers_[j]) ** 2, axis=1)
        chosen.append(int(idx[int(np.argmin(d2))]))
    chosen = list(dict.fromkeys(chosen))
    while len(chosen) < k:
        remaining = np.setdiff1d(np.arange(n), np.asarray(chosen, int),
                                 assume_unique=False)
        if not len(remaining):
            break
        if chosen:
            d2 = np.min(np.sum((feat[remaining, None, :] -
                                feat[np.asarray(chosen), :][None, :, :]) ** 2,
                               axis=2), axis=1)
            chosen.append(int(remaining[int(np.argmax(d2))]))
        else:
            chosen.append(int(remaining[0]))
    return np.asarray(sorted(chosen, key=lambda i: (-total[i], i)), int)


def _stable_sampling_seed(map_id, seed=0):
    """Same map + configured seed gives the same representatives in every run."""
    import hashlib
    key = os.path.normcase(os.path.normpath(str(map_id or "map"))).encode(
        "utf-8", "replace")
    return int(seed) ^ int.from_bytes(
        hashlib.blake2s(key, digest_size=4).digest(), "little")


def _map_spectra(cube, mask, n_px=0, spread=False, baseline_correct=True,
                 sampling="legacy", sampling_seed=0, map_id=None):
    """Return representative spectra from one map.

    ``n_px=0`` returns one intensity-weighted mean spectrum. ``n_px>0`` returns
    individual pixels, brightest first unless ``spread`` is requested. Pixels from the
    same map must remain in the same validation split because they share one map label.
    """
    from sers_mixture import als_baseline, desaturate
    cube = np.asarray(cube, float)[:, mask]
    cube, _sat = desaturate(cube)      # bridge clipped plateaus before any ALS
    ok = _sat <= 0.01                  # quarantine: clipped pixels don't train
    if ok.any() and not ok.all():
        cube = cube[ok]
    w = cube.sum(1); wn_ = w / (w.sum() + 1e-12)
    mean = wn_ @ cube
    out = ([] if n_px else
           [np.clip(mean - als_baseline(mean), 0, None) if baseline_correct
            else np.clip(mean, 0, None)])
    if n_px and len(cube) > 1:
        if str(sampling).lower() == "representative":
            order = _representative_indices(
                cube, int(n_px), _stable_sampling_seed(map_id, sampling_seed))
        else:
            order = np.argsort(-w)
            if spread:
                k = max(1, len(order) // int(n_px))
                order = order[::k][:int(n_px)]
        for i in order[:int(n_px)]:
            y = cube[i]
            out.append(np.clip(y - als_baseline(y), 0, None) if baseline_correct
                       else np.clip(y, 0, None))
    return out


def _mean_spectrum(cube, mask, baseline_correct=True):
    cube = np.asarray(cube, float)[:, mask]
    from sers_mixture import als_baseline, desaturate
    cube, _sat = desaturate(cube)      # bridge clipped plateaus before any ALS
    ok = _sat <= 0.01                  # quarantine: clipped pixels don't train
    if ok.any() and not ok.all():
        cube = cube[ok]
    w = cube.sum(1); w = w / (w.sum() + 1e-12)
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

    The concentration inverse sees the local composition-head ratio and signal size,
    plus P10/median/P90 summaries of those quantities over the same experimental map.
    It therefore learns the applied-concentration relation without memorising thousands
    of wavelength channels from only a few dozen maps.
    """
    X = np.clip(np.asarray(X, float), 0, None)
    if ratios is None:
        raise ValueError("composition ratios are required for concentration context")
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


# --------------------------------------------------------------------------
# Calibration-residual µM head ("calibration_residual_uM_v1")
#
# The integrated 20260824 re-evaluation compared three concentration routes on the
# same absolute-condition-grouped 5-fold split (composition head AND corrector refit
# per fold): pure single-band calibration inversion, a Ridge Δlog10 corrector and a
# small neural Δlog10 corrector on top of the inversion. On the 64 calibration-range
# maps the neural corrector won every headline number (MAE 8.81→5.01 µM, within-2×
# 29→78%), so when a calibration is available this head replaces the direct
# log10-µM pixel head below. Construction (documentation/scripts/
# run_integrated_reevaluation.py, results/integrated_final_20260824):
#
#   per-substance marker band  →  composition-probability-weighted band signal Ieq
#   →  log-linear inversion  Ccal = 10^((Ieq − b)/a)  clipped to the calibration range
#   →  12 map-level features [log10 Ccal, ratio, log1p Ieq, intensity P10/50/90]
#   →  128→32 net  →  Δlog10(C)   →   C = Ccal · 10^Δ  (Δ clipped to ±2 decades)
# --------------------------------------------------------------------------
_MARKER_BANDS_CM = {"DQ": 1570.0, "TBZ": 1270.0, "THI": 1367.0}
_BAND_HALF_WIDTH_CM = 10.0
_DELTA_CLIP_DECADES = 2.0


def _marker_bands(conc_subs, wn_axis, P=None):
    """One characteristic band (cm⁻¹) per substance: the known VIP marker for the
    DQ/TBZ/THI trio, else the strongest channel of that substance's unit template."""
    wn_axis = np.asarray(wn_axis, float)
    bands = []
    for j, s in enumerate(conc_subs):
        b = next((v for k, v in _MARKER_BANDS_CM.items()
                  if str(s).upper().startswith(k)), None)
        if b is None:
            b = float(wn_axis[int(np.argmax(P[j]))]) if P is not None else float(np.median(wn_axis))
        bands.append(float(b))
    return np.asarray(bands, float)


def _band_signal(raw, wn_axis, bands, half_width=_BAND_HALF_WIDTH_CM):
    """Local band maxima: (n_px, n_feat) raw intensities → (n_px, n_bands). The maximum
    inside ±half_width tolerates the instrument grid and small peak drift."""
    raw = np.clip(np.atleast_2d(np.asarray(raw, float)), 0, None)
    wn_axis = np.asarray(wn_axis, float)
    out = np.zeros((len(raw), len(bands)))
    for j, centre in enumerate(np.asarray(bands, float)):
        m = np.abs(wn_axis - centre) <= float(half_width)
        if not m.any():
            m = np.zeros(raw.shape[1], bool)
            m[int(np.argmin(np.abs(wn_axis - centre)))] = True
        out[:, j] = raw[:, m].max(axis=1)
    return out


def _fit_loglinear_ab(calib_path, conc_subs, bands, lo, hi):
    """Per-substance log-linear calibration Ieq = a·log10(C µM) + b, fitted on the SAME
    band extraction the head applies to maps (so fit and inversion share one
    convention — a per-substance offset in the convention is absorbed by Δlog10).
    Returns (ab (n,2), calibrated range in µM (n,2)); rows are NaN when a substance is
    missing from the CSV or its slope comes out non-positive."""
    from io_utils import load_calibration_csv
    ax_c, names_c, dils = load_calibration_csv(calib_path)
    ax_c = np.asarray(ax_c, float)
    m = (ax_c >= lo) & (ax_c <= hi)
    if m.sum() < 10:
        m = np.ones(len(ax_c), bool)
    ab = np.full((len(conc_subs), 2), np.nan)
    rng_uM = np.full((len(conc_subs), 2), np.nan)
    for j, s in enumerate(conc_subs):
        if s not in names_c:
            continue
        Cser, spec = dils[names_c.index(s)]
        sig = _band_signal(np.asarray(spec, float)[:, m], ax_c[m], [bands[j]])[:, 0]
        c_uM = np.asarray(Cser, float) * 1e6
        ok = (c_uM > 0) & (sig > 0)
        if ok.sum() < 2:
            continue
        a, b = np.polyfit(np.log10(c_uM[ok]), sig[ok], 1)
        if a <= 0:                      # inversion needs signal growing with C
            continue
        ab[j] = [float(a), float(b)]
        rng_uM[j] = [float(c_uM[ok].min()), float(c_uM[ok].max())]
    return ab, rng_uM


def _invert_calibration(band_sig, ab, cal_rng_uM):
    """Band signal → Ccal (µM), clipped to each substance's calibrated range."""
    band_sig = np.atleast_2d(np.asarray(band_sig, float))
    lo_c = np.where(np.isfinite(cal_rng_uM[:, 0]), cal_rng_uM[:, 0], 0.1)
    hi_c = np.where(np.isfinite(cal_rng_uM[:, 1]), cal_rng_uM[:, 1], 100.0)
    return np.clip(10.0 ** ((band_sig - ab[:, 1]) / ab[:, 0]), lo_c, hi_c)


def _residual_context(raw_px, ratios_px, map_keys, wn_axis, bands, ab, cal_rng_uM):
    """Map-level feature rows for the residual corrector.

    Returns (map_names, F (n_maps, 3n+3), Ccal_uM (n_maps, n)). Each pixel contributes
    to Ieq in proportion to its composition probability for that component, so hotspot
    pixels of a component dominate its band readout."""
    raw_px = np.clip(np.asarray(raw_px, float), 0, None)
    R = np.clip(np.asarray(ratios_px, float), 0, None)
    keys = np.asarray(map_keys, object)
    band_px = _band_signal(raw_px, wn_axis, bands)
    log_total = np.log1p(raw_px.sum(axis=1))
    names, F, CC = [], [], []
    for g in dict.fromkeys(keys.tolist()):
        idx = np.where(keys == g)[0]
        p = R[idx]
        ratio = p.mean(0); ratio = ratio / (ratio.sum() + 1e-12)
        w = p + 1e-6
        ieq = (w * band_px[idx]).sum(0) / w.sum(0)
        ccal = _invert_calibration(ieq, ab, cal_rng_uM)[0]
        iq = np.percentile(log_total[idx], [10, 50, 90])
        F.append(np.concatenate([np.log10(ccal), ratio, np.log1p(ieq), iq]))
        names.append(g); CC.append(ccal)
    return names, np.asarray(F, float), np.asarray(CC, float)


def _residual_net_torch(n_in, n_out, seed):
    import torch, torch.nn as nn
    torch.manual_seed(int(seed))
    return nn.Sequential(nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(),
                         nn.Dropout(0.25), nn.Linear(128, 32), nn.ReLU(),
                         nn.Linear(32, n_out))


def _fit_residual_net(F, Ccal_uM, T_uM, cond_keys, seed=0, max_epochs=800):
    """Masked-Huber Δlog10 fit. The epoch budget is selected on a condition-grouped 20%
    holdout, then the net is refit on every map for that many epochs (the same recipe
    the validated run used). Absent components (true 0) are masked out of the loss.
    Returns (net, mu, sd, best_epoch, best_val, loss_curve)."""
    import torch
    F = np.asarray(F, np.float32); Ccal = np.asarray(Ccal_uM, float); T = np.asarray(T_uM, float)
    target = (np.log10(np.clip(T, 0.05, None))
              - np.log10(np.clip(Ccal, 0.05, None))).astype(np.float32)
    mask = (T > 0).astype(np.float32)

    def masked_huber(pred, tgt, msk):
        loss = torch.nn.functional.smooth_l1_loss(pred, tgt, reduction="none", beta=0.25)
        return (loss * msk).sum() / msk.sum().clamp_min(1.0)

    tr, va = _group_validation_indices(np.asarray(cond_keys, object), seed=seed)
    best_ep, best = min(300, int(max_epochs)), None
    if len(va):
        mu = F[tr].mean(0); sd = F[tr].std(0) + 1e-6
        A = torch.tensor(((F - mu) / sd).astype(np.float32))
        Y = torch.tensor(target); M = torch.tensor(mask)
        net = _residual_net_torch(F.shape[1], T.shape[1], seed)
        op = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=3e-3)
        best = float("inf"); stale = 0
        for ep in range(int(max_epochs)):
            net.train(); op.zero_grad()
            l = masked_huber(net(A[tr]), Y[tr], M[tr]); l.backward(); op.step()
            net.eval()
            with torch.no_grad():
                v = float(masked_huber(net(A[va]), Y[va], M[va]))
            if v < best - 1e-4:
                best, best_ep, stale = v, ep + 1, 0
            else:
                stale += 1
            if stale >= 80 and ep >= 100:
                break
    mu = F.mean(0); sd = F.std(0) + 1e-6
    A = torch.tensor(((F - mu) / sd).astype(np.float32))
    Y = torch.tensor(target); M = torch.tensor(mask)
    net = _residual_net_torch(F.shape[1], T.shape[1], seed + 1000)
    op = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=3e-3)
    hist = []
    for _ in range(max(5, int(best_ep))):
        net.train(); op.zero_grad()
        l = masked_huber(net(A), Y, M); l.backward(); op.step()
        hist.append(float(l.detach()))
    net.eval()
    return net, mu, sd, int(best_ep), best, hist


def _predict_residual_uM(net, mu, sd, F, Ccal_uM):
    """Features + calibration inversion → corrected µM (Δ clipped to ±2 decades)."""
    import torch
    A = torch.tensor(((np.asarray(F, np.float32) - mu) / sd).astype(np.float32))
    net.eval()
    with torch.no_grad():
        delta = net(A).numpy()
    delta = np.clip(delta, -_DELTA_CLIP_DECADES, _DELTA_CLIP_DECADES)
    return np.clip(np.asarray(Ccal_uM, float) * (10.0 ** delta), 1e-3, 5e3)


def _train_calibration_residual_head(Xraw, Rabs, gp, gcond, C_uM_rows, conc_subs,
                                     wn_axis, P, calib_path, lo, hi, *, seed=0,
                                     loo=False, progress=None):
    """Train the calibration-residual µM head on the hit pixels train_model collected.
    Returns the portable uM dict, or None when no usable calibration line exists."""
    bands = _marker_bands(conc_subs, wn_axis, P)
    ab, cal_rng = _fit_loglinear_ab(calib_path, conc_subs, bands, lo, hi)
    if not np.isfinite(ab).all():
        return None
    if progress:
        progress("training concentration head (calibration residual)")
    map_names, F, Ccal = _residual_context(Xraw, Rabs, gp, wn_axis, bands, ab, cal_rng)
    gp = np.asarray(gp, object); gcond = np.asarray(gcond, object)
    first_px = {g: np.where(gp == g)[0][0] for g in map_names}
    T = np.stack([np.asarray(C_uM_rows[first_px[g]], float) for g in map_names])
    cond_of_map = np.array([gcond[first_px[g]] for g in map_names], object)
    net, mu, sd, best_ep, best_val, hist = _fit_residual_net(
        F, Ccal, T, cond_of_map, seed=seed)
    uM = {"kind": "calibration_residual_uM_v1",
          "state": {k: v.detach().numpy() for k, v in net.state_dict().items()},
          "mu": mu, "sd": sd, "subs": list(conc_subs), "pool": "map",
          "hidden": (128, 32), "input_mode": "comp_ratio+band_loglinear_residual_v1",
          "bands_cm": bands, "band_half_width_cm": _BAND_HALF_WIDTH_CM,
          "loglinear_ab": ab, "cal_range_uM": cal_rng,
          "ab_source": "log-linear fit of the embedded calibration CSV at the marker bands",
          "delta_clip_decades": _DELTA_CLIP_DECADES,
          "ranges_M": np.asarray(cal_rng, float) * 1e-6,
          "selected_epochs": best_ep, "selection_val_loss": best_val,
          "selection_level": "condition", "train_loss": hist,
          "n_maps": len(map_names), "n_pixels": int(len(Xraw)),
          "protocol": "integrated_v1_20260824 (abs-condition-grouped held-out validated)"}

    # ---- the VALIDATED window, same meaning as the pixel head's ---------------
    # 20% condition-grouped holdout: a level counts as validated for a substance when
    # its held-out maps recover within 2-fold. Cheap — the head sees one row per map.
    try:
        mtr, mva = _group_validation_indices(cond_of_map, seed=seed + 31)
        if len(mva) and len(mtr) >= 3:
            if progress:
                progress("validating the reportable window (20% held-out)")
            vnet, vmu, vsd, _, _, _ = _fit_residual_net(
                F[mtr], Ccal[mtr], T[mtr], cond_of_map[mtr], seed=seed + 31)
            vpred = _predict_residual_uM(vnet, vmu, vsd, F[mva], Ccal[mva])
            lv_err = {}
            for i, row in zip(mva, vpred):
                for j in range(len(conc_subs)):
                    t = float(T[i][j])
                    if t > 0 and np.isfinite(row[j]) and row[j] > 0:
                        lv_err.setdefault((j, round(t, 4)), []).append(
                            abs(np.log10(row[j] / t)))
            vr = np.full((len(conc_subs), 2), np.nan)
            for j in range(len(conc_subs)):
                good = [lvl for (jj, lvl), es in lv_err.items()
                        if jj == j and 10.0 ** float(np.median(es)) <= 2.0]
                if len(good) >= 2:
                    vr[j] = [min(good) * 1e-6, max(good) * 1e-6]
            uM["validated_ranges_M"] = vr
            uM["validated_note"] = ("levels recovered within 2-fold on a 20% "
                                    "condition-grouped holdout")
    except Exception as _e:
        if progress:
            progress(f"validated-window check skipped ({_e})")

    # ---- honest per-condition scoring, same boundary as composition -----------
    # The residual net is refit per held-out absolute condition. Ratios/Ieq come from
    # the deployed composition head (refitting IT per fold is the offline benchmark's
    # job) — noted in the dict so the number is never sold as the fully-nested one.
    if loo:
        loo_true, loo_pred, loo_paths = [], [], []
        uniq = list(dict.fromkeys(cond_of_map.tolist()))
        for fold_i, held in enumerate(uniq):
            te = np.where(cond_of_map == held)[0]
            tr = np.where(cond_of_map != held)[0]
            if not len(te) or len(tr) < 3:
                continue
            if progress:
                progress(f"concentration leave-one-condition-out {fold_i + 1}/{len(uniq)}")
            fnet, fmu, fsd, _, _, _ = _fit_residual_net(
                F[tr], Ccal[tr], T[tr], cond_of_map[tr], seed=seed + 1000 + fold_i)
            fpred = _predict_residual_uM(fnet, fmu, fsd, F[te], Ccal[te])
            for i, row in zip(te, fpred):
                loo_true.append(T[i].tolist())
                loo_pred.append(np.asarray(row, float).tolist())
                loo_paths.append(map_names[i])
        uM["loo_eval"] = {"true_uM": loo_true, "pred_uM": loo_pred, "paths": loo_paths,
                          "level": "condition",
                          "ratio_source": "deployed composition head (not refit per fold)"}
    return uM


def _apply_calibration_residual(model, wn, spectra, return_meta):
    """Apply path for the calibration-residual head. The batch is treated as ONE map
    (the same contract the pixel head's context features rely on). Returns per-pixel
    µM whose per-component median equals the map-level corrected estimate, so every
    downstream median-pool reports exactly the validated construction."""
    u = model["uM"]
    usubs = list(u.get("subs") or model["subs"])
    wn = np.asarray(wn, float)
    mask = (wn >= model["lo"]) & (wn <= model["hi"])
    X = np.asarray(spectra, float)
    if X.shape[1] == len(wn):
        X = X[:, mask]
    X = np.clip(X, 0, None)
    wn_axis = wn[mask] if mask.sum() == X.shape[1] else wn
    pk = np.clip(np.asarray(apply_model_pixels(model, wn, spectra), float), 0, None)
    subs_all = list(model["subs"])
    cols = [subs_all.index(s_) for s_ in usubs if s_ in subs_all]
    analyte_mass = pk[:, cols].sum(axis=1)
    R = pk[:, cols] / (pk[:, cols].sum(axis=1, keepdims=True) + 1e-12)
    # Map context from analyte-dominant pixels only: Real hands EVERY pixel through
    # here (background included), while training saw hit-screened pixels. The model's
    # own blank channel is the closest stand-in for that gate; without a blank class
    # every pixel passes, which matches the all-hit training sets.
    sel = np.where(analyte_mass >= 0.5)[0]
    if not len(sel):
        sel = np.arange(len(X))
    ab = np.asarray(u["loglinear_ab"], float)
    cal_rng = np.asarray(u["cal_range_uM"], float)
    bands = np.asarray(u["bands_cm"], float)
    import torch
    net = _residual_net_torch(len(u["mu"]), len(usubs), 0)
    net.load_state_dict({k: torch.tensor(v) for k, v in u["state"].items()})
    _, F, Ccal = _residual_context(X[sel], R[sel], np.zeros(len(sel), int),
                                   wn_axis, bands, ab, cal_rng)
    c_map = _predict_residual_uM(net, u["mu"], u["sd"], F, Ccal)[0]   # (n_subs,) µM
    # Per-pixel display: each pixel's own calibration inversion, rescaled per component
    # so the median equals the map estimate — the spatial pattern is the pixels', the
    # reported number is the validated map-level one.
    ccal_px = _invert_calibration(_band_signal(X, wn_axis, bands), ab, cal_rng)
    med = np.median(ccal_px, axis=0)
    um = ccal_px * (c_map / np.where(med > 0, med, 1.0))[None, :]
    um = np.clip(um, 1e-3, 5e3)
    result = (um, usubs)
    if not return_meta:
        return result
    rngs_out = np.asarray(u.get("ranges_M"), float).copy()
    vr = u.get("validated_ranges_M")
    if vr is not None:
        vr = np.asarray(vr, float)
        ok_ = np.isfinite(vr).all(axis=1)
        rngs_out[ok_] = vr[ok_]                  # validated window beats label range
    r_um = rngs_out * 1e6
    component_ood = np.zeros_like(um, bool)
    for k in range(min(len(usubs), len(r_um))):
        if np.isfinite(r_um[k][1]) and r_um[k][1] > 0:
            component_ood[:, k] = um[:, k] > r_um[k][1]
    meta = {"feature_ood": np.zeros(len(um), bool),
            "component_ood": component_ood, "ranges_M": rngs_out,
            "map_uM": {s_: float(c_map[j]) for j, s_ in enumerate(usubs)}}
    return (*result, meta)


def train_model(data_dir, items, calib_path=None, baseline=True, trim=None, progress=None,
                method="mlp", epochs=300, seed=0, use_pretrain=True, epoch_diagnostics=False,
                n_components=8, n_trees=300, loo=False, test_items=None, px_per_map=0,
                include_blank=False, noise_aug=0, nnls_screen=True,
                screen_min_frac=0.15, equal_volume_mix=False, sim_nuisance=True,
                sim_iso=None, pixel_sampling="legacy", sampling_seed=0):
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
        for j, ya in enumerate(_map_spectra(
                cube, local_mask, px_per_map, baseline_correct=not nnls_screen,
                sampling=pixel_sampling, sampling_seed=sampling_seed,
                map_id=it[0])):
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
    # The PURE references are the vertices of the composition simplex, and until now the
    # learned heads never saw them: `_refs` folds them into the NNLS template matrix P,
    # which only the classical methods read. The mixtures alone cover the middle of the
    # simplex — nothing sits at a true (1,0,0) — so held-out predictions came back shrunk
    # toward the centroid (slope 0.48-0.82) with a +0.13 intercept: a genuinely absent
    # compound was reported at ~13%. It costs DQ and TBZ the most, whose unit templates
    # have cosine 0.68 — the pure maps are the examples that separate them (they do have
    # exclusive bands: DQ 1167-1184 / 1375-1381, TBZ 626-641 / 753-756 / 1246-1273 cm-1).
    # Feeding them in also puts the MLP on the same footing as NNLS/MCR-ALS, which are
    # handed those same spectra as templates.
    #
    # Only the composition head is affected: pure maps carry no mixture µM label, so they
    # are not appended to Xabs/Cabs and the concentration head is untouched.
    from dataset import discover_references, is_blank, base_and_batch, load_manifest
    manifest = load_manifest(data_dir) or {}
    for c_, pth in discover_references(data_dir):
        name = base_and_batch(c_)[0]
        if is_blank(name) or name not in subs:
            continue
        # Honour samples.csv: '-3' is marked test and stays out, so it remains a genuine
        # external check. (The blank branch below still ignores these roles — see TODO.)
        if manifest.get(os.path.abspath(pth), (None, None, "train"))[2] != "train":
            continue
        try:
            if nnls_screen:
                _w, cube, smeta = nnls_hit_spectra(
                    data_dir, pth, baseline=baseline, trim=trim,
                    min_frac=screen_min_frac, progress=None)
                local_mask = np.ones(cube.shape[1], bool)
                screen_stats.append({"path": pth, **smeta})
            else:
                _w, cube, _m, _c = load_map(pth); local_mask = mask
        except Exception:
            continue
        vec = _ratio([1.0 if s == name else 0.0 for s in subs])
        # Pure references keep the MAP as their split group, like the blanks below: they
        # are reference material rather than a test condition, and grouping all of one
        # compound together would leave that fold with no vertex at all — the very thing
        # this block exists to supply. No mixture condition shares this key, so nothing
        # held out can be recited back.
        if progress:
            progress(f"adding pure reference {os.path.basename(pth)}")
        for j, ya in enumerate(_map_spectra(
                cube, local_mask, px_per_map, baseline_correct=not nnls_screen,
                sampling=pixel_sampling, sampling_seed=sampling_seed,
                map_id=pth)):
            X.append(_composition_features(ya)); Y.append(vec)
            paths.append(pth); conds.append(pth)
            for yn in (_noisy_copies(ya, noise_aug, aug_rng) if (noise_aug and j) else ()):
                X.append(_composition_features(yn)); Y.append(vec)
                paths.append(pth); conds.append(pth)

    # Optionally learn BACKGROUND as its own class: pixels from the blank reference map
    # labelled "100% blank". Without it the model must spread every spectrum — even bare
    # substrate — across the substances, so it can never say "nothing here".
    # TODO: this branch ignores samples.csv roles — BLK-1 is marked 'exclude' and BLK-4 /
    # INK-3 'test', yet all of them train. Left as-is so this retrain changes exactly one
    # thing; worth fixing separately.
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
                for ya in _map_spectra(
                        cube, mask, px_per_map or 60, spread=True,
                        sampling=pixel_sampling, sampling_seed=sampling_seed, map_id=pth):
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
    comp = None                       # torch composition head (mlp branch only)
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
    elif method in ("cnn", "mlp"):
        # The filename label belongs to the MAP. Train on its pixel distribution, but
        # optimise the mean prediction of each map; this is the same estimator used by
        # Benchmark and prevents 400 correlated pixels becoming 400 fake labels.
        tp, comp = _fit_torch_bag(method, X, Y, paths, X, pre=pre, epochs=epochs,
                                  seed=seed, return_net=True, progress=progress)
        comp_store = {"method": method,
                      "comp_state": {k: v.detach().cpu().numpy()
                                     for k, v in comp.state_dict().items()},
                      "selected_epochs": int(epochs), "epoch_rule": "fixed-map-pooled",
                      "selection_level": "map"}
        if method == "mlp":
            comp_store["comp_hidden"] = (256, 64)
    else:
        raise ValueError(f"unknown composition method: {method}")
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
                                 baseline_correct=not nnls_screen,
                                 sampling=pixel_sampling, sampling_seed=sampling_seed,
                                 map_id=it[0])
            truth_by_map[it[0]] = vec
            for ya in specs:
                Xt_.append(_composition_features(ya))
                row_paths.append(it[0])
        if Xt_:
            Pt = _fit_predict(method, X, Y, np.array(Xt_, np.float32), pre=pre,
                              epochs=epochs, seed=seed, n_components=n_components,
                              n_trees=n_trees, P_ref=P, train_maps=paths)
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
                                n_trees=n_trees, P_ref=P, train_maps=paths[tr])
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
            progress("preparing concentration inputs")
        hv = np.array(have)
        C = np.array([list(Cabs[i])[:len(conc_subs)] if Cabs[i] is not None
                      else [0.0] * len(conc_subs) for i in range(len(Xabs))], float)
        # `gp` stays the MAP — it is the POOLING key (the label is one number per map, so
        # the loss compares the per-map median) and the context-feature key. `gcond` is the
        # SPLITTING key. Conflating the two would pool repeats of one condition into a
        # single pseudo-map and change what the head is trained to predict.
        gp = np.asarray(abs_paths, object)[hv]
        gcond = np.asarray(abs_conds, object)[hv]
        # ratio input from the trained COMPOSITION HEAD (full spectrum), not the
        # template projection: the projection reads any strong ~1580 band as DQ,
        # and the µM head inherited that single-band misreading. Falls back to
        # the projection for non-torch composition heads.
        uM_input_mode = "nnls_ratio+intensity_quantiles_v1"
        Rabs = None
        if method in ("mlp", "cnn") and comp is not None:
            feats_abs = np.stack([_composition_features(y) for y in Xabs[hv]]).astype(np.float32)
            comp.eval()
            with torch.no_grad():
                pk = torch.softmax(comp(torch.tensor(feats_abs)), 1).numpy()
            cols = [subs.index(s_) for s_ in conc_subs]
            R_ = np.clip(pk[:, cols], 0, None)
            Rabs = R_ / (R_.sum(axis=1, keepdims=True) + 1e-12)
            uM_input_mode = f"{method}_ratio+intensity_quantiles_v1"
        if Rabs is None:
            from dl_quantify import surface_composition
            Rabs = surface_composition(_composition_features(Xabs[hv], 'legacy_l2'), P)
        # ---- calibration available → the validated residual corrector wins -------
        # Integrated 20260824 benchmark (abs-condition-grouped 5-fold, everything
        # refit per fold): MAE 8.81→5.01 µM, within-2× 29→78% over the pure
        # inversion on the calibration-range maps. The direct pixel head below stays
        # as the no-calibration fallback; any failure here falls through to it.
        if calib_path:
            try:
                uM = _train_calibration_residual_head(
                    Xabs[hv], Rabs, gp, gcond, C[hv] * 1e6, conc_subs,
                    wn[mask], P, calib_path, lo, hi, seed=seed, loo=loo,
                    progress=progress)
                if uM is not None:
                    uM["ratio_head"] = uM_input_mode.split("_ratio")[0]
            except Exception as _e:
                if progress:
                    progress(f"calibration-residual µM head failed ({_e}) — "
                             "falling back to the pixel head")
                uM = None
        if uM is None:
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
                  "input_mode": uM_input_mode, "hidden": (128, 32),
                  "raw_n_feat": Xabs.shape[1], "ratio_n_feat": len(conc_subs),
                  "ood_threshold": float(np.quantile(dist, 0.99)),
                  "ranges_M": np.asarray(ranges, float), "train_loss": loss_curve_uM,
                  "selected_epochs": selected_uM_epochs,
                  "selection_val_loss": (best_val if selection_uM_val else None),
                  "selection_val_history": selection_uM_val,
                  "selection_level": "condition",
                  "n_maps": len(dict.fromkeys(gp.tolist())), "n_pixels": len(hv)}

            # ---- the VALIDATED window, derived from the file itself -----------------
            # One condition-grouped 20% holdout of the µM head (cheap - one extra fit,
            # not the ~100x LOO): a concentration level counts as validated for a
            # substance when its held-out maps recover within 2-fold at that level.
            # Real reports inside this window automatically; nothing is typed by hand.
            try:
                v_tr, v_va = _group_validation_indices(gcond, seed=seed + 31)
                if len(v_va) and len(v_tr) >= 3:
                    if progress:
                        progress("validating the reportable window (20% held-out)")
                    vtrain = _concentration_context_features(Xabs[hv][v_tr], gp[v_tr], Rabs[v_tr])
                    vmu_ = vtrain.mean(0); vsd_ = vtrain.std(0) + 1e-8
                    vX = torch.tensor(((vtrain - vmu_) / vsd_).astype(np.float32))
                    vY = torch.tensor((np.log10(np.clip(C[hv][v_tr], 1e-8, None)) + 6.0
                                       ).astype(np.float32))
                    torch.manual_seed(seed + 31); vnet = make_conc_net()
                    vop = torch.optim.Adam(vnet.parameters(), lr=1e-3, weight_decay=3e-3)
                    for _ in range(selected_uM_epochs):
                        vnet.train(); vop.zero_grad()
                        vl = pooled_loss(vnet(vX), vY, gp[v_tr]); vl.backward(); vop.step()
                    vnet.eval()
                    vtest = _concentration_context_features(Xabs[hv][v_va], ratios=Rabs[v_va])
                    with torch.no_grad():
                        vlog = vnet(torch.tensor(((vtest - vmu_) / vsd_
                                                  ).astype(np.float32))).numpy()
                    vgp = gp[v_va]
                    lv_err = {}                       # (subs j, level µM) -> [fold errors]
                    for mp in dict.fromkeys(vgp.tolist()):
                        sel = np.where(vgp == mp)[0]
                        pred = 10.0 ** np.clip(np.median(vlog[sel], axis=0), -3.0, 6.0)
                        true = C[hv][v_va[sel[0]]] * 1e6
                        for j in range(len(conc_subs)):
                            if true[j] > 0 and np.isfinite(pred[j]) and pred[j] > 0:
                                lv_err.setdefault((j, round(float(true[j]), 4)),
                                                  []).append(abs(np.log10(pred[j] / true[j])))
                    vr = np.full((len(conc_subs), 2), np.nan)
                    for j in range(len(conc_subs)):
                        good = [lvl for (jj, lvl), es in lv_err.items()
                                if jj == j and 10.0 ** float(np.median(es)) <= 2.0]
                        if len(good) >= 2:
                            vr[j] = [min(good) * 1e-6, max(good) * 1e-6]
                    uM["validated_ranges_M"] = vr
                    uM["validated_note"] = ("levels recovered within 2-fold on a 20% "
                                            "condition-grouped holdout")
                    if progress:
                        _txt = " · ".join(
                            f"{conc_subs[j]} {vr[j][0]*1e6:.3g}-{vr[j][1]*1e6:.3g}uM"
                            if np.isfinite(vr[j]).all() else f"{conc_subs[j]} n/a"
                            for j in range(len(conc_subs)))
                        progress("validated window: " + _txt)
            except Exception as _e:
                if progress:
                    progress(f"validated-window check skipped ({_e})")

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

    # carry the calibration INSIDE the model: Real then quantifies µM from the
    # .dlm alone — no separate CSV to re-browse (and no way to pair the wrong one)
    calib_csv_text = calib_csv_name = None
    if calib_path:
        try:
            with open(calib_path, "r", encoding="utf-8", errors="replace") as fh:
                calib_csv_text = fh.read()
            calib_csv_name = os.path.basename(str(calib_path))
        except OSError:
            pass

    return {"subs": subs, "lo": lo, "hi": hi, "n_feat": int(mask.sum()), "P": P,
            "feature_mode": "log1p_raw",
            "calib_csv_text": calib_csv_text, "calib_csv_name": calib_csv_name,
            "uM": uM, "n_train": int(len(X)), "has_uM": uM is not None,
            "n_maps": int(len(set(paths.tolist()))),
            "px_per_map": int(px_per_map),
            "pixel_sampling": str(pixel_sampling), "sampling_seed": int(sampling_seed),
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


def _grouped_map_folds(group_keys, map_keys, n_folds=5, seed=0):
    """Assign whole composition groups to folds while balancing the number of maps."""
    g = np.asarray(group_keys, object); m = np.asarray(map_keys, object)
    groups = list(dict.fromkeys(g.tolist()))
    if len(groups) < 2:
        raise ValueError("benchmark needs at least two distinct composition ratios")
    n_folds = min(max(2, int(n_folds)), len(groups))
    counts = {key: len(set(m[g == key].tolist())) for key in groups}
    rng = np.random.default_rng(int(seed))
    ordered = [groups[i] for i in rng.permutation(len(groups))]
    ordered.sort(key=lambda key: counts[key], reverse=True)  # random tie order stays stable
    bins = [set() for _ in range(n_folds)]; loads = [0] * n_folds
    for key in ordered:
        j = int(np.argmin(loads)); bins[j].add(key); loads[j] += counts[key]
    return bins


def _pool_predictions_by_map(pred, true, map_keys, names):
    """Pool every pixel prediction once, returning one auditable row per held-out map."""
    pred = np.asarray(pred, float); true = np.asarray(true, float)
    maps = np.asarray(map_keys, object); names = np.asarray(names, object)
    tv, pv, out_names, n_pixels = [], [], [], []
    for mk in dict.fromkeys(maps.tolist()):
        sel = maps == mk; first = int(np.where(sel)[0][0])
        tv.append(true[first].tolist()); pv.append(pred[sel].mean(0).tolist())
        out_names.append(str(names[first])); n_pixels.append(int(sel.sum()))
    return tv, pv, out_names, n_pixels

def _pool_concentration_by_map(pred_uM, true_uM, map_keys, names):
    """Median-pool pixel µM predictions to one auditable held-out row per map."""
    pred_uM = np.asarray(pred_uM, float); true_uM = np.asarray(true_uM, float)
    maps = np.asarray(map_keys, object); names = np.asarray(names, object)
    tv, pv, out_names, n_pixels = [], [], [], []
    for mk in dict.fromkeys(maps.tolist()):
        sel = maps == mk; first = int(np.where(sel)[0][0])
        tv.append(true_uM[first].tolist())
        pv.append(np.median(pred_uM[sel], axis=0).tolist())
        out_names.append(str(names[first])); n_pixels.append(int(sel.sum()))
    return tv, pv, out_names, n_pixels


def _fit_concentration_fold(Xtr, Ctr_uM, train_maps, Xte, epochs=100, seed=0):
    """Fit the dedicated map-pooled µM head and predict held-out pixels."""
    import torch
    import torch.nn as nn
    Xtr = np.asarray(Xtr, np.float32); Xte = np.asarray(Xte, np.float32)
    Ctr_uM = np.asarray(Ctr_uM, np.float32)
    train_maps = np.asarray(train_maps, object)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    A = torch.tensor(((Xtr - mu) / sd).astype(np.float32))
    B = torch.tensor(((Xte - mu) / sd).astype(np.float32))
    Y = torch.tensor(np.log10(np.clip(Ctr_uM, 0.01, None)).astype(np.float32))
    torch.manual_seed(int(seed))
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.BatchNorm1d(128), nn.ReLU(),
                        nn.Dropout(0.25), nn.Linear(128, 32), nn.ReLU(),
                        nn.Linear(32, Ctr_uM.shape[1]))
    groups = [np.where(train_maps == mp)[0]
              for mp in dict.fromkeys(train_maps.tolist())]
    op = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=3e-3)
    for _ in range(int(epochs)):
        net.train(); op.zero_grad(); pred = net(A); losses = []
        for idx in groups:
            ii = torch.as_tensor(idx, dtype=torch.long)
            losses.append(torch.nn.functional.smooth_l1_loss(
                torch.quantile(pred[ii], 0.5, dim=0), Y[ii[0]], beta=0.25))
        loss = torch.stack(losses).mean()
        loss = loss + 0.05 * torch.nn.functional.smooth_l1_loss(pred, Y, beta=0.5)
        loss.backward(); op.step()
    net.eval()
    with torch.no_grad():
        log_uM = net(B).numpy()
    return 10.0 ** np.clip(log_uM, -2.0, 6.0)


def _fit_mlp_ratio_fold(Xtr, Ytr, train_maps, Xte, *, epochs=100, seed=0, pre=None):
    """Fit the composition MLP inside one outer fold and return train/test ratios.

    The concentration head in a saved MLP model consumes ratios predicted by that
    model's composition MLP, not NNLS ratios. Benchmarking must reproduce the same
    dependency without letting held-out maps participate in fitting the composition
    head. Training-set ratios are deliberately in-sample, matching train_model;
    test ratios come from the same fitted head but are fully held out.
    """
    Xtr = np.asarray(Xtr, np.float32)
    Xte = np.asarray(Xte, np.float32)
    if not len(Xtr) or not len(Xte):
        raise ValueError("composition-ratio fold needs non-empty train and test rows")
    both = np.vstack([Xtr, Xte])
    pred = _fit_torch_bag("mlp", Xtr, Ytr, train_maps, both, pre=pre,
                          epochs=epochs, seed=seed)
    pred = np.clip(np.asarray(pred, float), 0, None)
    pred /= pred.sum(axis=1, keepdims=True) + 1e-12
    return pred[:len(Xtr)], pred[len(Xtr):]


def benchmark_loo(data_dir, items, calib_path=None, baseline=True, trim=None, progress=None,
                  methods=("null", "band", "nnls", "nnls_rf", "mcr", "pls", "rf",
                           "cnn", "mlp"), epochs=350, seed=0,
                  use_pretrain=True,
                  n_components=8, n_trees=300, px_per_map=400, rf_max_features=None,
                  cnn_epochs=None, band_window=10.0, mcr_iter=6, cv_folds=5,
                  nnls_screen=True, screen_min_frac=0.15, equal_volume_mix=False,
                  pixel_sampling="legacy", sampling_seed=0):
    """Condition-grouped cross-validation using the pixel distribution of each map.

    All pixels from a map stay in one fold. Methods predict held-out pixels, then every
    map is scored once after mean-pooling its pixel predictions. ``cv_folds=None`` keeps
    the legacy leave-one-ratio-group-out protocol for reproducibility.

    The ladder, cheapest rung first — the four training-free ones cost almost nothing and
    each answers a different objection:
      null     always the mean training composition — the floor every method must clear
      band     NNLS on each compound's least-cross-talk marker windows (± ``band_window``
               cm⁻¹), the Validate tab's decomposition
      nnls     NNLS on the whole spectrum, the classical baseline
      nnls_rf  NNLS + one response factor per substance, fitted on the training fold —
               the control that asks how much of the learned gain is just that
      mcr      MCR-ALS: refine the component spectra from the data, then decompose —
               the answer to "your templates are not the real surface spectra"
    then pls / rf / cnn / mlp, which are refit per fold."""
    from real_data import load_map
    from dl_quantify import simulate_mixtures, _ratio
    subs, wn, mask, P, lo, hi = _refs(data_dir, baseline, trim)
    X, Y, Cabs, gkey, abskey, ekey, mapkey = [], [], [], [], [], [], []
    for k, it in enumerate(items):
        if progress:
            progress(f"loading maps {k + 1}/{len(items)}")
        vec = _ratio([float(it[1].get(s, 0.0)) for s in subs])
        if vec.sum() <= 0:
            continue
        conc = it[2] if len(it) > 2 else None
        if conc:
            stock_uM = [float(conc.get(s, 0.0)) * 1e6 for s in subs]
            dilution = (max(1, sum(v > 0 for v in stock_uM))
                        if equal_volume_mix else 1)
            cvec_uM = np.asarray([v / dilution for v in stock_uM], float)
            abs_ckey = "c:" + ",".join(f"{v:.8g}" for v in cvec_uM)
        else:
            cvec_uM = np.full(len(subs), np.nan)
            abs_ckey = "c:none"
        ckey = "r:" + ",".join(f"{v:.6f}" for v in vec)
        if nnls_screen:
            _w, cube, _sm = nnls_hit_spectra(
                data_dir, it[0], baseline=baseline, trim=trim,
                min_frac=screen_min_frac, progress=None)
            local_mask = np.ones(cube.shape[1], bool); correct_again = False
        else:
            _w, cube, _m, _c = load_map(it[0])
            local_mask = mask; correct_again = baseline
        mk = os.path.normcase(os.path.normpath(it[0]))
        ev = os.path.basename(it[0])
        spectra = _map_spectra(cube, local_mask, px_per_map, spread=True,
                               baseline_correct=correct_again,
                               sampling=pixel_sampling, sampling_seed=sampling_seed,
                               map_id=it[0])
        if not spectra:
            continue
        for ya in spectra:
            X.append(_composition_features(ya)); Y.append(vec); Cabs.append(cvec_uM)
            gkey.append(ckey); abskey.append(abs_ckey); ekey.append(ev); mapkey.append(mk)
    X = np.array(X, np.float32); Y = np.array(Y, np.float32)
    Cabs = np.asarray(Cabs, float)
    gkey = np.array(gkey, object); abskey = np.array(abskey, object); ekey = np.array(ekey, object)
    mapkey = np.array(mapkey, object)
    if len(X) < 3:
        raise ValueError("need ≥3 labelled maps for a grouped benchmark.")

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

    band_mask = None
    if "band" in methods:
        # VIP marker windows come from the PURE templates only — identical in every
        # fold (no leakage), so the mask is computed once, not per condition.
        from unmix import vip_bands, _vip_fit_mask
        axis = np.asarray(wn, float)[mask]
        band_mask = _vip_fit_mask(axis, vip_bands(axis, P, subs),
                                  float(band_window), len(subs))
        if progress and band_mask is not None:
            progress(f"VIP bands — fitting on {int(band_mask.sum())}/{len(axis)} points")

    uniq = np.asarray(list(dict.fromkeys(gkey.tolist())), object)
    if cv_folds is not None and int(cv_folds) >= 2:
        held_groups = _grouped_map_folds(gkey, mapkey, int(cv_folds), seed)
        protocol = f"condition-grouped-{len(held_groups)}-fold-pixel-pooled"
    else:
        held_groups = [{g} for g in uniq.tolist()]
        protocol = "leave-one-ratio-group-out-pixel-pooled"

    out = {"subs": subs, "protocol": protocol, "cv_folds": len(held_groups),
           "pixels_per_map": int(px_per_map), "scoring_unit": "held-out map",
           "n_maps_loaded": int(len(set(mapkey.tolist()))),
           "nnls_screen": bool(nnls_screen), "screen_min_frac": float(screen_min_frac),
           "pixel_sampling": str(pixel_sampling), "sampling_seed": int(sampling_seed)}
    for mi, meth in enumerate(methods):
        tv, pv, conditions, n_maps, n_pixels, fold_ids = [], [], [], [], [], []
        for i, held in enumerate(held_groups):
            if progress:
                progress(f"{meth.upper()} grouped fold {i + 1}/{len(held_groups)}  "
                         f"[{mi + 1}/{len(methods)} methods]")
            is_test = np.array([g in held for g in gkey], bool)
            te = np.where(is_test)[0]; tr = np.where(~is_test)[0]
            ep = int(cnn_epochs) if (meth == "cnn" and cnn_epochs) else epochs
            pred = _fit_predict(meth, X[tr], Y[tr], X[te], pre=pre, epochs=ep,
                                seed=seed + i, n_components=n_components,
                                n_trees=n_trees, P_ref=P,
                                rf_max_features=rf_max_features, band_mask=band_mask,
                                mcr_iter=mcr_iter,
                                train_maps=(mapkey[tr] if int(px_per_map) > 0 else None))
            a, b, c, d = _pool_predictions_by_map(
                pred, Y[te], mapkey[te], ekey[te])
            tv.extend(a); pv.extend(b); conditions.extend(c); n_pixels.extend(d)
            n_maps.extend([1] * len(a))
            fold_ids.extend([i + 1] * len(a))
        out[meth] = {"true": tv, "pred": pv, "condition": conditions,
                     "n_maps": n_maps, "n_pixels": n_pixels, "fold": fold_ids}
    # Absolute concentration is a separate inverse problem from composition. Reproduce
    # the saved pipeline inside each fold: NNLS-screened pixels -> composition-MLP ratio
    # -> ratio/intensity quantiles -> concentration MLP. The composition head is refitted
    # without the held-out conditions, and the null is scored on those same held-out maps.
    # Composition-fraction recovery is never relabelled as absolute concentration.
    have_uM = np.isfinite(Cabs).all(axis=1) & (Cabs.sum(axis=1) > 0)
    if len(set(mapkey[have_uM].tolist())) >= 3:
        raw_uM = np.expm1(np.clip(X[have_uM], 0, None))
        comp_uM = X[have_uM]
        comp_truth_uM = Y[have_uM]
        maps_uM = mapkey[have_uM]; names_uM = ekey[have_uM]
        truth_uM = Cabs[have_uM]; cond_uM = abskey[have_uM]
        ufolds = _grouped_map_folds(
            cond_uM, maps_uM, int(cv_folds or 5), int(seed) + 701)
        scored = {
            "head": {"true_uM": [], "pred_uM": [], "condition": [],
                     "fold": [], "n_pixels": []},
            "null": {"true_uM": [], "pred_uM": [], "condition": [],
                     "fold": [], "n_pixels": []},
        }
        for fi, held in enumerate(ufolds):
            if progress:
                progress(f"uM pipeline grouped fold {fi + 1}/{len(ufolds)}  "
                         "(composition MLP -> concentration MLP)")
            is_test = np.array([g in held for g in cond_uM], bool)
            te = np.where(is_test)[0]; tr = np.where(~is_test)[0]
            ratio_tr, ratio_te = _fit_mlp_ratio_fold(
                comp_uM[tr], comp_truth_uM[tr], maps_uM[tr], comp_uM[te],
                epochs=epochs, seed=int(seed) + 1201 + fi, pre=pre)
            ctx_tr = _concentration_context_features(
                raw_uM[tr], groups=maps_uM[tr], ratios=ratio_tr)
            ctx_te = _concentration_context_features(
                raw_uM[te], groups=maps_uM[te], ratios=ratio_te)
            pred_px = _fit_concentration_fold(
                ctx_tr, truth_uM[tr], maps_uM[tr], ctx_te,
                epochs=epochs, seed=int(seed) + 1701 + fi)
            train_map_truth = np.asarray([
                truth_uM[np.where(maps_uM == mk)[0][0]]
                for mk in dict.fromkeys(maps_uM[tr].tolist())], float)
            null_centre = np.median(train_map_truth, axis=0, keepdims=True)
            null_px = np.repeat(null_centre, len(te), axis=0)
            for key, pp in (("head", pred_px), ("null", null_px)):
                a, b, c, d = _pool_concentration_by_map(
                    pp, truth_uM[te], maps_uM[te], names_uM[te])
                scored[key]["true_uM"].extend(a)
                scored[key]["pred_uM"].extend(b)
                scored[key]["condition"].extend(c)
                scored[key]["n_pixels"].extend(d)
                scored[key]["fold"].extend([fi + 1] * len(a))
        out["uM"] = {
            "subs": list(subs),
            "protocol": f"absolute-condition-grouped-{len(ufolds)}-fold-map-pooled",
            "scoring_unit": "held-out map",
            "label_source": "filename absolute values (uM)",
            "pipeline": (("nnls_screen" if nnls_screen else "all_valid_pixels")
                         + "->mlp_composition_ratio->ratio+intensity_quantiles"
                           "->mlp_concentration"),
            "input_mode": "mlp_ratio+intensity_quantiles_v1",
            "composition_hidden": [256, 64],
            "concentration_hidden": [128, 32],
            "epochs_per_fold": int(epochs),
            **scored,
        }
    return out


def benchmark_summary(bench, cut_pp=None, ref="nnls"):
    """The benchmark table's numbers, from a ``benchmark_loo`` result — computed in ONE
    place so the UI table and the CSV export cannot drift apart. Returns
    {method: {"n", "mean_dev_pp", "recovery", ["acc_at_cut"]}}.

    mean_dev_pp — mean composition deviation in percentage points (%p):
        0.5·Σ|pred−true|·100 per condition, averaged. The headline number; report it
        NEXT TO the measured reproducibility floor rather than through an accuracy cut.
    recovery — {substance: (mean %, SE %, n)} of pred/true·100 over the conditions where
        the substance is actually present. true=0 conditions are excluded (recovery is
        undefined there — detection is the ROC's job). Direction shows (over/under), but
        over- and under-recovery CANCEL in the mean, so never read it without mean_dev_pp.
    acc_at_cut — fraction of conditions with deviation ≤ ``cut_pp``. Only computed when a
        cut is passed, and the cut must be DECLARED before looking at the results —
        picking it afterwards is cherry-picking (see HANDOFF 2026-08-19 §3).
    dev_by_k — {"pure"/"binary"/"ternary"/"mean": (mean %p, SE %p, n)} — the deviation
        split by how many components the condition actually contains. "ternary" is k≥3;
        groups with no conditions are absent; "mean" is over every condition, identical
        to mean_dev_pp.
    rmse_pp — (mean, SE, n) of the per-condition root-mean-square component error, in
        percentage points. Same units as the deviation but squared-weighted, so one badly
        missed component costs more than three small slips. Report alongside, not instead:
        deviation is the number the reproducibility floor is measured in.
    logratio — (mean, SE, n) of the Aitchison (centred-log-ratio) distance, the standard
        distance for compositional data — it compares the RATIOS themselves, so being 2x
        off on a minor component costs exactly what being 2x off on the major one costs.
        The %p metrics cannot see that: on DQ24-TB12-TH6, predicting THI six times too
        low is 11.6 %p (inside a 15 %p cut) while halving DQ is 17.1 %p (outside), even
        though the second is the smaller ratio error. Dimensionless; 0 = exact, and ~0.57
        is what a single 2-fold miss costs on a ternary mixture.
    vs_ref — (median Δ %p, p, n) against method ``ref`` (default the classical NNLS
        baseline), paired condition by condition since every method is scored on the same
        folds. Negative Δ = better than the reference; p is a two-sided Wilcoxon
        signed-rank. The pairing is what makes ~100 conditions enough to separate methods:
        unpaired, the per-condition spread (sd ≈ 10 %p on this grid) swamps a few-%p
        difference, while paired it resolves ~2 %p. Fix ``ref`` before the run — choosing
        the comparator afterwards is the cherry-picking the declared cut avoids.
    """
    subs = list(bench.get("subs", []))
    devs = {}                    # per-method, per-condition deviation — for the pairing
    for m, r in bench.items():
        if isinstance(r, dict) and "true" in r:
            T_ = np.asarray(r["true"], float); P_ = np.asarray(r["pred"], float)
            devs[m] = 0.5 * np.abs(P_ - T_).sum(1) * 100.0
    out = {}
    for m, r in bench.items():
        if not isinstance(r, dict) or "true" not in r:
            continue
        T = np.asarray(r["true"], float); Pd = np.asarray(r["pred"], float)
        dev = 0.5 * np.abs(Pd - T).sum(1) * 100.0
        rec = {}
        for j, s in enumerate(subs):
            pres = T[:, j] > 0
            n = int(pres.sum())
            if n:
                q = Pd[pres, j] / T[pres, j] * 100.0
                se = float(q.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
                rec[s] = (float(q.mean()), se, n)
            else:
                rec[s] = (float("nan"), float("nan"), 0)

        def _stat(d):
            n = int(len(d))
            se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
            return (float(d.mean()), se, n) if n else (float("nan"), float("nan"), 0)

        k = (T > 0).sum(axis=1)
        by = {lab: _stat(dev[sel]) for lab, sel in
              (("pure", k == 1), ("binary", k == 2), ("ternary", k >= 3))
              if sel.any()}
        by["mean"] = _stat(dev)
        rmse = np.sqrt(((Pd - T) ** 2).mean(axis=1)) * 100.0
        from composition import aitchison_distance
        lr = np.array([aitchison_distance(T[i], Pd[i]) for i in range(len(T))])
        abs_pp = np.abs(Pd - T) * 100.0
        present = T > 0
        fold_ratio = np.full_like(T, np.nan, dtype=float)
        fold_ratio[present] = Pd[present] / T[present]
        whole_pp = {cut: float(np.mean(np.all(abs_pp <= cut, axis=1)))
                    for cut in (1.0, 3.0, 5.0)}
        whole5 = np.all(abs_pp <= 5.0, axis=1)
        whole2x = np.array([
            bool(np.all((fold_ratio[i, present[i]] >= 0.5) &
                        (fold_ratio[i, present[i]] <= 2.0)))
            for i in range(len(T))], bool)
        comp = {}
        for j, s in enumerate(subs):
            pres = present[:, j]
            comp[s] = {
                "bias_pp": float((Pd[:, j] - T[:, j]).mean() * 100.0),
                "mae_pp": float(abs_pp[:, j].mean()),
                "rmse_pp": float(np.sqrt(np.mean((Pd[:, j] - T[:, j]) ** 2)) * 100.0),
                "within_1pp": float(np.mean(abs_pp[:, j] <= 1.0)),
                "within_3pp": float(np.mean(abs_pp[:, j] <= 3.0)),
                "within_5pp": float(np.mean(abs_pp[:, j] <= 5.0)),
                "within_2fold": (float(np.mean((fold_ratio[pres, j] >= 0.5) &
                                                (fold_ratio[pres, j] <= 2.0)))
                                   if pres.any() else float("nan")),
            }
        entry = {"n": int(len(T)), "mean_dev_pp": float(dev.mean()), "recovery": rec,
                 "dev_by_k": by, "rmse_pp": _stat(rmse), "logratio": _stat(lr),
                 "whole_within_pp": whole_pp,
                 "whole_within_5pp": float(whole5.mean()),
                 "whole_within_2fold": float(whole2x.mean()),
                 "component_metrics": comp}
        dref = devs.get(ref)
        if dref is not None and m != ref and len(dref) == len(dev):
            d = dev - dref
            try:
                from scipy.stats import wilcoxon
                p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            except Exception:
                p = float("nan")
            entry["vs_ref"] = (float(np.median(d)), p, int(len(d)))
        if cut_pp is not None:
            entry["acc_at_cut"] = float((dev <= float(cut_pp)).mean()) if len(dev) else float("nan")
        out[m] = entry
    return out


def concentration_summary(eval_result, subs):
    """Summarise held-out concentration predictions without hiding multiplicative error.

    Metrics are computed only where true_uM > 0. Zero-concentration components are
    reported separately because recovery and fold error are undefined at zero.
    """
    T = np.asarray((eval_result or {}).get("true_uM", []), float)
    P = np.asarray((eval_result or {}).get("pred_uM", []), float)
    if T.ndim != 2 or P.shape != T.shape or not len(T):
        return {}
    names = list(subs)[:T.shape[1]]
    factors = (1.25, 1.5, 2.0)

    def stats(t, p):
        ok = np.isfinite(t) & np.isfinite(p) & (t > 0) & (p > 0)
        if not ok.any():
            return {"n": 0}
        fold = p[ok] / t[ok]
        loge = np.log10(fold)
        out = {
            "n": int(ok.sum()),
            "mean_recovery_pct": float(fold.mean() * 100.0),
            "median_recovery_pct": float(np.median(fold) * 100.0),
            "geometric_bias_fold": float(10.0 ** np.mean(loge)),
            "median_abs_log10_error": float(np.median(np.abs(loge))),
            "rmse_log10": float(np.sqrt(np.mean(loge ** 2))),
        }
        out["median_fold_error"] = float(10.0 ** out["median_abs_log10_error"])
        for f in factors:
            out[f"within_{str(f).replace('.', '_')}x"] = float(
                np.mean((fold >= 1.0 / f) & (fold <= f)))
        return out

    present = T > 0
    comp = {}
    for j, name in enumerate(names):
        entry = stats(T[:, j], P[:, j])
        absent = (~present[:, j]) & np.isfinite(P[:, j])
        entry["absent_n"] = int(absent.sum())
        entry["absent_median_pred_uM"] = (float(np.median(P[absent, j]))
                                           if absent.any() else float("nan"))
        comp[name] = entry
    whole = {}
    for f in factors:
        passed = []
        for i in range(len(T)):
            pr = present[i]
            if not pr.any() or np.any(~np.isfinite(P[i, pr])) or np.any(P[i, pr] <= 0):
                passed.append(False)
            else:
                fold = P[i, pr] / T[i, pr]
                passed.append(bool(np.all((fold >= 1.0 / f) & (fold <= f))))
        whole[f"all_present_within_{str(f).replace('.', '_')}x"] = float(np.mean(passed))
    return {"n_conditions": int(len(T)), "overall": stats(T, P),
            "whole_condition": whole, "component": comp}


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
    if u.get("kind") == "calibration_residual_uM_v1":
        return _apply_calibration_residual(model, wn, spectra, return_meta)
    usubs = u.get("subs") or model["subs"]
    wn = np.asarray(wn); mask = (wn >= model["lo"]) & (wn <= model["hi"])
    X = np.asarray(spectra, float)
    if X.shape[1] == len(wn):
        X = X[:, mask]
    if u.get("input_mode") in ("mlp_ratio+intensity_quantiles_v1",
                                  "cnn_ratio+intensity_quantiles_v1"):
        # the composition head's full-spectrum ratios — the same eyes that already
        # call this pixel right — normalised over the µM head's substances
        pk = np.clip(np.asarray(apply_model_pixels(model, wn, spectra), float), 0, None)
        subs_all = list(model["subs"])
        cols = [subs_all.index(s_) for s_ in usubs if s_ in subs_all]
        R_ = pk[:, cols]
        ratios = R_ / (R_.sum(axis=1, keepdims=True) + 1e-12)
        X = _concentration_context_features(X, ratios=ratios)
    elif u.get("input_mode") == "nnls_ratio+intensity_quantiles_v1":
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
    um = 10.0 ** np.clip(logv, -3.0, 6.0)
    # ---- answer INSIDE the trained range, never outside it -------------------
    # The head extrapolates freely, and on out-of-distribution input the weak
    # binder's inversion runs to thousands of µM ("한정된 범위 내에서 답해야지").
    # Predictions are clamped to each substance's training range; a clamped pixel
    # is FLAGGED, so downstream medians can drop it and the export names it.
    # UPPER side only: the explosion artifact lives above the range, and clipping
    # the LOW side up (or flagging it out) truncated the low tail — the medians of
    # every minor component came back inflated 3–9× ("여전히 튀어나가는데"). A
    # below-range prediction is information ("small"), not an artifact.
    clamped = np.zeros_like(um, bool)
    rngs = u.get("ranges_M")
    if rngs is not None:
        r_um = np.asarray(rngs, float) * 1e6               # (n_subs, 2) in µM
        for k in range(min(len(usubs), len(r_um))):
            hi_u = r_um[k][1]
            if np.isfinite(hi_u) and hi_u > 0:
                clamped[:, k] = um[:, k] > hi_u
                um[:, k] = np.minimum(um[:, k], hi_u)
    result = (um, list(usubs))
    if not return_meta:
        return result
    meta = {}
    if u.get("kind") == "map_pooled_pixel_concentration_v1":
        dist = np.sqrt(np.mean(Xe ** 2, axis=1))
        ood = dist > float(u.get("ood_threshold", np.inf))
        rngs_out = u.get("ranges_M")
        vr = u.get("validated_ranges_M")
        if vr is not None and rngs_out is not None:
            vr = np.asarray(vr, float); rngs_out = np.asarray(rngs_out, float).copy()
            ok_ = np.isfinite(vr).all(axis=1)
            rngs_out[ok_] = vr[ok_]              # validated window beats label range
        meta = {"feature_ood": ood,
                "component_ood": np.repeat(ood[:, None], len(usubs), axis=1) | clamped,
                "ranges_M": rngs_out}
    return (*result, meta)


def _presence_gate(model, path, res, comp, wn=None, hit_cube=None):
    """presence 판정을 res 에 붙이고 ND 성분을 조성 0·µM 0(ND 플래그)으로 게이트한다.

    "NNLS/스크린 근거가 없으면 회귀가 되살리지 않는다" 규칙의 배선. hit_cube 를 안
    받으면 스크린을 직접 돌린다 — 모델의 nnls_screen 설정과 무관하게 presence
    feature 는 스크린된 픽셀에서 계산돼야 학습(bundle)과 일치한다. 원값은
    comp_raw / uM_raw 로 보존한다. 사이드카가 없으면 아무것도 하지 않는다."""
    if not model.get("_presence_head"):
        return comp
    try:
        if hit_cube is None:
            wn, hit_cube, _sm = nnls_hit_spectra(
                model.get("data_dir") or os.path.dirname(path), path,
                baseline=model.get("baseline", True), trim=model.get("trim"),
                min_frac=model.get("screen_min_frac", 0.15), progress=None)
        pres = apply_presence(model, wn, hit_cube)
    except Exception:
        return comp
    if not pres:
        return comp
    res["presence"] = pres
    nd = [sn for sn, p in pres.items() if p.get("state") == "ND" and sn in comp]
    if nd:
        res["comp_raw"] = dict(comp)
        for sn in nd:
            comp[sn] = 0.0
        tot = sum(comp.values())
        if tot > 0:
            comp = {sn: v / tot for sn, v in comp.items()}
        if res.get("uM"):
            res["uM_raw"] = dict(res["uM"])
            res["uM_nd"] = {sn: (sn in nd) for sn in res["uM"]}
            for sn in nd:
                if sn in res["uM"]:
                    res["uM"][sn] = 0.0
    return comp


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
            comp = _presence_gate(model, path, res, comp, wn, hit_cube)
        else:
            wn, cube, _m, _c = load_map(path)
            res = apply_model(model, wn, cube); comp = dict(res["composition"])
            comp = _presence_gate(model, path, res, comp)
        s = sum(float(ratio.get(sn, 0)) for sn in subs)
        nom = np.zeros(len(subs)); mn = np.zeros(len(subs))
        for sn in subs:
            o = subs.index(sn)
            nom[o] = float(ratio.get(sn, 0)) / s if s > 0 else 0.0
            mn[o] = comp.get(sn, 0.0)
        row = {"name": os.path.basename(path).replace("_corrected", "").replace(".csv", ""),
               "nominal": nom, "mean": mn}
        if res.get("presence"):
            row["presence"] = res["presence"]
        if res.get("uM_nd"):
            row["uM_nd"] = res["uM_nd"]
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
        model = pickle.load(f)
    # presence 사이드카(<dlm이름>.presence.json)가 있으면 싣는다 — 없으면 예전 그대로.
    try:
        side = os.path.splitext(str(path))[0] + ".presence.json"
        if os.path.exists(side):
            import json as _json
            with open(side, encoding="utf-8") as fh:
                model["_presence_head"] = _json.load(fh)
    except Exception:
        model.pop("_presence_head", None)
    return model


# ------------------------------------------------------------------ presence 헤드
# 회귀는 0을 출력하지 못한다(softmax·µM 곱셈 보정). 존재/부재는 별도의 성분별
# 로지스틱 게이트가 판정한다 — 근거와 검증은 documentation/PRESENCE_HEAD_2026-08-31.md.
_PRESENCE_BANDS = {"DQ": 1570.0, "TBZ": 1270.0, "THI": 1367.0}
_PRESENCE_REF_CACHE = {}


def _presence_refs(model):
    """_refs/_nuisance_refs 캐시 — 맵마다 순물질을 디스크에서 다시 읽지 않는다
    (Google Drive 위라 맵당 수 초짜리 I/O 가 된다)."""
    key = (str(model.get("data_dir")), bool(model.get("baseline", True)),
           tuple(model.get("trim") or ()))
    if key not in _PRESENCE_REF_CACHE:
        subs_r, wn_r, mask_r, P, lo, hi = _refs(model.get("data_dir"),
                                                model.get("baseline", True), model.get("trim"))
        NU = _nuisance_refs(model.get("data_dir"), model.get("baseline", True), mask_r)
        PA = np.vstack([P, NU]) if NU is not None else P
        _PRESENCE_REF_CACHE[key] = (subs_r, mask_r, PA)
    return _PRESENCE_REF_CACHE[key]


def presence_map_context(model, raw_full, wn):
    """맵 하나의 presence feature 재료. raw_full: (n_px, n_feat) baseline 제거·음수 클립.

    배경(BLK/INK) 템플릿을 포함해 NNLS 를 풀어야 한다 — 배경 없이 3-템플릿으로
    풀면 저신호 맵에서 부재 성분 겉보기 분율이 26~36%까지 부풀어 판정이 무너진다."""
    from scipy.optimize import nnls as _scinnls
    subs_r, mask_r, PA = _presence_refs(model)
    raw_full = np.clip(np.asarray(raw_full, float), 0, None)
    # nnls_hit_spectra 가 준 큐브는 이미 트림 창(mask)으로 잘려 있다 — 폭으로 구분.
    sub = raw_full if raw_full.shape[1] == int(np.sum(mask_r)) else raw_full[:, mask_r]
    fr_an, shares = [], []
    for y in sub:
        yn = y / (np.linalg.norm(y) + 1e-12)
        w, _ = _scinnls(PA.T, yn)
        a = w[:len(subs_r)]
        shares.append(a.sum() / (w.sum() + 1e-12))
        fr_an.append(a / (a.sum() + 1e-12))
    fr_an = np.asarray(fr_an); shares = np.asarray(shares)
    hit = shares >= float(model.get("screen_min_frac", 0.15))
    use = hit if hit.any() else np.ones(len(shares), bool)
    wn = np.asarray(wn, float)
    band = {}
    for s, c in _PRESENCE_BANDS.items():
        m = np.abs(wn - c) <= 10.0
        band[s] = float(np.median(raw_full[use][:, m].max(1))) if m.any() else 0.0
    return {"subs": subs_r, "frac": fr_an[use].mean(0),
            "analyte_share": float(shares[use].mean()),
            "band": band, "total": float(np.median(raw_full[use].sum(1)))}


def presence_feature_vector(ctx, s):
    """학습·적용이 공유하는 feature 정의 — 순서를 바꾸면 사이드카가 무효가 된다."""
    subs_r = ctx["subs"]; j = subs_r.index(s)
    others = [k for k in range(len(subs_r)) if k != j]
    return [float(ctx["frac"][j]),
            float(max(ctx["frac"][k] for k in others)),
            float(np.log1p(ctx["band"].get(s, 0.0))),
            float(np.log1p(max(ctx["band"].get(subs_r[k], 0.0) for k in others))),
            float(np.log1p(ctx["total"])),
            float(ctx["analyte_share"])]


def apply_presence(model, wn, cube):
    """성분별 presence 확률과 3-상태 판정. 사이드카가 없으면 None."""
    head = model.get("_presence_head")
    if not head:
        return None
    ctx = presence_map_context(model, cube, wn)
    lo_t = float(head.get("thresholds", {}).get("nd", 0.2))
    hi_t = float(head.get("thresholds", {}).get("detected", 0.8))
    out = {}
    for s, prm in head["components"].items():
        f = np.asarray(presence_feature_vector(ctx, s), float)
        z = (f - np.asarray(prm["mu"], float)) / (np.asarray(prm["sd"], float) + 1e-12)
        p = 1.0 / (1.0 + np.exp(-(float(np.dot(prm["coef"], z)) + float(prm["intercept"]))))
        out[s] = {"prob": float(p),
                  "state": ("Detected" if p > hi_t else ("ND" if p < lo_t else "Indeterminate"))}
    return out
