"""unmix.py — unmix ONE test SERS map against the reference substances (background
included), by NNLS (fixed reference templates) or MCR-ALS (spectra refined to the
sample). Returns everything the Real-data tab draws:

    intensity   raw baseline-removed spectra (for a band-intensity image + per-pixel
                spectrum on click)
    composition per-pixel abundance / ratio among the substances (background greyed)

UI-agnostic (numpy / scipy only).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

from real_data import load_map
from dataset import discover_dataset, is_blank
from sers_mixture import als_baseline
from calibration import calibrate, quantify, _langmuir_B
from io_utils import load_calibration_csv


@dataclass
class UnmixResult:
    comps: list                  # ALL reference classes, background last
    bg_mask: np.ndarray          # (K,) True where the class is background/blank
    nonbg: list                  # indices of the non-background substances
    method: str                  # "nnls" or "mcr"
    wn: np.ndarray
    coords: np.ndarray           # (n_pix, 2)
    spectra: np.ndarray          # (n_pix, n_feat) baseline-removed measured spectra
    templates: np.ndarray        # (K, n_feat) unit templates used for the fit
    A: np.ndarray                # (n_pix, K) abundance (chosen method)
    ratio_nb: np.ndarray         # (n_pix, Knb) composition among non-bg (rows sum 1)
    hit: np.ndarray              # (n_pix,) True where a substance (not bg) dominates
    reliab: np.ndarray           # (n_pix,) reconstruction R²
    n_pixels: int
    hit_frac: float
    mean_ratio: np.ndarray       # (Knb,) mean non-bg composition over hit pixels
    dominant: str
    mean_r2: float
    calibrated: bool = False     # True if a dilution-series calibration was applied
    conc: np.ndarray = None      # (n_pix, Knb) per-pixel absolute concentration (M)
    conc_avg: np.ndarray = None  # (Knb,) mean concentration over hit pixels (M)
    pp_theta: np.ndarray = None  # (n_pix,) total surface coverage Σθ per pixel
    calib_r2: np.ndarray = None  # (Knb,) isotherm fit R² per substance
    bg_score: np.ndarray = None  # (n_pix,) match to the MEASURED background (0..1)
    bg_thr: float = None         # score at/above which a pixel is judged background
    hit_rule: str = ""           # which single rule decided background vs substance


def _baseline_removed(cube, baseline):
    """ALS-baseline-subtract each spectrum (or just clip if baseline off) — the
    absolute-ish intensity used for the band image and per-pixel display."""
    X = np.asarray(cube, float)
    if baseline:
        X = np.stack([y - als_baseline(y) for y in X])
    return np.clip(X, 0.0, None)


def _l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n > 0, n, 1.0)


def _mcr_als(X, S0, n_iter=6, progress=None):
    """MCR-ALS with non-negativity on C and S, seeded by the references S0."""
    S = np.clip(np.asarray(S0, float), 0.0, None)
    npix, nfeat = X.shape
    C = np.zeros((npix, S.shape[0]))
    for it in range(n_iter):
        for i in range(npix):
            C[i], _ = nnls(S.T, X[i])
        for f in range(nfeat):
            S[:, f], _ = nnls(C, X[:, f])
        S = _l2(S)
        if progress:
            progress(f"MCR-ALS — iteration {it + 1}/{n_iter}")
    return C, S


def _templates(data_dir, baseline, progress):
    groups = discover_dataset(data_dir)
    if not groups:
        raise FileNotFoundError(
            f"no reference classes found in {data_dir} — organise them in Samples.")
    names, means, wn = [], [], None
    for c, maps in groups:
        cbs = []
        for _b, p, _r in maps:
            wn, cube, _m, _c = load_map(p)
            cbs.append(cube)
        names.append(c); means.append(np.vstack(cbs).mean(axis=0))
    return names, wn, np.array(means)


def vip_bands(axis, ref_spectra, names, k=2, min_purity=0.5, lo=400.0, hi=1800.0):
    """Per-compound VIP-style marker band(s): among each compound's REAL peaks, the
    one(s) where its L2-normalised reference SHAPE most exceeds every OTHER compound
    (least cross-talk). L2-normalising first removes the response-factor bias so a
    strong emitter doesn't win every band. Returns {name: [wavenumbers]}.

    UI-agnostic mirror of the Quantify page's ``_vip_peaks`` so Validate / Real can
    unmix on the same discriminative bands the user quantifies on."""
    from scipy.signal import find_peaks
    axis = np.asarray(axis, float)
    R = np.asarray(ref_spectra, float)
    R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)      # shape, not magnitude
    band = (axis >= lo) & (axis <= hi)
    if band.sum() < 5:
        band = np.ones(len(axis), bool)
    out = {}
    for i, nm in enumerate(names):
        others = np.delete(R, i, axis=0)
        omax = others.max(axis=0) if len(others) else np.zeros(R.shape[1])
        score = R[i] - omax                                        # one-vs-rest separation
        idx, _ = find_peaks(R[i], prominence=(R[i].max() or 1.0) * 0.05)
        idx = [j for j in idx if band[j]]
        if not idx:
            idx = [int(np.argmax(np.where(band, score, -np.inf)))]
        ordered = sorted(idx, key=lambda j: score[j], reverse=True)
        chosen = []
        for j in ordered:                                          # up to k CLEAN bands
            pur = float(np.clip(score[j] / (R[i][j] + 1e-12), 0.0, 1.0))
            if not chosen or (pur >= min_purity and len(chosen) < k):
                chosen.append(float(axis[j]))
            if len(chosen) >= k:
                break
        out[nm] = chosen
    return out


def compute_vip_bands(data_dir, baseline=True, trim=None, k=2, lo=400.0, hi=1800.0):
    """Load the references in ``data_dir`` and return (nb_names, {name: [wavenumbers]})
    of the non-background substances' VIP marker bands — for the Validate UI to show
    and edit before it drives the VIP-band NNLS."""
    names, wn, means = _templates(data_dir, baseline, None)
    if trim is not None:
        lo_t, hi_t = trim
        m = (wn >= lo_t) & (wn <= hi_t)
        if m.sum() >= 10:
            means = means[:, m]; wn = wn[m]
    nonbg = [i for i in range(len(names)) if not is_blank(names[i])]
    nb_names = [names[i] for i in nonbg]
    ref = _baseline_removed(means[nonbg], baseline)
    return nb_names, vip_bands(wn, ref, nb_names, k=k, lo=lo, hi=hi)


def _bg_match(X, mu, V):
    """Per-row R² of reconstructing X from the background subspace (mean ``mu`` +
    principal directions ``V``). 1 = the pixel is fully explained by background."""
    Z = X - mu
    rec = Z @ V.T @ V
    ss_res = ((Z - rec) ** 2).sum(axis=1)
    ss_tot = ((X - X.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    return np.clip(1.0 - np.divide(ss_res, ss_tot, out=np.ones_like(ss_res),
                                   where=ss_tot > 0), 0.0, 1.0)


def _bg_subspace(bg_paths, wn, baseline, trim, n_comp=6):
    """Learn the MEASURED background from blank map(s): (mu, V, thr) where mu + V
    span the background spectra (same preprocessing as the test pixels) and thr is
    a self-calibrated match score — fit on half the background pixels, thr = the
    5th percentile of the OTHER half's scores, so ~95% of true background passes
    without hand-tuning. A test pixel scoring >= thr is judged background."""
    rows = []
    for p in bg_paths:
        wn_b, cube, _m, _c = load_map(p)
        wn_b = np.asarray(wn_b, float); cube = np.asarray(cube, float)
        if trim is not None:
            lo, hi = trim
            mb = (wn_b >= lo) & (wn_b <= hi)
            if mb.sum() >= 10:
                wn_b = wn_b[mb]; cube = cube[:, mb]
        Xb = _baseline_removed(cube, baseline)
        if len(wn_b) != len(wn) or not np.allclose(wn_b, wn):
            Xb = np.stack([np.interp(wn, wn_b, y) for y in Xb])
        rows.append(_l2(Xb))
    B = np.vstack(rows)
    if len(B) >= 8:                       # held-out half keeps the threshold honest
        fit, hold = B[0::2], B[1::2]
    else:
        fit, hold = B, B
    mu = fit.mean(axis=0)
    k = int(min(n_comp, max(1, len(fit) - 1)))
    _u, _s, Vt = np.linalg.svd(fit - mu, full_matrices=False)
    V = Vt[:k]
    thr = float(np.quantile(_bg_match(hold, mu, V), 0.05))
    return mu, V, thr


def _vip_fit_mask(wn, peak_map, window, min_pts):
    """Boolean mask over ``wn`` = union of each compound's VIP band ± window. None when
    no peak_map, or too few points to fit ``min_pts`` components (falls back to full)."""
    if not peak_map:
        return None
    mask = np.zeros(len(wn), bool)
    for bands in peak_map.values():
        for b in np.atleast_1d(bands):
            mask |= (wn >= float(b) - window) & (wn <= float(b) + window)
    return mask if mask.sum() >= max(min_pts, 3) else None


def unmix_map(data_dir, test_path, method="nnls", baseline=True, trim=None,
              min_frac=0.05, hit_mode="threshold", calib_path=None,
              peak_map=None, peak_window=10.0, dl_model=None, bg_map=None,
              progress=None) -> UnmixResult:
    """Unmix ``test_path`` against the substances in ``data_dir`` (background
    included) by ``method`` ('nnls' or 'mcr'). ``hit_mode`` decides which pixels
    count as a substance rather than background: 'auto' uses the learned background
    directly (a pixel is a hit when its strongest component is a substance, not the
    blank — threshold-free), 'threshold' uses ``min_frac`` (the substances must make
    up at least that fraction of the pixel). If ``calib_path`` (a dilution-series
    CSV) is given, also recover per-pixel absolute concentration (M).

    ``peak_map`` ({name: [wavenumbers]}) restricts the NNLS fit to each compound's VIP
    marker-band windows (± ``peak_window`` cm⁻¹): the composition is then decomposed on
    the least-cross-talk discriminative bands instead of the whole spectrum. Display /
    band-image spectra stay full; only the abundance fit (and its reconstruction R²) use
    the masked region. Ignored for method='mcr'.

    ``bg_map`` (path or list of paths to MEASURED blank/background maps, e.g. Pest/BLk)
    judges each pixel background vs substance directly: pixels whose spectrum is
    explained by the measured-background subspace are background no matter what the
    unmix says — so the hit decision no longer leans on NNLS abundances."""
    names, wn, means = _templates(data_dir, baseline, progress)
    wn_u, cube_u, _mean_u, coord = load_map(test_path)

    if trim is not None:
        lo, hi = trim
        m = (wn >= lo) & (wn <= hi)
        if m.sum() >= 10:
            means = means[:, m]; cube_u = cube_u[:, m]; wn = wn[m]

    if progress:
        progress("preprocessing spectra")
    spectra = _baseline_removed(cube_u, baseline)          # for the band image / display
    X = _l2(spectra)                                        # unit spectra for the fit
    templates = _l2(_baseline_removed(means, baseline))
    ref_templates = templates.copy()                       # references (MCR won't touch)
    K = len(names)
    bg_mask = np.array([is_blank(c) for c in names])
    nonbg = [i for i in range(K) if not bg_mask[i]]

    bg_score = bg_thr = None
    if bg_map:
        paths = [bg_map] if isinstance(bg_map, str) else list(bg_map)
        if progress:
            progress("learning the measured background")
        mu_b, V_b, bg_thr = _bg_subspace(paths, wn, baseline, trim)
        bg_score = _bg_match(X, mu_b, V_b)

    if method == "mcr":
        if progress:
            progress("MCR-ALS refining component spectra")
        A, templates = _mcr_als(X, templates, progress=progress)
        fit_T, fit_X = templates, X                        # peak_map ignored for MCR
    else:
        # optional: fit only on the VIP marker-band windows (least cross-talk). Keeps
        # `templates`/`spectra` full for display; A + reliab use the masked region.
        fit_mask = _vip_fit_mask(wn, peak_map, peak_window, K)
        if fit_mask is not None:
            fit_T = _l2(_baseline_removed(means, baseline)[:, fit_mask])
            fit_X = _l2(spectra[:, fit_mask])
            if progress:
                progress(f"VIP-band NNLS — {int(fit_mask.sum())}/{len(wn)} points")
        else:
            fit_T, fit_X = templates, X
        A = np.zeros((len(fit_X), K))
        for i, y in enumerate(fit_X):
            A[i], _ = nnls(fit_T.T, y)
            if progress and i % 300 == 0:
                progress(f"NNLS unmixing — pixel {i}/{len(fit_X)}")
    A_ls = A                             # least-squares abundances, for the R² QC
    if method == "dlpx" and dl_model is not None:
        # the trained composition model, run per pixel: it replaces the abundance
        # channels. The reliability (R²) mask stays on the NNLS reconstruction
        # (A_ls) — the model outputs are probabilities, not fit coefficients, so
        # reconstructing from them would misread nearly every pixel as low-R².
        if progress:
            progress("composition model — per-pixel prediction")
        from dl_model import apply_model_pixels
        A_ls = A.copy()
        Pk = apply_model_pixels(dl_model, wn, spectra)              # (n_px, n_subs)
        sub_names = dl_model.get("subs", [])
        blank = dl_model.get("blank")
        if blank and blank in sub_names:
            # the model judges background itself: use its channels directly, including
            # the blank one, so hit/background no longer depends on NNLS
            for j, nm in enumerate(sub_names):
                tgt = nm if nm in names else next((c for c in names if is_blank(c)), None)
                if tgt in names:
                    A[:, names.index(tgt)] = Pk[:, j]
        else:
            tot = A[:, nonbg].sum(1, keepdims=True)                 # keep NNLS' substance mass
            for j, nm in enumerate(sub_names):
                if nm in names:
                    A[:, names.index(nm)] = Pk[:, j] * tot[:, 0]

    recon = A_ls @ fit_T
    ss_res = np.sum((fit_X - recon) ** 2, axis=1)
    ss_tot = np.sum((fit_X - fit_X.mean(axis=1, keepdims=True)) ** 2, axis=1)
    reliab = np.clip(1.0 - np.divide(ss_res, ss_tot, out=np.ones_like(ss_res),
                                     where=ss_tot > 0), 0.0, 1.0)

    tot = A.sum(axis=1, keepdims=True)
    frac = np.divide(A, tot, out=np.zeros_like(A), where=tot > 0)
    Anb = A[:, nonbg]
    nb_tot = Anb.sum(axis=1, keepdims=True)
    ratio_nb = np.divide(Anb, nb_tot, out=np.zeros_like(Anb), where=nb_tot > 0)
    # ---- ONE rule decides background vs substance ----------------------------
    # Stacking gates (model blank channel AND a fraction threshold AND a measured
    # background AND the R² filter) meant a dropped pixel could not be traced to a
    # cause. Exactly one rule applies, and the result records which:
    #   1. a loaded background map — the user measured THIS sample's substrate, so
    #      that measurement wins outright;
    #   2. else a composition model carrying a blank class — it judges background
    #      itself, per pixel, and nothing else gets a vote;
    #   3. else the classical NNLS/MCR behaviour (auto argmax or min_frac).
    dl_blank = bool(method == "dlpx" and dl_model is not None
                    and dl_model.get("blank")
                    and dl_model.get("blank") in (dl_model.get("subs") or []))
    if bg_score is not None:
        hit = bg_score < bg_thr
        hit_rule = f"measured background map (match < {bg_thr:.2f})"
    elif dl_blank:
        hit = ~bg_mask[A.argmax(axis=1)]
        hit_rule = f"composition model's {dl_model['blank']} channel (per pixel)"
    elif hit_mode == "auto":                          # BLK-based: strongest wins
        hit = ~bg_mask[A.argmax(axis=1)]
        hit_rule = "strongest component is a substance (auto)"
    else:                                             # substance share above threshold
        hit = frac[:, nonbg].sum(axis=1) >= min_frac
        hit_rule = f"substance share ≥ {min_frac:.2f}"
    hit_frac = float(hit.mean())
    mean_ratio = ratio_nb[hit].mean(axis=0) if hit.any() else ratio_nb.mean(axis=0)
    dominant = [names[i] for i in nonbg][int(mean_ratio.argmax())] if nonbg else names[0]

    # ---- optional: per-pixel ABSOLUTE concentration via Langmuir calibration ----
    calibrated, conc, conc_avg, pp_theta, calib_r2 = False, None, None, None, None
    if calib_path and nonbg:
        nb_names = [names[i] for i in nonbg]
        pures = ref_templates[nonbg]                       # calibrate against the references
        conc, pp_theta, calib_r2 = _quantify_map(
            calib_path, nb_names, pures, spectra, wn, trim, baseline, hit, progress)
        conc_avg = conc[hit].mean(axis=0) if hit.any() else conc.mean(axis=0)
        calibrated = True

    return UnmixResult(
        comps=names, bg_mask=bg_mask, nonbg=nonbg, method=method, wn=wn,
        coords=coord, spectra=spectra.astype(np.float32), templates=templates,
        A=A, ratio_nb=ratio_nb, hit=hit, reliab=reliab, n_pixels=len(X),
        hit_frac=hit_frac, mean_ratio=mean_ratio, dominant=dominant,
        mean_r2=float(reliab.mean()), calibrated=calibrated, conc=conc,
        conc_avg=conc_avg, pp_theta=pp_theta, calib_r2=calib_r2,
        bg_score=bg_score, bg_thr=bg_thr, hit_rule=hit_rule)


def _quantify_map(calib_path, nb_names, pures, spectra, wn, trim, baseline, hit,
                  progress=None):
    """Absolute concentration (M) per pixel for the non-background substances, from
    a dilution-series calibration. Returns (conc (n,Knb), theta (n,), r2 (Knb,))."""
    axis_c, names_c, dils = load_calibration_csv(calib_path)
    cidx = {n: k for k, n in enumerate(names_c)}
    missing = [c for c in nb_names if c not in cidx]
    if missing:
        raise ValueError(f"calibration is missing substances {missing} "
                         f"(it has {names_c}); calibrate the same references.")
    aligned = []
    for c in nb_names:
        Cg, specs = dils[cidx[c]]
        specs = np.asarray(specs, float)
        if trim is not None:
            lo, hi = trim; mc = (axis_c >= lo) & (axis_c <= hi)
            if mc.sum() >= 10:
                specs = specs[:, mc]
        aligned.append((Cg, _baseline_removed(specs, baseline)))
    if aligned[0][1].shape[1] != pures.shape[1]:
        raise ValueError("calibration axis does not match the reference maps "
                         f"({aligned[0][1].shape[1]} vs {pures.shape[1]} points).")
    calib = calibrate(aligned, pures, nb_names)
    r2 = np.zeros(len(nb_names))
    for k in range(len(nb_names)):
        C, B = np.asarray(calib.C_series[k]), np.asarray(calib.B_series[k])
        pred = _langmuir_B(C, calib.gA[k], calib.K[k])
        sst = float(np.sum((B - B.mean()) ** 2))
        r2[k] = 1.0 - float(np.sum((B - pred) ** 2)) / sst if sst > 0 else 0.0
    conc = np.zeros((len(spectra), len(nb_names))); theta = np.zeros(len(spectra))
    idx = np.where(hit)[0]
    for n, i in enumerate(idx):
        if progress and n % 300 == 0:
            progress(f"quantifying — pixel {n}/{len(idx)}")
        q = quantify(spectra[i], pures, calib)
        conc[i] = q["C"]; theta[i] = q["theta_total"]
    return conc, theta, r2


if __name__ == "__main__":
    import sys
    r = unmix_map(sys.argv[1], sys.argv[2],
                  method=sys.argv[3] if len(sys.argv) > 3 else "nnls")
    nb = [r.comps[i] for i in r.nonbg]
    print("method:", r.method, "| dominant:", r.dominant, "| hit%:",
          round(r.hit_frac * 100))
    print("mean ratio:", {n: round(float(v), 3) for n, v in zip(nb, r.mean_ratio)})
