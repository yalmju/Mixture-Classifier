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


def unmix_map(data_dir, test_path, method="nnls", baseline=True, trim=None,
              min_frac=0.05, calib_path=None, progress=None) -> UnmixResult:
    """Unmix ``test_path`` against the substances in ``data_dir`` (background
    included) by ``method`` ('nnls' or 'mcr'). ``min_frac`` is how much non-bg
    abundance a pixel needs to count as a hit rather than background. If
    ``calib_path`` (a dilution-series CSV) is given, also recover per-pixel absolute
    concentration (M) via Langmuir calibration of the non-background substances."""
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

    if method == "mcr":
        if progress:
            progress("MCR-ALS refining component spectra")
        A, templates = _mcr_als(X, templates, progress=progress)
    else:
        A = np.zeros((len(X), K))
        for i, y in enumerate(X):
            A[i], _ = nnls(templates.T, y)
            if progress and i % 300 == 0:
                progress(f"NNLS unmixing — pixel {i}/{len(X)}")

    recon = A @ templates
    ss_res = np.sum((X - recon) ** 2, axis=1)
    ss_tot = np.sum((X - X.mean(axis=1, keepdims=True)) ** 2, axis=1)
    reliab = np.clip(1.0 - np.divide(ss_res, ss_tot, out=np.ones_like(ss_res),
                                     where=ss_tot > 0), 0.0, 1.0)

    tot = A.sum(axis=1, keepdims=True)
    frac = np.divide(A, tot, out=np.zeros_like(A), where=tot > 0)
    Anb = A[:, nonbg]
    nb_tot = Anb.sum(axis=1, keepdims=True)
    ratio_nb = np.divide(Anb, nb_tot, out=np.zeros_like(Anb), where=nb_tot > 0)
    hit = frac[:, nonbg].sum(axis=1) >= min_frac      # substance share above threshold
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
        conc_avg=conc_avg, pp_theta=pp_theta, calib_r2=calib_r2)


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
