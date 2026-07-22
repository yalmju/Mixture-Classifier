"""unmix.py — unmix ONE test SERS map against the reference substances and report
it three ways, the way the old SERS discriminator did:

    map          per-substance intensity (abundance) map — where each substance is
    reliability  per-pixel fit quality (R² of the linear reconstruction)
    validation   measured vs reconstructed mean spectrum + spectral angle (SAM)

plus the overall composition as a pie. Two methods:

    "nnls"   per-pixel non-negative least squares against the FIXED reference
             templates (mean spectrum per substance).
    "mcr"    MCR-ALS: start from those templates and alternately refine the
             concentrations and the component spectra under non-negativity — it
             adapts the spectra to the sample instead of assuming the references
             are exact.

UI-agnostic (numpy / scipy only); the Qt tab just draws the arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

from real_data import load_map
from dataset import discover_dataset, is_blank
from sers_mixture import preprocess


@dataclass
class UnmixResult:
    comps: list                  # reference substances (non-blank)
    method: str                  # "nnls" or "mcr"
    wn: np.ndarray               # wavenumber axis
    coords: np.ndarray           # (n_pix, 2) pixel X/Y
    A: np.ndarray                # (n_pix, K) non-negative abundance per substance
    frac: np.ndarray             # (n_pix, K) per-pixel composition (rows sum to 1)
    reliab: np.ndarray           # (n_pix,) per-pixel reconstruction R² (0..1)
    sam: np.ndarray              # (n_pix,) spectral angle measured↔reconstructed (deg)
    comp_frac: np.ndarray        # (K,) overall composition fraction
    templates: np.ndarray        # (K, n_feat) component spectra used for the fit
    meas_mean: np.ndarray        # (n_feat,) preprocessed measured mean spectrum
    recon_mean: np.ndarray       # (n_feat,) reconstructed mean spectrum
    n_pixels: int
    dominant: str                # substance with the largest overall fraction
    mean_r2: float               # mean per-pixel reliability


def _reference_templates(data_dir, trim, baseline):
    """Mean preprocessed spectrum per non-blank substance, plus the axis."""
    groups = discover_dataset(data_dir)
    comps = [c for c, _ in groups if not is_blank(c)]
    if not comps:
        raise FileNotFoundError(
            f"no substance references found in {data_dir} (need at least one "
            "non-blank class). Organise them in Samples first.")
    path_of = {c: [p for _b, p, _r in maps] for c, maps in groups}
    means, wn = [], None
    for c in comps:
        cbs = []
        for p in path_of[c]:
            wn, cube, _m, _c = load_map(p)
            cbs.append(cube)
        means.append(np.vstack(cbs).mean(axis=0))
    return comps, wn, np.array(means)


def _mcr_als(X, S0, n_iter=8, progress=None):
    """Compact MCR-ALS with non-negativity on both C and S, seeded by the
    references S0. Returns (C (n_pix, K), S (K, n_feat))."""
    S = np.clip(np.asarray(S0, float), 0.0, None)
    npix, nfeat = X.shape
    C = np.zeros((npix, S.shape[0]))
    for it in range(n_iter):
        for i in range(npix):                     # concentrations given spectra
            C[i], _ = nnls(S.T, X[i])
        for f in range(nfeat):                    # spectra given concentrations
            S[:, f], _ = nnls(C, X[:, f])
        n = np.linalg.norm(S, axis=1, keepdims=True)   # fix scale ambiguity
        S = S / np.where(n > 0, n, 1.0)
        if progress:
            progress(f"MCR-ALS — iteration {it + 1}/{n_iter}")
    return C, S


def unmix_map(data_dir, test_path, method="nnls", baseline=True, trim=None,
              progress=None) -> UnmixResult:
    """Unmix the map at ``test_path`` against the substances in ``data_dir``."""
    comps, wn, means = _reference_templates(data_dir, trim, baseline)
    wn_u, cube_u, _mean_u, coord = load_map(test_path)

    if trim is not None:
        lo, hi = trim
        m = (wn >= lo) & (wn <= hi)
        if m.sum() >= 10:
            means = means[:, m]; cube_u = cube_u[:, m]; wn = wn[m]

    if progress:
        progress("preprocessing spectra")
    templates = preprocess(means, do_baseline=baseline)      # (K, n_feat), L2 rows
    X = preprocess(cube_u, do_baseline=baseline)             # (n_pix, n_feat)
    K = len(comps)

    if method == "mcr":
        if progress:
            progress("MCR-ALS refining component spectra")
        A, S = _mcr_als(X, templates, progress=progress)
        templates = S
    else:                                                    # per-pixel NNLS
        A = np.zeros((len(X), K))
        for i, y in enumerate(X):
            A[i], _ = nnls(templates.T, y)
            if progress and i % 200 == 0:
                progress(f"NNLS unmixing — pixel {i}/{len(X)}")

    # reconstruction, reliability (R²) and validation (spectral angle) per pixel
    recon = A @ templates                                    # (n_pix, n_feat)
    resid = X - recon
    ss_res = np.sum(resid ** 2, axis=1)
    ss_tot = np.sum((X - X.mean(axis=1, keepdims=True)) ** 2, axis=1)
    reliab = 1.0 - np.divide(ss_res, ss_tot, out=np.ones_like(ss_res),
                             where=ss_tot > 0)
    reliab = np.clip(reliab, 0.0, 1.0)
    dot = np.sum(X * recon, axis=1)
    nx = np.linalg.norm(X, axis=1); nr = np.linalg.norm(recon, axis=1)
    cos = np.divide(dot, nx * nr, out=np.zeros_like(dot), where=(nx * nr) > 0)
    sam = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    tot = A.sum(axis=1, keepdims=True)
    frac = np.divide(A, tot, out=np.zeros_like(A), where=tot > 0)
    comp_frac = frac.mean(axis=0)
    dominant = comps[int(comp_frac.argmax())]

    return UnmixResult(
        comps=comps, method=method, wn=wn, coords=coord, A=A, frac=frac,
        reliab=reliab, sam=sam, comp_frac=comp_frac, templates=templates,
        meas_mean=X.mean(axis=0), recon_mean=recon.mean(axis=0),
        n_pixels=len(X), dominant=dominant, mean_r2=float(reliab.mean()))


if __name__ == "__main__":
    import sys
    r = unmix_map(sys.argv[1], sys.argv[2], method=sys.argv[3] if len(sys.argv) > 3 else "nnls")
    print("method:", r.method, "| dominant:", r.dominant, "| mean R²:", round(r.mean_r2, 3))
    print("composition:", {c: round(float(v), 3) for c, v in zip(r.comps, r.comp_frac)})
