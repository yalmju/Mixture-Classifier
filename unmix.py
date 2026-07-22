"""unmix.py — unmix ONE test SERS map against the reference substances (BACKGROUND
included) and report it the way the old SERS discriminator did:

    RGB intensity   the substances merged into one false-colour image (NNLS)
    MCR-ALS         the same, but with the component spectra refined to the sample
    per-pixel pie   each pixel's composition as a pie glyph; background pixels grey
    composition     overall ratio among the (non-background) substances

Background (a blank / BLK class) is unmixed as its own component, so pixels the
substances don't explain fall to "no hit (background)". UI-agnostic (numpy/scipy).
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
    comps: list                  # ALL reference classes, background last
    bg_mask: np.ndarray          # (K,) True where the class is background/blank
    nonbg: list                  # indices of the non-background substances
    wn: np.ndarray
    coords: np.ndarray           # (n_pix, 2)
    A: np.ndarray                # (n_pix, K) NNLS abundance (all classes)
    A_mcr: np.ndarray            # (n_pix, K) MCR-ALS abundance, or None
    frac: np.ndarray             # (n_pix, K) NNLS composition, rows sum to 1
    ratio_nb: np.ndarray         # (n_pix, Knb) composition among non-bg (rows sum 1)
    hit: np.ndarray              # (n_pix,) True where a substance (not bg) dominates
    reliab: np.ndarray           # (n_pix,) reconstruction R²
    sam: np.ndarray              # (n_pix,) spectral angle (deg)
    templates: np.ndarray        # (K, n_feat) NNLS templates
    templates_mcr: np.ndarray    # (K, n_feat) MCR-refined spectra, or None
    meas_mean: np.ndarray
    recon_mean: np.ndarray
    n_pixels: int
    hit_frac: float              # fraction of pixels that are a substance (not bg)
    mean_ratio: np.ndarray       # (Knb,) mean non-bg composition over hit pixels
    dominant: str
    mean_r2: float


def _templates(data_dir, baseline):
    """Mean spectrum per class (background included, kept last), plus the axis."""
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
        n = np.linalg.norm(S, axis=1, keepdims=True)
        S = S / np.where(n > 0, n, 1.0)
        if progress:
            progress(f"MCR-ALS — iteration {it + 1}/{n_iter}")
    return C, S


def unmix_map(data_dir, test_path, method="nnls", baseline=True, trim=None,
              min_frac=0.05, progress=None) -> UnmixResult:
    """Unmix ``test_path`` against the substances in ``data_dir`` (background
    included). Always computes the NNLS solution; also runs MCR-ALS so the RGB
    views can be compared. ``min_frac`` sets how much non-background abundance a
    pixel needs to count as a hit rather than background."""
    names, wn, means = _templates(data_dir, baseline)
    wn_u, cube_u, _mean_u, coord = load_map(test_path)

    if trim is not None:
        lo, hi = trim
        m = (wn >= lo) & (wn <= hi)
        if m.sum() >= 10:
            means = means[:, m]; cube_u = cube_u[:, m]; wn = wn[m]

    if progress:
        progress("preprocessing spectra")
    templates = preprocess(means, do_baseline=baseline)
    X = preprocess(cube_u, do_baseline=baseline)
    K = len(names)
    bg_mask = np.array([is_blank(c) for c in names])
    nonbg = [i for i in range(K) if not bg_mask[i]]

    # per-pixel NNLS against all classes (background included)
    A = np.zeros((len(X), K))
    for i, y in enumerate(X):
        A[i], _ = nnls(templates.T, y)
        if progress and i % 300 == 0:
            progress(f"NNLS unmixing — pixel {i}/{len(X)}")

    # MCR-ALS refinement (seeded from the references) for the second RGB view
    A_mcr = templates_mcr = None
    if method != "nnls-only":
        if progress:
            progress("MCR-ALS refining component spectra")
        A_mcr, templates_mcr = _mcr_als(X, templates, progress=progress)

    recon = A @ templates
    ss_res = np.sum((X - recon) ** 2, axis=1)
    ss_tot = np.sum((X - X.mean(axis=1, keepdims=True)) ** 2, axis=1)
    reliab = np.clip(1.0 - np.divide(ss_res, ss_tot, out=np.ones_like(ss_res),
                                     where=ss_tot > 0), 0.0, 1.0)
    dot = np.sum(X * recon, axis=1)
    nn = np.linalg.norm(X, axis=1) * np.linalg.norm(recon, axis=1)
    cos = np.divide(dot, nn, out=np.zeros_like(dot), where=nn > 0)
    sam = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    tot = A.sum(axis=1, keepdims=True)
    frac = np.divide(A, tot, out=np.zeros_like(A), where=tot > 0)

    # non-background composition + hit test (a substance, not background, dominates)
    Anb = A[:, nonbg]
    nb_tot = Anb.sum(axis=1, keepdims=True)
    ratio_nb = np.divide(Anb, nb_tot, out=np.zeros_like(Anb), where=nb_tot > 0)
    nonbg_share = frac[:, nonbg].sum(axis=1)            # how much is substance vs bg
    hit = nonbg_share >= max(min_frac, 0.5)             # substance dominates the pixel
    hit_frac = float(hit.mean())
    mean_ratio = (ratio_nb[hit].mean(axis=0) if hit.any()
                  else ratio_nb.mean(axis=0))
    dominant = [names[i] for i in nonbg][int(mean_ratio.argmax())] if nonbg else names[0]

    return UnmixResult(
        comps=names, bg_mask=bg_mask, nonbg=nonbg, wn=wn, coords=coord,
        A=A, A_mcr=A_mcr, frac=frac, ratio_nb=ratio_nb, hit=hit,
        reliab=reliab, sam=sam, templates=templates, templates_mcr=templates_mcr,
        meas_mean=X.mean(axis=0), recon_mean=recon.mean(axis=0),
        n_pixels=len(X), hit_frac=hit_frac, mean_ratio=mean_ratio,
        dominant=dominant, mean_r2=float(reliab.mean()))


if __name__ == "__main__":
    import sys
    r = unmix_map(sys.argv[1], sys.argv[2])
    nb = [r.comps[i] for i in r.nonbg]
    print("dominant:", r.dominant, "| hit%:", round(r.hit_frac * 100), "| mean R²:",
          round(r.mean_r2, 3))
    print("mean ratio:", {n: round(float(v), 3) for n, v in zip(nb, r.mean_ratio)})
