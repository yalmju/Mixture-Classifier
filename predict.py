"""predict.py — apply the reference set to an unknown sample and read its
component ratio.

The simplified "Map tool": you already organised references (Samples) and can
train on them (Model); here you load ONE unknown sample map and get the
composition of the known substances in it. The ratio is the PER-PIXEL NNLS
composition averaged over the map — this recovers minor components that a single
mean spectrum buries — with a per-pixel dominant-component map and a per-pixel
vote for robust presence detection.

UI-agnostic (numpy / scipy), so the Qt tab just draws the numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

from real_data import load_map, per_pixel_vote
from dataset import discover_dataset, is_blank
from sers_mixture import preprocess, als_baseline
from calibration import calibrate, quantify
from io_utils import load_calibration_csv


@dataclass
class PredictResult:
    comps: list                 # reference substances (non-blank)
    ratio: dict                 # name -> proportion, PER-PIXEL NNLS averaged
    ratio_mean: dict            # name -> proportion from the mean spectrum (compare)
    detected: list              # component names present (per-pixel vote)
    wn: np.ndarray              # wavenumber axis
    mean_spectrum: np.ndarray   # preprocessed unknown mean spectrum
    templates: np.ndarray       # (K, n_feat) preprocessed reference templates
    coords: np.ndarray          # (n_pixels, 2) pixel X/Y
    pp: np.ndarray              # (n_pixels, K) per-pixel NNLS proportions
    pp_dominant: np.ndarray     # (n_pixels,) dominant component index per pixel
    n_pixels: int
    calibrated: bool = False    # True if a dilution-series calibration was applied
    conc: np.ndarray = None     # (n_pixels, K) per-pixel absolute concentration (M)
    conc_avg: np.ndarray = None  # (K,) map-mean absolute concentration (M)
    pp_theta: np.ndarray = None  # (n_pixels,) total surface coverage Σθ per pixel


def _baseline_only(spectra, do_baseline=True):
    """ALS baseline removal + clip, but NO L2 normalisation — keeps the absolute
    intensity that quantification needs (unlike the ratio path, which L2-norms)."""
    X = np.asarray(spectra, float)
    if do_baseline:
        X = np.stack([y - als_baseline(y) for y in X])
    return np.clip(X, 0.0, None)


def predict_sample(data_dir, sample_path, threshold=0.30, baseline=True,
                   trim=None, calib_path=None) -> PredictResult:
    """Load the reference substances from ``data_dir`` (Samples grouping) and the
    unknown map at ``sample_path``, and return the estimated composition — from
    per-pixel NNLS averaged over the map. If ``calib_path`` (a dilution-series
    CSV) is given, also recover per-pixel ABSOLUTE concentration (M) via Langmuir
    calibration."""
    groups = discover_dataset(data_dir)
    comps = [c for c, _ in groups if not is_blank(c)]
    if not comps:
        raise FileNotFoundError(
            f"no substance references found in {data_dir} (need at least one "
            "non-blank class).")
    path_of = {c: [p for _b, p, _r in maps] for c, maps in groups}

    # reference templates: mean spectrum per substance (across its batch maps)
    means, wn = [], None
    for c in comps:
        cbs = []
        for p in path_of[c]:
            wn, cube, _m, _c = load_map(p)
            cbs.append(cube)
        means.append(np.vstack(cbs).mean(axis=0))
    means = np.array(means)

    # unknown sample
    wn_u, cube_u, mean_u, coord = load_map(sample_path)

    if trim is not None:
        lo, hi = trim
        m = (wn >= lo) & (wn <= hi)
        if m.sum() >= 10:
            means = means[:, m]; mean_u = mean_u[m]; cube_u = cube_u[:, m]; wn = wn[m]

    pures = preprocess(means, do_baseline=baseline)              # (K, n_feat)
    Xp = preprocess(cube_u, do_baseline=baseline)               # (n_pix, n_feat)
    mean_f = preprocess(mean_u[None, :], do_baseline=baseline)[0]

    # ---- per-pixel NNLS composition (the robust ratio) ----
    K = len(comps)
    pp = np.zeros((len(Xp), K))
    for i, yv in enumerate(Xp):
        B, _ = nnls(pures.T, yv)
        s = B.sum()
        pp[i] = B / s if s > 0 else 0.0
    ratio_pp = pp.mean(axis=0)
    pp_dominant = pp.argmax(axis=1)

    # mean-spectrum NNLS (for comparison — buries minor components)
    Bm, _ = nnls(pures.T, mean_f); sm = Bm.sum()
    ratio_mean_v = Bm / sm if sm > 0 else Bm

    # presence detection: per-pixel vote (falls back to the strongest)
    voted = per_pixel_vote(cube_u, pures, comps, pix_thr=threshold if threshold < 0.5 else 0.15)
    detected = sorted({comps[i] for i in voted}) or [comps[int(ratio_pp.argmax())]]

    # ---- optional: per-pixel ABSOLUTE concentration via Langmuir calibration ----
    calibrated, conc, conc_avg = False, None, None
    if calib_path:
        axis_c, names_c, dils = load_calibration_csv(calib_path)
        cidx = {n: k for k, n in enumerate(names_c)}
        missing = [c for c in comps if c not in cidx]
        if missing:
            raise ValueError(f"calibration is missing substances {missing} "
                             f"(it has {names_c}); calibrate the same references.")
        aligned = []                                    # per-comp (C_grid, spectra)
        for c in comps:
            Cg, specs = dils[cidx[c]]
            specs = np.asarray(specs, float)
            if trim is not None:
                lo, hi = trim; mc = (axis_c >= lo) & (axis_c <= hi)
                if mc.sum() >= 10:
                    specs = specs[:, mc]
            aligned.append((Cg, _baseline_only(specs, baseline)))
        if aligned[0][1].shape[1] != pures.shape[1]:
            raise ValueError(
                "calibration axis does not match the reference maps "
                f"({aligned[0][1].shape[1]} vs {pures.shape[1]} points) — the "
                "calibration must be on the same instrument axis.")
        calib = calibrate(aligned, pures, comps)        # unit-norm templates = pures
        Xq = _baseline_only(cube_u, baseline)           # absolute-intensity pixels
        qs = [quantify(Xq[i], pures, calib) for i in range(len(Xq))]
        conc = np.array([q["C"] for q in qs])
        pp_theta = np.array([q["theta_total"] for q in qs])
        conc_avg = quantify(_baseline_only(mean_u[None, :], baseline)[0],
                            pures, calib)["C"]
        calibrated = True

    return PredictResult(
        comps=comps,
        ratio={c: float(ratio_pp[i]) for i, c in enumerate(comps)},
        ratio_mean={c: float(ratio_mean_v[i]) for i, c in enumerate(comps)},
        detected=detected, wn=wn, mean_spectrum=mean_f, templates=pures,
        coords=coord, pp=pp, pp_dominant=pp_dominant, n_pixels=len(cube_u),
        calibrated=calibrated, conc=conc, conc_avg=conc_avg, pp_theta=pp_theta)


if __name__ == "__main__":
    import sys
    r = predict_sample(sys.argv[1], sys.argv[2])
    print("detected:", r.detected)
    print("per-pixel ratio:", {k: round(v, 3) for k, v in r.ratio.items()})
    print("mean-spec ratio:", {k: round(v, 3) for k, v in r.ratio_mean.items()})
