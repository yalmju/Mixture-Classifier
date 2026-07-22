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
from sers_mixture import preprocess


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
    pp_dominant: np.ndarray     # (n_pixels,) dominant component index per pixel
    n_pixels: int


def predict_sample(data_dir, sample_path, threshold=0.30, baseline=True,
                   trim=None) -> PredictResult:
    """Load the reference substances from ``data_dir`` (Samples grouping) and the
    unknown map at ``sample_path``, and return the estimated composition — from
    per-pixel NNLS averaged over the map."""
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

    return PredictResult(
        comps=comps,
        ratio={c: float(ratio_pp[i]) for i, c in enumerate(comps)},
        ratio_mean={c: float(ratio_mean_v[i]) for i, c in enumerate(comps)},
        detected=detected, wn=wn, mean_spectrum=mean_f, templates=pures,
        coords=coord, pp_dominant=pp_dominant, n_pixels=len(cube_u))


if __name__ == "__main__":
    import sys
    r = predict_sample(sys.argv[1], sys.argv[2])
    print("detected:", r.detected)
    print("per-pixel ratio:", {k: round(v, 3) for k, v in r.ratio.items()})
    print("mean-spec ratio:", {k: round(v, 3) for k, v in r.ratio_mean.items()})
