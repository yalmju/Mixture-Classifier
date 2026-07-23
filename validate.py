"""validate.py — validate the pure-reference unmixing against KNOWN-ratio mixtures.

Pure references decompose any mixture, but they can't tell you whether the reported
ratio is the true SOLUTION ratio: a strongly-adsorbing / high-response substance
(e.g. THI) dominates the surface signal even in a balanced mixture. Measuring a few
known-ratio mixtures lets us separate the two explanations and correct for it:

    observed_i  =  r_i · t_i / Σ_j (r_j · t_j)

where t_i is the true (solution) fraction and r_i the RESPONSE FACTOR (relative
sensitivity) of substance i. Given several mixtures we recover r (up to a global
scale) and can convert any observed surface ratio back to a solution ratio.

UI-agnostic (numpy only, plus unmix_map for the per-map decomposition).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from unmix import unmix_map


@dataclass
class ValidateResult:
    names: list                    # non-background substances (columns)
    rows: list                     # per mixture: {path, label, true(dict), obs(dict)}
    response: dict                 # name -> response factor (min normalised to 1)
    corrected: list                # per mixture: corrected (solution) fractions dict
    ref: str = ""                  # substance the factors are anchored to (r≈1)


def parse_mixture_label(basename, ref_names):
    """Read a true ratio from a filename, e.g. 'mix_DQ_TBZ_1to3' → {DQ:0.25, TBZ:0.75}.
    Compounds are matched (in order of appearance) against ``ref_names``; the ratio is
    the first 'AtoB(toC…)' / 'A_B' / 'A-B' number group. Returns None if unparseable."""
    low = basename.lower()
    present = sorted((low.find(n.lower()), n) for n in ref_names if n.lower() in low)
    comps = [n for idx, n in present if idx >= 0]
    if len(comps) < 2:
        return None
    m = re.search(r"(\d+(?:\s*(?:to|[_\-x:])\s*\d+)+)", low)
    if not m:
        return None
    nums = [float(x) for x in re.findall(r"\d+", m.group(1))]
    if len(nums) != len(comps):
        return None
    s = sum(nums) or 1.0
    return {c: nums[i] / s for i, c in enumerate(comps)}


def _response_factors(rows, names):
    """Relative response factor per substance from the observed-vs-true fractions.
    r_i / r_ref = geometric mean over mixtures of (obs_i/obs_ref)·(t_ref/t_i). The
    reference is the substance present in the most mixtures; factors are rescaled so
    the smallest is 1 (so r>1 means 'over-reported on the surface')."""
    counts = {n: sum(1 for r in rows if r["true"].get(n, 0) > 0 and r["obs"].get(n, 0) > 0)
              for n in names}
    ref = max(names, key=lambda n: counts[n])
    logr = {n: [] for n in names}
    for r in rows:
        obs, true = r["obs"], r["true"]
        oref, tref = obs.get(ref, 0.0), true.get(ref, 0.0)
        if oref <= 0 or tref <= 0:
            continue
        for n in names:
            oi, ti = obs.get(n, 0.0), true.get(n, 0.0)
            if oi > 0 and ti > 0:
                logr[n].append(np.log((oi / oref) * (tref / ti)))
    rf = {n: (float(np.exp(np.mean(logr[n]))) if logr[n] else 1.0) for n in names}
    mn = min(rf.values()) or 1.0
    return {n: rf[n] / mn for n in names}, ref


def correct_fractions(obs, response, names):
    """Convert an observed surface ratio to a solution ratio: divide each substance's
    abundance by its response factor and renormalise. ``obs`` is a dict or array."""
    v = np.array([obs.get(n, 0.0) if isinstance(obs, dict) else obs[i]
                  for i, n in enumerate(names)], float)
    r = np.array([response.get(n, 1.0) for n in names], float)
    c = np.divide(v, r, out=np.zeros_like(v), where=r > 0)
    s = c.sum()
    return c / s if s > 0 else c


def validate_mixtures(data_dir, items, method="nnls", baseline=True, trim=None,
                      progress=None) -> ValidateResult:
    """``items`` is a list of (map_path, true_dict) where true_dict maps substance →
    true fraction (need not be normalised). Each map is unmixed against the pure
    references; we collect the observed mean surface fraction, then fit the response
    factors and the corrected (solution) fractions."""
    rows, names = [], None
    for path, true in items:
        if progress:
            progress(f"unmixing {path}")
        r = unmix_map(data_dir, path, method=method, baseline=baseline, trim=trim,
                      hit_mode="auto", progress=progress)
        nb = [r.comps[i] for i in r.nonbg]
        names = nb
        obs = {nb[k]: float(r.mean_ratio[k]) for k in range(len(nb))}
        tot = sum(true.values()) or 1.0
        true_n = {k: v / tot for k, v in true.items()}
        rows.append({"path": path, "true": true_n, "obs": obs})
    if not names:
        raise ValueError("no mixtures to validate.")
    response, ref = _response_factors(rows, names)
    corrected = [dict(zip(names, correct_fractions(r["obs"], response, names)))
                 for r in rows]
    return ValidateResult(names=names, rows=rows, response=response,
                          corrected=corrected, ref=ref)
