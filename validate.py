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


def _decode_amount(s):
    """A filename amount code → value. Leading-zero codes are decimals: '1'→1, '01'→0.1,
    '001'→0.01; plain integers keep their value ('3'→3, '10'→10)."""
    return int(s) / 10 ** (len(s) - 1) if s.startswith("0") and len(s) > 1 else float(int(s))


def _match_ref(tok, ref_names):
    """Map a filename letter-token to a reference substance: exact match, else a unique
    prefix (so 'TH' → 'THI'). None if ambiguous / no match."""
    for r in ref_names:
        if tok == r.lower():
            return r
    cand = [r for r in ref_names if len(tok) >= 2 and r.lower().startswith(tok)]
    return cand[0] if len(cand) == 1 else None


def parse_mixture_label(basename, ref_names, base=None):
    """Read a true ratio from a filename by scanning <letters><digits> tokens and
    pairing each recognised substance with the number that follows it, e.g.
    'TBZ1DQ3' → {TBZ:1, DQ:3}, 'THI001' → {THI:0.01}, 'DQ1TH3' → {DQ:1, THI:3}. A
    ``base`` dict (the fixed components, e.g. {TBZ:1, DQ:1}) is applied first and any
    named substance overrides it — so 'THI01' → {TBZ:1, DQ:1, THI:0.1}. Also accepts
    the older 'A_B_1to3' form. Returns None if nothing recognised."""
    low = basename.lower()
    out = dict(base) if base else {}
    matched = False
    # older explicit 'DQ_TBZ_1to3' style first (compounds then a NtoM ratio group)
    present = [n for n in ref_names if n.lower() in low]
    m = re.search(r"(\d+(?:\s*(?:to|[_\-x:])\s*\d+)+)", low)
    if len(present) >= 2 and m:
        order = sorted(present, key=lambda n: low.find(n.lower()))
        nums = [_decode_amount(x) for x in re.findall(r"\d+", m.group(1))]
        if len(nums) == len(order):
            for c, v in zip(order, nums):
                out[c] = v
            return out
    # token style: pair each substance with the number immediately after it
    toks = re.findall(r"[a-z]+|\d+", low)
    i = 0
    while i < len(toks):
        if toks[i].isalpha():
            c = _match_ref(toks[i], ref_names)
            if c and i + 1 < len(toks) and toks[i + 1].isdigit():
                out[c] = _decode_amount(toks[i + 1]); matched = True; i += 2; continue
        i += 1
    return out if matched else None


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
