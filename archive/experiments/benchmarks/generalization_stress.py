"""Stress-test the deployed NNLS-hit MLP under synthetic acquisition shifts.

This is not a substitute for a genuinely independent day/substrate batch. It measures
controlled sensitivity to gain, baseline, noise and spectral registration changes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def perturb(x, condition, rng):
    x = np.asarray(x, float).copy()
    if condition == "clean":
        return x
    if condition == "gain_0.5x":
        return x * 0.5
    if condition == "gain_2x":
        return x * 2.0
    if condition == "baseline_drift":
        ramp = np.linspace(0.25, 1.0, x.shape[1])[None, :]
        return x + 0.15 * np.max(x, axis=1, keepdims=True) * ramp
    if condition == "noise_2pct":
        sd = 0.02 * np.max(x, axis=1, keepdims=True)
        return np.clip(x + rng.normal(0, sd, x.shape), 0, None)
    if condition == "shift_plus3":
        return np.roll(x, 3, axis=1)
    raise ValueError(condition)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", default="output/generalization_stress.json")
    args = ap.parse_args()

    from dataset import load_mixture_list
    from dl_model import apply_model_pixels, apply_uM_pixels, load_model
    from unmix import unmix_map

    model = load_model(args.model)
    items = load_mixture_list(args.data_dir, "test")
    conditions = ("clean", "gain_0.5x", "gain_2x", "baseline_drift",
                  "noise_2pct", "shift_plus3")
    rows = {c: [] for c in conditions}
    rng = np.random.default_rng(20260803)

    for item in items:
        path, ratio = item[0], item[1]
        conc = item[2] if len(item) > 2 else {}
        base = unmix_map(args.data_dir, path, method="nnls", baseline=False,
                         trim=(500, 2500), min_frac=0.15)
        names = [base.comps[j] for j in base.nonbg]
        true = np.array([ratio.get(s, 0.0) for s in names], float)
        true /= true.sum() + 1e-12
        stock = np.array([conc.get(s, 0.0) * 1e6 for s in names], float)
        dilution = max(1, np.count_nonzero(stock > 0)); true_um = stock / dilution
        for condition in conditions:
            x = perturb(base.spectra, condition, rng)
            A = np.zeros((len(x), len(base.comps)))
            for i, y in enumerate(l2(x)):
                A[i], _ = nnls(base.templates.T, y)
            frac = A / (A.sum(1, keepdims=True) + 1e-12)
            hit = frac[:, base.nonbg].sum(1) >= 0.15
            if not hit.any():
                pred = np.full(len(names), np.nan); med_um = np.full(len(names), np.nan)
            else:
                pred = apply_model_pixels(model, base.wn, x[hit]).mean(0)
                um, unames = apply_uM_pixels(model, base.wn, x[hit])
                med_um = np.array([np.median(um[:, unames.index(s)]) for s in names])
            inter = np.count_nonzero(hit & base.hit); union = np.count_nonzero(hit | base.hit)
            pos = true_um > 0
            rows[condition].append({
                "composition_error": float(0.5 * np.nansum(np.abs(pred - true))),
                "hit_fraction": float(hit.mean()),
                "hit_jaccard_vs_clean": float(inter / max(1, union)),
                "log10_uM_mae_present": float(np.nanmean(
                    np.abs(np.log10(med_um[pos] + 1) - np.log10(true_um[pos] + 1)))),
            })

    summary = {}
    for condition, rr in rows.items():
        summary[condition] = {k: float(np.mean([r[k] for r in rr])) for k in rr[0]}
    result = {
        "scope": "synthetic acquisition-shift stress test on six independent Ratio_mix maps",
        "limitation": "No labeled independent day/substrate mixture batch was available.",
        "conditions": summary,
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
