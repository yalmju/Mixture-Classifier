"""make_demo_data.py — generate a realistic demo dataset for UNMIXR so the whole
pipeline (Samples → Model → Quantify → Real data) can be exercised without real
measurements. Modelled on a typical run: pure references measured near-saturation
(~1 mM), a dilution series spanning 100 nM–1 mM with an LOD around ~10 µM, and a
few known-ratio test mixtures (1:1, 1:3, 3:1).

    python examples/make_demo_data.py [out_dir]      # default: examples/demo

Produces
    <out>/Reference/<Class>-<batch>.csv   pure maps: DQ, TBZ, THI, BLK × 3 batches
    <out>/samples.csv                     class / batch / role manifest
    <out>/preprocess.json                 ALS baseline · L2 · no trim
    <out>/calibration.csv                 dilution series (compound, conc_M, wn…)
    <out>/tests/mix_*.csv                 known-ratio mixture maps to load in Real data
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

WN = np.linspace(400.0, 1800.0, 300)                 # instrument axis
PEAKS = {                                            # substance → (main, secondary) cm⁻¹
    "DQ":  [(600, 1.0), (1180, 0.5), (1520, 0.3)],
    "TBZ": [(900, 1.0), (1010, 0.6), (1280, 0.4)],
    "THI": [(1300, 1.0), (760, 0.4), (1590, 0.5)],
}
GA = {"DQ": 0.9, "TBZ": 1.0, "THI": 0.8}             # brightness (response factor)
K = {"DQ": 1.0e4, "TBZ": 1.5e4, "THI": 0.8e4}        # Langmuir affinity (half-sat ~1/K)
NOISE = 0.03                                          # → LOD ≈ 10 µM at this SNR
SPOT_SIGMA = 0.18                                     # spot-to-spot intensity spread
CAL_CONC = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]   # 100 nM–1 mM


def _rng(seed):
    return np.random.default_rng(seed)


def _pure_shape(name):
    y = np.zeros_like(WN)
    for c, a in PEAKS[name]:
        y += a * np.exp(-0.5 * ((WN - c) / 11.0) ** 2)
    return y


def _baseline(scale, curve):
    # broad fluorescence-like background that ALS removes
    return scale * curve


def _theta(name, C):
    return K[name] * C / (1.0 + K[name] * C)          # Langmuir coverage


def _map_rows(spectra_fn, n=15, seed=0):
    rng = _rng(seed)
    bg_curve = np.exp(-(WN - 400) / 900.0)            # smooth decaying baseline
    rows = [["X num", str(n)], ["Y num", str(n)],
            ["X", "Y", *[f"{v:.1f}" for v in WN]]]
    for i in range(n):
        for j in range(n):
            spec = spectra_fn(i, j, rng)
            spec = spec + _baseline(0.15 + 0.05 * rng.standard_normal(), bg_curve)
            spec = spec + NOISE * rng.standard_normal(len(WN))
            rows.append([i, j, *[f"{max(v, 0):.4f}" for v in spec]])
    return rows


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def generate(out):
    ref = os.path.join(out, "Reference")
    os.makedirs(ref, exist_ok=True)

    # ---- pure reference maps (near-saturation, ~1 mM), 3 batches each ----
    manifest = [["file", "class", "batch", "role"]]
    for ci, name in enumerate(["DQ", "TBZ", "THI"]):
        shape = _pure_shape(name)
        amp = GA[name] * _theta(name, 1e-3)           # saturated
        for b in (1, 2, 3):
            gain = 1.0                                 # batch gain
            def fn(i, j, rng, shape=shape, amp=amp):
                return amp * shape * np.exp(SPOT_SIGMA * rng.standard_normal())
            _write(os.path.join(ref, f"{name}-{b}.csv"),
                   _map_rows(fn, seed=100 * ci + b))
            manifest.append([f"{name}-{b}.csv", name, b,
                             "test" if b == 3 else "train"])
    # blank / background maps
    for b in (1, 2, 3):
        def fn(i, j, rng):
            return np.zeros_like(WN)
        _write(os.path.join(ref, f"BLK-{b}.csv"), _map_rows(fn, seed=900 + b))
        manifest.append([f"BLK-{b}.csv", "BLK", b, "test" if b == 3 else "train"])

    with open(os.path.join(out, "samples.csv"), "w", newline="") as f:
        csv.writer(f).writerows(manifest)
    with open(os.path.join(out, "preprocess.json"), "w") as f:
        json.dump({"baseline": True, "deriv": 0, "norm": "l2", "trim": None}, f, indent=2)

    # ---- calibration: dilution series per compound (100 nM–1 mM) ----
    rng = _rng(7)
    cal = [["compound", "concentration_M", *[f"{v:.1f}" for v in WN]]]
    for name in ["DQ", "TBZ", "THI"]:
        shape = _pure_shape(name)
        for C in CAL_CONC:
            b = GA[name] * _theta(name, C)
            spec = b * shape + NOISE * rng.standard_normal(len(WN))
            cal.append([name, f"{C:.3e}", *[f"{max(v, 0):.4f}" for v in spec]])
    with open(os.path.join(out, "calibration.csv"), "w", newline="") as f:
        csv.writer(f).writerows(cal)

    # ---- known-ratio test mixtures (a central blob on a blank background) ----
    def mixture(a_name, b_name, ra, rb, total_C, seed):
        sa, sb = _pure_shape(a_name), _pure_shape(b_name)
        Ca, Cb = total_C * ra / (ra + rb), total_C * rb / (ra + rb)
        amp_a = GA[a_name] * _theta(a_name, Ca)
        amp_b = GA[b_name] * _theta(b_name, Cb)

        def fn(i, j, rng, n=15):
            d = ((i - 7) ** 2 + (j - 7) ** 2) ** 0.5
            if d > 5.5:
                return np.zeros_like(WN)               # background
            s = rng.uniform(0.7, 1.3)
            return s * (amp_a * sa + amp_b * sb)
        return _map_rows(fn, seed=seed)

    tests = os.path.join(out, "tests")
    for (ra, rb), tag in [((1, 1), "1to1"), ((1, 3), "1to3"), ((3, 1), "3to1")]:
        _write(os.path.join(tests, f"mix_DQ_TBZ_{tag}.csv"),
               mixture("DQ", "TBZ", ra, rb, 3e-5, seed=int(ra * 10 + rb)))
    return out


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "demo")
    generate(out)
    print("demo dataset written to", out)
