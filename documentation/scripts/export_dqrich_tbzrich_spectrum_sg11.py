"""Export SG(11,3) pixel-spectrum mean and SD for DQ-/TBZ-rich cases."""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from dl_model import nnls_hit_spectra
from unmix import _baseline_removed, _l2

CASES = [("DQ-rich", "DQ12-TB3-TH3.csv", "DQ12_TBZ3_THI3"),
         ("TBZ-rich", "DQ6-TB24-TH6.csv", "DQ6_TBZ24_THI6")]


def summarize(pure, path):
    wn, cube, _ = nnls_hit_spectra(
        pure, path, baseline=True, trim=None, min_frac=.15
    )
    raw = np.asarray(cube, float).reshape(-1, np.asarray(cube).shape[-1])
    raw_norm = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    pre = _l2(_baseline_removed(raw, True))
    raw_sg = savgol_filter(raw_norm, 11, 3, axis=1, mode="interp")
    pre_sg = savgol_filter(pre, 11, 3, axis=1, mode="interp")
    return np.asarray(wn), raw_sg, pre_sg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pure = os.path.join(a.db, "Pure")
    folder = os.path.join(a.db, "Ratio", "260814_mixture_final")
    results = []
    for regime, filename, label in CASES:
        print("processing", filename, flush=True)
        wn, raw, pre = summarize(pure, os.path.join(folder, filename))
        results.append((regime, label, wn, raw, pre))
    axis = results[0][2]
    if any(len(wn) != len(axis) or not np.allclose(wn, axis)
           for _, _, wn, _, _ in results[1:]):
        raise ValueError("Raman axes differ")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        header = ["raman_shift_cm-1"]
        for _, label, _, _, _ in results:
            header += [f"{label}_raw_sg11_mean", f"{label}_raw_sg11_std",
                       f"{label}_preprocessed_sg11_mean",
                       f"{label}_preprocessed_sg11_std"]
        w.writerow(header)
        for i, shift in enumerate(axis):
            row = [shift]
            for _, _, _, raw, pre in results:
                row += [raw[:, i].mean(), raw[:, i].std(ddof=1),
                        pre[:, i].mean(), pre[:, i].std(ddof=1)]
            w.writerow(row)
    meta = os.path.splitext(a.out)[0] + "_metadata.csv"
    with open(meta, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["regime", "condition", "n_NNLS_hit_pixels", "sg_window",
                    "sg_polyorder", "sg_mode", "std_ddof"])
        for regime, label, _, raw, _ in results:
            w.writerow([regime, label, len(raw), 11, 3, "interp", 1])
    print(a.out, flush=True); print(meta, flush=True)


if __name__ == "__main__":
    main()
