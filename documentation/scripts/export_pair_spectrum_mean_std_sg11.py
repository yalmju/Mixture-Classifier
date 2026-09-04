"""Export pixelwise SG(11,3)-smoothed measured spectra as mean and SD."""
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

CONDITIONS = ["DQ12-TB6-TH3.csv", "DQ24-TB12-TH6.csv"]
LABELS = ["DQ12_TBZ6_THI3", "DQ24_TBZ12_THI6"]


def summarize(pure, path):
    wn, cube, _ = nnls_hit_spectra(
        pure, path, baseline=True, trim=None, min_frac=.15
    )
    raw = np.asarray(cube, float).reshape(-1, np.asarray(cube).shape[-1])
    raw_norm = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    preprocessed = _l2(_baseline_removed(raw, True))
    raw_sg = savgol_filter(raw_norm, window_length=11, polyorder=3, axis=1,
                           mode="interp")
    pre_sg = savgol_filter(preprocessed, window_length=11, polyorder=3, axis=1,
                           mode="interp")
    return np.asarray(wn), {
        "raw_l2_sg11_mean": raw_sg.mean(axis=0),
        "raw_l2_sg11_std": raw_sg.std(axis=0, ddof=1),
        "preprocessed_sg11_mean": pre_sg.mean(axis=0),
        "preprocessed_sg11_std": pre_sg.std(axis=0, ddof=1),
        "n_pixels": raw.shape[0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pure = os.path.join(a.db, "Pure")
    folder = os.path.join(a.db, "Ratio", "260814_mixture_final")
    results = [summarize(pure, os.path.join(folder, name)) for name in CONDITIONS]
    wn = results[0][0]
    if any(len(w) != len(wn) or not np.allclose(w, wn) for w, _ in results[1:]):
        raise ValueError("Raman axes differ between conditions")
    keys = ["raw_l2_sg11_mean", "raw_l2_sg11_std",
            "preprocessed_sg11_mean", "preprocessed_sg11_std"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["raman_shift_cm-1"] +
                   [f"{label}_{key}" for label in LABELS for key in keys])
        for i, shift in enumerate(wn):
            w.writerow([shift] + [values[key][i]
                                  for _, values in results for key in keys])
    meta = os.path.splitext(a.out)[0] + "_metadata.csv"
    with open(meta, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["condition", "n_NNLS_hit_pixels", "sg_window",
                    "sg_polyorder", "sg_mode", "std_ddof"])
        for label, (_, values) in zip(LABELS, results):
            w.writerow([label, values["n_pixels"], 11, 3, "interp", 1])
    print(a.out); print(meta)


if __name__ == "__main__":
    main()
