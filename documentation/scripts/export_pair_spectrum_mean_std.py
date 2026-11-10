"""Export measured pixel-spectrum mean and SD for the identical-ratio pair."""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from dl_model import nnls_hit_spectra
from unmix import _baseline_removed, _l2

CONDITIONS = ["DQ12-TB6-TH3.csv", "DQ24-TB12-TH6.csv"]


def summarize(pure, path):
    wn, cube, _ = nnls_hit_spectra(
        pure, path, baseline=True, trim=None, min_frac=.15
    )
    raw = np.asarray(cube, float)
    if raw.ndim != 2:
        raw = raw.reshape(-1, raw.shape[-1])
    corrected = _l2(_baseline_removed(raw, True))
    raw_scale = np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    raw_norm = raw / raw_scale
    return np.asarray(wn), {
        "raw_l2_mean": raw_norm.mean(axis=0),
        "raw_l2_std": raw_norm.std(axis=0, ddof=1),
        "preprocessed_mean": corrected.mean(axis=0),
        "preprocessed_std": corrected.std(axis=0, ddof=1),
        "n_pixels": raw.shape[0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pure = os.path.join(a.db, "Pure")
    folder = os.path.join(a.db, "Ratio", "260814_mixture_final")
    results = []
    for name in CONDITIONS:
        print("processing", name, flush=True)
        results.append(summarize(pure, os.path.join(folder, name)))
    wn = results[0][0]
    if any(len(w) != len(wn) or not np.allclose(w, wn) for w, _ in results[1:]):
        raise ValueError("Raman axes differ between conditions")
    labels = ["DQ12_TBZ6_THI3", "DQ24_TBZ12_THI6"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        header = ["raman_shift_cm-1"]
        for label in labels:
            header += [f"{label}_raw_l2_mean", f"{label}_raw_l2_std",
                       f"{label}_preprocessed_mean", f"{label}_preprocessed_std"]
        w.writerow(header)
        for i, shift in enumerate(wn):
            row = [shift]
            for _, values in results:
                row += [values["raw_l2_mean"][i], values["raw_l2_std"][i],
                        values["preprocessed_mean"][i], values["preprocessed_std"][i]]
            w.writerow(row)
    meta = os.path.splitext(a.out)[0] + "_metadata.csv"
    with open(meta, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["condition", "n_NNLS_hit_pixels", "ddof"])
        for label, (_, values) in zip(labels, results):
            w.writerow([label, values["n_pixels"], 1])
    print(a.out, flush=True); print(meta, flush=True)


if __name__ == "__main__":
    main()
