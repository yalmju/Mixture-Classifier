"""Rank final low-concentration maps by NNLS THI overestimation."""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import physics_max_rescue_experiment as exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    folder = os.path.join(args.db, "Ratio", "260814_mixture_final")
    pure = os.path.join(args.db, "Pure")
    low, _ = exp.parse_folder(folder)
    rows = []
    for i, (path, conc) in enumerate(low, 1):
        print(f"{i}/{len(low)} {os.path.basename(path)}", flush=True)
        wn, raw, y, P, B, pred = exp.nnls_and_spectrum(pure, path)
        true = exp._ratio(np.asarray(conc, float))
        err = float(0.5 * np.abs(pred - true).sum() * 100)
        rows.append({
            "file": os.path.basename(path),
            "DQ_uM": conc[0], "TBZ_uM": conc[1], "THI_uM": conc[2],
            "true_DQ_pct": true[0] * 100, "true_TBZ_pct": true[1] * 100,
            "true_THI_pct": true[2] * 100,
            "nnls_DQ_pct": pred[0] * 100, "nnls_TBZ_pct": pred[1] * 100,
            "nnls_THI_pct": pred[2] * 100,
            "THI_overestimate_pct_point": (pred[2] - true[2]) * 100,
            "composition_error_pct": err,
        })
    rows.sort(key=lambda r: r["THI_overestimate_pct_point"], reverse=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("top", rows[0], flush=True)


if __name__ == "__main__":
    main()
