"""bench_preprocessing_ablation.py — does a different spectral preprocessing beat the
adopted one?

    python documentation/scripts/bench_preprocessing_ablation.py [--modes a,b,c] [--quick]

Runs the whole leave-one-condition-out benchmark once per feature mode and prints the
methods x modes table of mean composition deviation (%p), so the choice is made on
measured numbers rather than on what is conventional.

Why this and not another method
-------------------------------
The 2026-08-19 run says the method is no longer the lever: response factors bought
0.2 %p, MCR-ALS refinement bought nothing, and the flexible learners (RF, 1D-CNN) came
in WORSE than plain NNLS at 61 conditions. Everything from VIP band to MLP lands in a
16-20 %p band against an 11.6 %p reproducibility floor. What has never been varied is
what the models read: every run so far used log1p of the raw spectrum.

SNV and a first derivative are the standard chemometrics answers to exactly the two
nuisances this substrate has - multiplicative scatter (gain CV 37-50%) and additive
baseline. The sweat paper this project compares itself to used first derivative plus
standardisation. If they help, they help every learned method at once and cost nothing
at inference; if they do not, that is a real result and the ceiling is the measurement.

The unmixing methods (VIP band, NNLS, NNLS+resp, MCR-ALS) are NOT affected: they always
receive the untouched spectra, so their column is a fixed reference across modes.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataset import load_mixture_list, load_preprocess   # noqa: E402
from dl_model import benchmark_loo, benchmark_summary     # noqa: E402
from real_data import PEST_DEFAULT                        # noqa: E402

MODES = ["log1p_raw", "legacy_l2", "snv", "deriv1", "snv_deriv1"]
LEARNED = ("pls", "rf", "cnn", "mlp")      # the only ones a feature mode can move


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=PEST_DEFAULT)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--cut", type=float, default=15.0)
    ap.add_argument("--quick", action="store_true",
                    help="learned methods only, fewer epochs — a first look in minutes")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cfg = load_preprocess(a.data_dir)
    items = load_mixture_list(a.data_dir)
    if not items:
        raise SystemExit(f"no mixtures.json under {a.data_dir}")
    methods = LEARNED if a.quick else ("null", "band", "nnls", "nnls_rf", "mcr",
                                       "pls", "rf", "cnn", "mlp")
    print(f"{len(items)} maps · baseline={cfg['baseline']} · trim={cfg['trim']} · "
          f"methods={len(methods)} · modes={a.modes}\n")

    table = {}
    for mode in a.modes.split(","):
        print(f"--- {mode} ---", flush=True)
        bench = benchmark_loo(a.data_dir, items, baseline=cfg["baseline"], trim=cfg["trim"],
                              methods=methods, epochs=120 if a.quick else 350,
                              rf_max_features="sqrt", px_per_map=0, feature_mode=mode,
                              progress=lambda s: print("   ·", s, flush=True)
                              if "1/" in s else None)
        table[mode] = benchmark_summary(bench, a.cut)

    modes = list(table)
    n = table[modes[0]][methods[0]]["n"]
    print(f"\n=== mean composition deviation (%p), {n} conditions — lower is better ===")
    hdr = f"{'method':10s}" + "".join(f"{m:>13s}" for m in modes)
    print(hdr); print("-" * len(hdr))
    for meth in methods:
        row = f"{meth:10s}"
        best = min(table[m][meth]["mean_dev_pp"] for m in modes)
        for m in modes:
            v = table[m][meth]["mean_dev_pp"]
            row += f"{v:12.1f}" + ("*" if abs(v - best) < 1e-9 else " ")
        print(row)
    print("\n* = best mode for that method. The four unmixing methods read the untouched")
    print("  spectra, so their row should be constant — any drift there is a bug.")
    print(f"\naccuracy ≤{a.cut:g} %p")
    for meth in methods:
        print(f"  {meth:10s}" + "".join(
            f"{table[m][meth].get('acc_at_cut', float('nan'))*100:12.0f}%" for m in modes))

    if a.out:
        import csv
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["method", "feature_mode", "mean_deviation_pp", "rmse_pp",
                        "logratio", f"accuracy_within_{a.cut:g}pp", "n"])
            for meth in methods:
                for m in modes:
                    e = table[m][meth]
                    w.writerow([meth, m, f"{e['mean_dev_pp']:.3f}", f"{e['rmse_pp'][0]:.3f}",
                                f"{e['logratio'][0]:.4f}",
                                f"{e.get('acc_at_cut', float('nan')):.4f}", e["n"]])
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
