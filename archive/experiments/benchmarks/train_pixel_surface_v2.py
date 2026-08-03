"""Train the experimental per-pixel surface-composition model.

Example:
  python experiments/benchmarks/train_pixel_surface_v2.py S:/.../ACF_PEST_DB
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset import load_preprocess
from pixel_surface import save_pixel_surface, train_pixel_surface, train_concentration_head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acf_root", help="ACF_PEST_DB folder")
    ap.add_argument("--output", default=str(ROOT / "output/models/pixel_surface_v2.psm"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--conc-epochs", type=int, default=70)
    args = ap.parse_args()
    pure = os.path.join(args.acf_root, "Pure")
    cfg = load_preprocess(pure)
    trim = cfg.get("trim") or (400.0, 1800.0)
    lo, hi = max(400.0, float(trim[0])), min(1800.0, float(trim[1]))
    model = train_pixel_surface(
        pure, subs=["DQ", "TBZ", "THI"], lo=lo, hi=hi,
        baseline=bool(cfg.get("baseline", True)), per_map=100,
        n_synth=8000, epochs=args.epochs, seed=args.seed, progress=print)
    calibration = args.calibration or os.path.join(
        args.acf_root, "Pest", "Standard", "260729", "calibration_spectra.csv")
    model = train_concentration_head(
        model, calibration, pure, n_train=20000, n_val=4000,
        epochs=args.conc_epochs, seed=args.seed, progress=print)
    out = pathlib.Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    save_pixel_surface(model, out)
    print(f"saved {out}")
    print(f"synthetic validation TV error: {model['synthetic_val_tv']:.1%}")
    print(f"concentration validation log-MAE: {model['uM']['training']['val_present_log_mae']:.3f} decades")


if __name__ == "__main__":
    main()