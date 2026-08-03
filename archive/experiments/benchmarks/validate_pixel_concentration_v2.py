"""Validate the v2 concentration head on calibration spectra and one real SERS map."""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from io_utils import load_calibration_csv
from pixel_surface import apply_pixel_concentration, load_pixel_surface
from unmix import unmix_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acf_root")
    ap.add_argument("--model", default=str(ROOT / "output/models/pixel_surface_v2.psm"))
    args = ap.parse_args(); root = pathlib.Path(args.acf_root)
    model = load_pixel_surface(args.model)
    cal = root / "Pest/Standard/260729/calibration_spectra.csv"
    wn, names, dils = load_calibration_csv(cal); calibration = {}
    for name, (truth_M, spectra) in zip(names, dils):
        pred, subs, meta = apply_pixel_concentration(model, wn, spectra)
        j = subs.index(name); truth = np.asarray(truth_M) * 1e6; p = pred[:, j]
        valid = np.isfinite(p)
        calibration[name] = {
            "valid_points": int(valid.sum()),
            "log_order_correlation": float(np.corrcoef(np.log10(truth[valid]), np.log10(p[valid]))[0, 1]),
            "mae_decades": float(np.mean(np.abs(np.log10(p[valid]) - np.log10(truth[valid])))),
        }
    test = root / "Pest/Mixtures/260723/260723_mixture(bg).csv"
    r = unmix_map(root / "Pure", test, method="dlpx", baseline=False,
                  trim=(500, 2500), min_frac=.15, hit_mode="threshold", dl_model=model)
    subs = [r.comps[i] for i in r.nonbg]
    result = {"scope": "order-of-magnitude, semi-quantitative",
              "calibration_range_uM": [0.1, 1000.0], "calibration": calibration,
              "real_260723": {"hit_fraction": r.hit_frac,
                  "composition": dict(zip(subs, map(float, r.mean_ratio))),
                  "median_uM": dict(zip(subs, map(float, r.conc_median * 1e6))),
                  "p10_uM": dict(zip(subs, map(float, r.conc_p10 * 1e6))),
                  "p90_uM": dict(zip(subs, map(float, r.conc_p90 * 1e6))),
                  "median_ci95_low_uM": dict(zip(subs, map(float, r.conc_ci_low * 1e6))),
                  "median_ci95_high_uM": dict(zip(subs, map(float, r.conc_ci_high * 1e6))),
                  "median_se_uM": dict(zip(subs, map(float, r.conc_se * 1e6))),
                  "ood_fraction": dict(zip(subs, map(float, r.conc_ood[r.hit].mean(0))))}}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()