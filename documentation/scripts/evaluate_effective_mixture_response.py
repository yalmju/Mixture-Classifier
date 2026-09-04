"""Estimate leakage-free effective mixture-response coefficients.

This analysis treats the coefficients as empirical SERS response corrections, not
as adsorption equilibrium constants.  Every learned correction is fit on the four
outer training folds and applied only to the held-out fold.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import r2_score


ROOT = Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
RESULT = ROOT / "documentation/results/integrated_final_20260824/full/concentration_results.json"
BUNDLE = ROOT / "documentation/results/composition_all_no100/v2_search/bundle_89maps_24px.npz"
SUBS = ["DQ", "TBZ", "THI"]
BANDS = np.array([1570.0, 1270.0, 1367.0], float)
AXIS = 100.0 + 1.55 * np.arange(2001)


def jdefault(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    raise TypeError(type(x).__name__)


def inv_cal(intensity, ab):
    intensity = np.asarray(intensity, float)
    c = 10.0 ** ((intensity - ab[:, 1]) / ab[:, 0])
    return np.clip(c, 0.1, 100.0)


def concordance(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 2:
        return float("nan")
    cov = np.mean((y - y.mean()) * (p - p.mean()))
    den = y.var() + p.var() + (y.mean() - p.mean()) ** 2
    return float(2 * cov / den) if den > 0 else float("nan")


def safe_corr(fun, x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(fun(x, y).statistic)


def metrics(T, P, sel):
    T, P = np.asarray(T)[sel], np.asarray(P)[sel]
    present = T > 0
    t, p = T[present], P[present]
    rec = 100.0 * p / t
    euclid = np.sqrt(np.sum((P - T) ** 2, axis=1))
    false = P[~present]
    out = {
        "n_maps": int(len(T)),
        "n_present": int(len(t)),
        "recovery_mean_pct": float(np.mean(rec)),
        "recovery_median_pct": float(np.median(rec)),
        "recovery_q1_pct": float(np.quantile(rec, 0.25)),
        "recovery_q3_pct": float(np.quantile(rec, 0.75)),
        "median_abs_recovery_error_pct": float(np.median(np.abs(rec - 100.0))),
        "mean_abs_recovery_error_pct": float(np.mean(np.abs(rec - 100.0))),
        "within_80_120_pct": float(np.mean((rec >= 80.0) & (rec <= 120.0)) * 100.0),
        "within_70_130_pct": float(np.mean((rec >= 70.0) & (rec <= 130.0)) * 100.0),
        "within_1_5x_pct": float(np.mean((rec >= 100/1.5) & (rec <= 150.0)) * 100.0),
        "mae_uM": float(np.mean(np.abs(p - t))),
        "euclidean_mean_uM": float(np.mean(euclid)),
        "false_positive_mean_uM": float(np.mean(false)) if len(false) else 0.0,
        "pearson_r": safe_corr(pearsonr, np.log10(np.clip(t, .1, None)), np.log10(np.clip(p, .1, None))),
        "spearman_r": safe_corr(spearmanr, t, p),
        "r2_linear": float(r2_score(t, p)),
        "ccc": concordance(t, p),
    }
    out["components"] = {}
    for j, sub in enumerate(SUBS):
        ok = T[:, j] > 0
        rr = 100.0 * P[ok, j] / T[ok, j]
        out["components"][sub] = {
            "n": int(ok.sum()),
            "recovery_mean_pct": float(np.mean(rr)),
            "recovery_median_pct": float(np.median(rr)),
            "recovery_q1_pct": float(np.quantile(rr, .25)),
            "recovery_q3_pct": float(np.quantile(rr, .75)),
            "median_abs_recovery_error_pct": float(np.median(np.abs(rr - 100))),
            "within_80_120_pct": float(np.mean((rr >= 80) & (rr <= 120)) * 100),
            "mae_uM": float(np.mean(np.abs(P[ok, j] - T[ok, j]))),
            "pearson_r": safe_corr(pearsonr, np.log10(np.clip(T[ok, j], .1, None)), np.log10(np.clip(P[ok, j], .1, None))),
            "ccc": concordance(T[ok, j], P[ok, j]),
        }
    return out


def map_band_means(bundle):
    names = bundle["names"].astype(object)
    raw = np.expm1(np.clip(bundle["X"].astype(float), 0, None))
    result = {}
    for name in dict.fromkeys(names.tolist()):
        idx = np.where(names == name)[0]
        vals = []
        for centre in BANDS:
            mask = np.abs(AXIS - centre) <= 10.0
            vals.append(float(np.mean(np.max(raw[idx][:, mask], axis=1))))
        result[str(name)] = np.array(vals, float)
    return result


def fit_predict_corrections(base, ratio, T, folds):
    """Return cross-fitted correction candidates and fold coefficients."""
    n = len(T)
    pred = {
        "raw": base.copy(),
        "constant_factor": np.full_like(base, np.nan),
        "power_huber": np.full_like(base, np.nan),
        "competition_ridge": np.full_like(base, np.nan),
    }
    coefs = []
    for fold in sorted(np.unique(folds)):
        tr, te = folds != fold, folds == fold
        total_base = np.log10(np.clip(base.sum(1), .1, None))
        for j, sub in enumerate(SUBS):
            ok = tr & (T[:, j] > 0)
            x = np.log10(np.clip(base[ok, j], .1, None))
            y = np.log10(np.clip(T[ok, j], .1, None))

            log_factor = float(np.median(y - x))
            pred["constant_factor"][te, j] = base[te, j] * 10.0 ** log_factor
            coefs.append({"fold": int(fold), "component": sub, "model": "constant_factor",
                          "intercept_log10": log_factor, "slope_log10": 1.0,
                          "multiplicative_factor": 10.0 ** log_factor})

            hub = HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=1000).fit(x[:, None], y)
            xp = np.log10(np.clip(base[te, j], .1, None))
            pred["power_huber"][te, j] = 10.0 ** hub.predict(xp[:, None])
            coefs.append({"fold": int(fold), "component": sub, "model": "power_huber",
                          "intercept_log10": float(hub.intercept_), "slope_log10": float(hub.coef_[0]),
                          "multiplicative_factor": float(10.0 ** hub.intercept_)})

            Xtr = np.column_stack([x, total_base[ok], ratio[ok, j]])
            Xte = np.column_stack([xp, total_base[te], ratio[te, j]])
            ridge = Ridge(alpha=1.0).fit(Xtr, y)
            pred["competition_ridge"][te, j] = 10.0 ** ridge.predict(Xte)
            coefs.append({"fold": int(fold), "component": sub, "model": "competition_ridge",
                          "intercept_log10": float(ridge.intercept_),
                          "slope_log10": float(ridge.coef_[0]),
                          "total_slope": float(ridge.coef_[1]),
                          "ratio_slope": float(ridge.coef_[2]),
                          "multiplicative_factor": float(10.0 ** ridge.intercept_)})
    for key in pred:
        pred[key] = np.clip(pred[key], .001, 5000.0)
    return pred, coefs


def run(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    records = data["records"]
    ab = np.asarray(data["protocol"]["calibration_ab"], float)
    bundle = np.load(BUNDLE, allow_pickle=True)
    band_by_map = map_band_means(bundle)

    names = np.array([str(r["map"]) for r in records], object)
    folds = np.array([int(r["fold"]) for r in records], int)
    T = np.array([r["true"] for r in records], float)
    ratio = np.array([r["ratio"] for r in records], float)
    weighted = np.array([r["Ccal"] for r in records], float)
    band = np.stack([band_by_map[n] for n in names])
    part_raw_I = np.clip(band * ratio, 0, None)
    part_excess_I = ab[:, 1] + ratio * (band - ab[:, 1])

    bases = {
        "probability_weighted_band": weighted,
        "direct_band_calibration": inv_cal(band, ab),
        "pie_partition_raw_signal": inv_cal(part_raw_I, ab),
        "pie_partition_above_intercept": inv_cal(part_excess_I, ab),
    }
    all_pred, all_coefs = {}, []
    for base_name, base in bases.items():
        candidates, coefs = fit_predict_corrections(base, ratio, T, folds)
        for model_name, P in candidates.items():
            all_pred[f"{base_name}__{model_name}"] = P
        for row in coefs:
            row["baseline"] = base_name
        all_coefs.extend(coefs)

    kval = (T > 0).sum(1)
    subsets = {
        "all89": np.ones(len(T), bool),
        "low_grid64": np.all(np.isin(T, [3, 6, 12, 24]), axis=1),
        "max100": T.max(1) <= 100,
        "binary": kval == 2,
        "ternary": kval == 3,
        "high_absolute": T.max(1) > 100,
    }
    summaries = {}
    for method, P in all_pred.items():
        summaries[method] = {sub: metrics(T, P, sel) for sub, sel in subsets.items() if sel.any()}

    rank = sorted(all_pred, key=lambda m: (
        -summaries[m]["low_grid64"]["within_80_120_pct"],
        summaries[m]["low_grid64"]["median_abs_recovery_error_pct"],
        summaries[m]["low_grid64"]["mae_uM"],
    ))
    result = {
        "protocol": {
            "n_maps": int(len(T)),
            "source": str(RESULT),
            "split": "existing absolute-condition grouped 5-fold OOF composition MLP; response correction fit on other four folds",
            "coefficient_interpretation": "empirical effective mixture-response coefficient; not an adsorption equilibrium constant",
            "primary_metric": "component predictions within 80–120% recovery",
            "calibration_bands_cm": BANDS.tolist(),
        },
        "ranking_low_grid64": rank,
        "summary": summaries,
        "coefficients": all_coefs,
    }
    (outdir / "effective_response_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=jdefault), encoding="utf-8")

    summary_fields = ["method", "subset", "n_maps", "n_present", "recovery_mean_pct", "recovery_median_pct",
                      "recovery_q1_pct", "recovery_q3_pct", "median_abs_recovery_error_pct",
                      "mean_abs_recovery_error_pct", "within_80_120_pct", "within_70_130_pct",
                      "within_1_5x_pct", "mae_uM", "euclidean_mean_uM", "false_positive_mean_uM",
                      "pearson_r", "spearman_r", "r2_linear", "ccc"]
    with (outdir / "effective_response_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields); w.writeheader()
        for method in rank:
            for sub, vals in summaries[method].items():
                w.writerow({k: method if k == "method" else sub if k == "subset" else vals[k] for k in summary_fields})

    comp_fields = ["method", "subset", "component", "n", "recovery_mean_pct", "recovery_median_pct",
                   "recovery_q1_pct", "recovery_q3_pct", "median_abs_recovery_error_pct",
                   "within_80_120_pct", "mae_uM", "pearson_r", "ccc"]
    with (outdir / "component_recovery_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=comp_fields); w.writeheader()
        for method in rank:
            for sub, vals in summaries[method].items():
                for component, cv in vals["components"].items():
                    w.writerow({k: method if k == "method" else sub if k == "subset" else component if k == "component" else cv[k] for k in comp_fields})

    coef_fields = sorted({k for row in all_coefs for k in row})
    with (outdir / "fold_coefficients.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=coef_fields); w.writeheader(); w.writerows(all_coefs)

    with (outdir / "effective_response_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["map", "fold", "component", "true_uM", "ratio_pct"] + rank
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for i, name in enumerate(names):
            for j, sub in enumerate(SUBS):
                row = {"map": name, "fold": int(folds[i]), "component": sub,
                       "true_uM": T[i, j], "ratio_pct": ratio[i, j] * 100}
                row.update({m: all_pred[m][i, j] for m in rank})
                w.writerow(row)

    print("TOP LOW-GRID64")
    for method in rank[:8]:
        s = summaries[method]["low_grid64"]
        print(f"{method}: within80-120={s['within_80_120_pct']:.1f}% medianRec={s['recovery_median_pct']:.1f}% "
              f"medAbsRecErr={s['median_abs_recovery_error_pct']:.1f}% MAE={s['mae_uM']:.2f} r={s['pearson_r']:.3f} CCC={s['ccc']:.3f}")
    print(f"OUTPUT={outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "documentation/results/effective_response_20260825")
    args = ap.parse_args()
    run(args.out)
