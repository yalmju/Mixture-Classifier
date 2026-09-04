"""Compare three concentration reconstruction routes on the same 92 conditions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
BASE = ROOT / "documentation/results/mlp_concentration_axis_92_20260825/mlp_concentration_axis_92_predictions.csv"
TOTAL = ROOT / "documentation/results/mlp_concentration_axis_92_20260825/label_learned_total_head_92_predictions.csv"
OUT = ROOT / "documentation/results/three_concentration_routes_92_20260825/results.json"
COMPONENTS = ("DQ", "TBZ", "THI")
ABSOLUTE_MODEL = "ExtraTrees"


def safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float | int | None]:
    present = true > 0
    t = true[present]
    p = pred[present]
    ratio = p / t
    log_error = np.log2(np.clip(ratio, 1e-12, None))
    per_condition_euclidean = np.linalg.norm(pred - true, axis=1)
    true_total = true.sum(axis=1)
    pred_total = pred.sum(axis=1)
    total_ratio = pred_total / true_total
    return {
        "n_conditions": int(len(true)),
        "n_present_components": int(present.sum()),
        "median_recovery_pct": float(100 * np.median(ratio)),
        "mean_recovery_pct": float(100 * np.mean(ratio)),
        "q1_recovery_pct": float(100 * np.quantile(ratio, 0.25)),
        "q3_recovery_pct": float(100 * np.quantile(ratio, 0.75)),
        "within_80_120_pct": float(100 * np.mean((ratio >= 0.8) & (ratio <= 1.2))),
        "within_1_25x_pct": float(100 * np.mean((ratio >= 0.8) & (ratio <= 1.25))),
        "within_1_5x_pct": float(100 * np.mean((ratio >= 2 / 3) & (ratio <= 1.5))),
        "within_2x_pct": float(100 * np.mean((ratio >= 0.5) & (ratio <= 2.0))),
        "mae_present_uM": float(np.mean(np.abs(p - t))),
        "mae_all_channels_uM": float(np.mean(np.abs(pred - true))),
        "rmse_log2_present": float(np.sqrt(np.mean(log_error**2))),
        "pearson_present": safe_corr(t, p),
        "euclidean_mean_uM": float(np.mean(per_condition_euclidean)),
        "euclidean_median_uM": float(np.median(per_condition_euclidean)),
        "total_mae_uM": float(np.mean(np.abs(pred_total - true_total))),
        "total_median_recovery_pct": float(100 * np.median(total_ratio)),
        "total_rmse_log2": float(np.sqrt(np.mean(np.log2(np.clip(total_ratio, 1e-12, None)) ** 2))),
        "total_pearson": safe_corr(true_total, pred_total),
    }


def component_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, dict[str, float | int | None]]:
    result = {}
    for j, component in enumerate(COMPONENTS):
        keep = true[:, j] > 0
        t = true[keep, j]
        p = pred[keep, j]
        r = p / t
        result[component] = {
            "n": int(len(t)),
            "median_recovery_pct": float(100 * np.median(r)),
            "mean_recovery_pct": float(100 * np.mean(r)),
            "within_1_5x_pct": float(100 * np.mean((r >= 2 / 3) & (r <= 1.5))),
            "within_2x_pct": float(100 * np.mean((r >= 0.5) & (r <= 2.0))),
            "mae_uM": float(np.mean(np.abs(p - t))),
            "pearson_r": safe_corr(t, p),
        }
    return result


def main() -> None:
    base = pd.read_csv(BASE)
    learned = pd.read_csv(TOTAL)
    learned = learned[learned["model"] == ABSOLUTE_MODEL].copy()
    if len(base) != 92 or len(learned) != 92:
        raise RuntimeError(f"Expected 92 conditions, got base={len(base)}, learned={len(learned)}")

    learned = learned.set_index("condition").loc[base["condition"]].reset_index()
    if not np.allclose(
        learned[[f"true_{c}" for c in COMPONENTS]].to_numpy(float),
        base[[f"true_{c}" for c in COMPONENTS]].to_numpy(float),
    ):
        raise RuntimeError("Truth vectors do not align after condition join")

    true = base[[f"true_{c}" for c in COMPONENTS]].to_numpy(float)
    pie = base[[f"mlp_ratio_pct_{c}" for c in COMPONENTS]].to_numpy(float) / 100
    pie = pie / pie.sum(axis=1, keepdims=True)
    predictions = {
        "known_total": pie * true.sum(axis=1, keepdims=True),
        "absolute_spectrum_ExtraTrees": learned[[f"pred_{c}" for c in COMPONENTS]].to_numpy(float),
        "single_calibration": base[[f"single_apparent_uM_{c}" for c in COMPONENTS]].to_numpy(float),
    }

    inrange = np.all(true <= 100, axis=1)
    subset_values = base["subset"].to_numpy()
    subsets = {
        "all_92": np.ones(len(base), dtype=bool),
        "ternary_all_71": subset_values == "ternary",
        "ternary_inrange_67": (subset_values == "ternary") & inrange,
        "ternary_out_of_range_4": (subset_values == "ternary") & ~inrange,
        "binary_21": subset_values == "binary",
    }

    summary = {}
    components = {}
    for subset_name, mask in subsets.items():
        summary[subset_name] = {}
        components[subset_name] = {}
        for method, pred in predictions.items():
            summary[subset_name][method] = metrics(true[mask], pred[mask])
            components[subset_name][method] = component_metrics(true[mask], pred[mask])

    detail = []
    for i, row in base.iterrows():
        item = {"condition": row["condition"], "subset": row["subset"], "in_calibration_range": bool(inrange[i])}
        for j, component in enumerate(COMPONENTS):
            item[f"true_{component}_uM"] = float(true[i, j])
            for method, pred in predictions.items():
                item[f"{method}_{component}_uM"] = float(pred[i, j])
        item["true_total_uM"] = float(true[i].sum())
        for method, pred in predictions.items():
            item[f"{method}_total_uM"] = float(pred[i].sum())
        detail.append(item)

    payload = {
        "protocol": {
            "conditions": 92,
            "exclusion": ">=100-fold nonzero imbalance excluded upstream",
            "composition": "stored four-class MLP held-out composition",
            "known_total": "MLP composition multiplied by label total; not blind absolute quantification",
            "absolute_spectrum": "ExtraTrees held-out normalized-ratio-group 5-fold prediction of log2 total from map-spectrum, signal and NNLS-amplitude summaries; multiplied by the same MLP composition",
            "single_calibration": "component-wise pure-analyte calibration inversion",
            "recovery_denominator": "truly present components only",
            "all_channel_errors": "include the absent binary output channel",
            "absolute_model_selection_note": "ExtraTrees was the lowest all-condition component MAE among the previously tested total heads; selection is exploratory",
        },
        "summary": summary,
        "component_summary": components,
        "detail": detail,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"protocol": payload["protocol"], "summary": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
