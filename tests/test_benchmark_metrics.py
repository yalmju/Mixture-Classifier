import math

import numpy as np

from dataset import filename_mixture_truth
from dl_model import (
    _grouped_map_folds,
    _pool_predictions_by_map,
    benchmark_summary,
    concentration_summary,
)


def test_filename_is_authoritative_truth():
    ratio, conc = filename_mixture_truth(
        r"C:\maps\DQ12-TB3-TH6_corrected.csv", ["DQ", "TBZ", "THI", "BLK"])
    assert ratio == {"DQ": 12.0, "TBZ": 3.0, "THI": 6.0}
    assert math.isclose(conc["DQ"], 12e-6)
    assert math.isclose(conc["TBZ"], 3e-6)
    assert math.isclose(conc["THI"], 6e-6)


def test_benchmark_reports_whole_ratio_and_component_metrics():
    bench = {
        "subs": ["DQ", "TBZ", "THI"],
        "mlp": {
            "true": [[0.5, 0.5, 0.0], [0.2, 0.3, 0.5]],
            "pred": [[0.48, 0.52, 0.0], [0.1, 0.3, 0.6]],
        },
    }
    result = benchmark_summary(bench, cut_pp=15)["mlp"]
    assert np.isclose(result["whole_within_pp"][1.0], 0.0)
    assert np.isclose(result["whole_within_pp"][3.0], 0.5)
    assert np.isclose(result["whole_within_pp"][5.0], 0.5)
    assert np.isclose(result["whole_within_5pp"], 0.5)
    assert np.isclose(result["whole_within_2fold"], 1.0)
    assert np.isclose(result["acc_at_cut"], 1.0)
    assert set(result["component_metrics"]) == {"DQ", "TBZ", "THI"}
    assert np.isclose(result["component_metrics"]["TBZ"]["bias_pp"], 1.0)

def test_concentration_summary_is_heldout_and_multiplicative():
    ev = {
        "true_uM": [[10.0, 20.0, 0.0], [10.0, 20.0, 5.0]],
        "pred_uM": [[10.0, 30.0, 1.0], [12.5, 10.0, 10.0]],
    }
    result = concentration_summary(ev, ["DQ", "TBZ", "THI"])
    assert result["overall"]["n"] == 5
    assert np.isclose(result["overall"]["within_1_25x"], 0.4)
    assert np.isclose(result["overall"]["within_1_5x"], 0.6)
    assert np.isclose(result["overall"]["within_2_0x"], 1.0)
    assert np.isclose(result["whole_condition"]["all_present_within_1_5x"], 0.5)
    assert result["component"]["THI"]["absent_n"] == 1
    assert np.isclose(result["component"]["THI"]["absent_median_pred_uM"], 1.0)

def test_grouped_folds_keep_pixels_together_and_balance_102_maps():
    group_keys, map_keys = [], []
    for gi in range(10):
        n_maps = 11 if gi < 2 else 10
        for mi in range(n_maps):
            for _pixel in range(4):
                group_keys.append(f"ratio-{gi}")
                map_keys.append(f"map-{gi}-{mi}")
    folds = _grouped_map_folds(group_keys, map_keys, 5, seed=7)
    assert len(folds) == 5
    assert set().union(*folds) == {f"ratio-{i}" for i in range(10)}
    assert sum(len(fold) for fold in folds) == 10
    loads = []
    g = np.asarray(group_keys, object); m = np.asarray(map_keys, object)
    for fold in folds:
        loads.append(len(set(m[np.isin(g, list(fold))].tolist())))
    assert sum(loads) == 102
    assert max(loads) - min(loads) <= 2


def test_pixel_predictions_pool_once_per_map_even_for_same_filename_label():
    pred = np.array([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3],
                     [0.1, 0.9], [0.3, 0.7]])
    true = np.array([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 2)
    maps = ["map-a"] * 3 + ["map-b"] * 2
    names = ["same-label.csv"] * 5
    tv, pv, out_names, n_pixels = _pool_predictions_by_map(
        pred, true, maps, names)
    assert len(tv) == 2
    assert out_names == ["same-label.csv", "same-label.csv"]
    assert n_pixels == [3, 2]
    assert np.allclose(pv, [[0.7, 0.3], [0.2, 0.8]])
