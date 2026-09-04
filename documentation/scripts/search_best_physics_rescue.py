"""Search all 64 low-concentration held-out conditions for best physics rescue."""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import physics_max_rescue_experiment as exp
from run_physics_max_rescue_fixed import physics_pretrain_fixed


def build_cache(pure, items, P, cache_path):
    records = {}
    for i, (path, conc) in enumerate(items, 1):
        name = os.path.basename(path)
        print(f"cache {i}/{len(items)} {name}", flush=True)
        _, cube, _ = exp.nnls_hit_spectra(pure, path, baseline=True,
                                          trim=None, min_frac=.15)
        mask = np.ones(cube.shape[1], bool)
        raw_train = exp._map_spectra(cube, mask, 20, baseline_correct=False)
        raw_eval = exp._map_spectra(cube, mask, 0, baseline_correct=False)
        train_x = np.asarray([
            exp._composition_features(r, "log1p_raw") for r in raw_train
        ], np.float32)
        eval_x = np.asarray([
            exp._composition_features(r, "log1p_raw") for r in raw_eval
        ], np.float32)
        surface = np.asarray([
            exp.surface_composition(
                exp._composition_features(np.asarray(r)[None, :], "legacy_l2"), P
            )[0] for r in raw_train
        ], np.float32)
        true_c = np.asarray(conc, float) * 1e-6
        records[tuple(conc)] = {
            "path": path, "train_x": train_x, "eval_x": eval_x,
            "surface": surface, "true_c": true_c,
            "true_y": exp._ratio(true_c).astype(np.float32),
        }
    with open(cache_path, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    return records


def arrays(records, keys):
    xs, ys, cs, ss = [], [], [], []
    for key in keys:
        r = records[key]; n = len(r["train_x"])
        xs.append(r["train_x"]); ss.append(r["surface"])
        ys.append(np.repeat(r["true_y"][None, :], n, axis=0))
        cs.append(np.repeat(r["true_c"][None, :], n, axis=0))
    return tuple(map(np.concatenate, (xs, ys, cs, ss)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=120)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    folder = os.path.join(a.db, "Ratio", "260814_mixture_final")
    pure = os.path.join(a.db, "Pure")
    low, high = exp.parse_folder(folder)
    all_items = high + low
    _, _, mask, P, _, _ = exp._refs(pure, True, None)
    cache_path = os.path.join(a.out, "physics_search_feature_cache.pkl")
    if os.path.exists(cache_path):
        print("loading cache", cache_path, flush=True)
        with open(cache_path, "rb") as f: records = pickle.load(f)
    else:
        records = build_cache(pure, all_items, P, cache_path)
    preX, preY, cal, m = physics_pretrain_fixed(
        pure, exp.calibration_path(a.db), P, mask
    )
    low_keys = sorted(c for _, c in low)
    high_keys = [c for _, c in high]
    order = np.random.default_rng(0).permutation(len(low_keys))
    folds = [[low_keys[i] for i in order[f::4]] for f in range(4)]
    predictions = {}
    for fi, held in enumerate(folds):
        print(f"training fold {fi+1}/4", flush=True)
        train_keys = high_keys + [k for k in low_keys if k not in set(held)]
        X, Y, C, S = arrays(records, train_keys)
        model = exp.fit_models(X, Y, C, S, (preX, preY), cal, m,
                               a.epochs, weights=(.03,))[.03][0]
        for key in held:
            predictions[key] = (fi + 1, exp.predict(model, records[key]["eval_x"]))
    rows = []
    for i, key in enumerate(low_keys, 1):
        print(f"exact NNLS {i}/{len(low_keys)} {key}", flush=True)
        r = records[key]; fold, phys = predictions[key]
        _, _, _, _, _, nnls = exp.nnls_and_spectrum(pure, r["path"])
        true = r["true_y"]
        err = lambda p: float(.5 * np.abs(np.asarray(p) - true).sum() * 100)
        ep, en = err(phys), err(nnls)
        rows.append({
            "condition": f"DQ{key[0]}-TBZ{key[1]}-THI{key[2]}",
            "fold": fold,
            "true_DQ_pct": true[0]*100, "true_TBZ_pct": true[1]*100,
            "true_THI_pct": true[2]*100,
            "nnls_DQ_pct": nnls[0]*100, "nnls_TBZ_pct": nnls[1]*100,
            "nnls_THI_pct": nnls[2]*100, "nnls_error_pct": en,
            "physics_DQ_pct": phys[0]*100, "physics_TBZ_pct": phys[1]*100,
            "physics_THI_pct": phys[2]*100, "physics_error_pct": ep,
            "rescue_gain_pct_point": en-ep,
            "eligible_physics_error_lt5": ep < 5,
        })
    rows.sort(key=lambda r: (r["eligible_physics_error_lt5"],
                             r["rescue_gain_pct_point"]), reverse=True)
    out = os.path.join(a.out, "physics_all_64_heldout_rescue.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("BEST", rows[0], flush=True); print(out, flush=True)


if __name__ == "__main__":
    main()
