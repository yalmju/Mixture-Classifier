"""Compare spectral inputs for NNLS-hit map-level composition prediction.

The split is read from Samples (mixtures.json): Role=train maps fit the model and
Role=test maps remain independent.  NNLS decides the hit pixels exactly as in Real;
only the representation supplied to the same MLP changes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def hit_mean(data_dir, path, baseline=False, trim=(500, 2500), min_frac=0.15):
    from dl_model import nnls_hit_spectra

    _wn, cube, meta = nnls_hit_spectra(
        data_dir, path, baseline=baseline, trim=trim, min_frac=min_frac)
    weight = cube.sum(axis=1)
    mean = (weight / (weight.sum() + 1e-12)) @ cube
    return np.asarray(mean, np.float32), meta


def transform(train, test, kind):
    eps = 1e-12
    l2_tr = train / (np.linalg.norm(train, axis=1, keepdims=True) + eps)
    l2_te = test / (np.linalg.norm(test, axis=1, keepdims=True) + eps)
    if kind == "l2":
        return l2_tr.astype(np.float32), l2_te.astype(np.float32)

    log_tr, log_te = np.log1p(train), np.log1p(test)
    mu, sd = log_tr.mean(axis=0), log_tr.std(axis=0) + 1e-6
    raw_tr, raw_te = (log_tr - mu) / sd, (log_te - mu) / sd
    if kind == "lograw":
        return raw_tr.astype(np.float32), raw_te.astype(np.float32)

    if kind == "hybrid":
        scale_tr = np.log(np.linalg.norm(train, axis=1, keepdims=True) + eps)
        scale_te = np.log(np.linalg.norm(test, axis=1, keepdims=True) + eps)
        sm, ss = scale_tr.mean(0), scale_tr.std(0) + 1e-6
        scale_tr, scale_te = (scale_tr - sm) / ss, (scale_te - sm) / ss
        return (np.hstack([raw_tr, l2_tr, scale_tr]).astype(np.float32),
                np.hstack([raw_te, l2_te, scale_te]).astype(np.float32))
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output", default="output/representation_benchmark.json")
    ap.add_argument("--epochs", type=int, default=350)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()

    from dataset import load_mixture_list
    from dl_model import _refs
    from dl_quantify import _ratio, train_composition, predict_composition

    train_items = load_mixture_list(args.data_dir, "train")
    test_items = load_mixture_list(args.data_dir, "test")
    subs, _wn, _mask, _P, _lo, _hi = _refs(args.data_dir, False, (500, 2500))

    def load(items):
        X, Y, paths, hits = [], [], [], []
        for i, item in enumerate(items):
            print(f"screening {i + 1}/{len(items)}: {os.path.basename(item[0])}", flush=True)
            x, meta = hit_mean(args.data_dir, item[0])
            X.append(x)
            Y.append(_ratio([float(item[1].get(s, 0.0)) for s in subs]))
            paths.append(item[0]); hits.append(meta["hit_fraction"])
        return np.asarray(X), np.asarray(Y, np.float32), paths, hits

    Xtr0, Ytr, train_paths, train_hits = load(train_items)
    Xte0, Yte, test_paths, test_hits = load(test_items)
    target = next(i for i, p in enumerate(test_paths) if "1000DQ1000THI1000" in p)
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    result = {"subs": subs, "train_maps": len(train_items), "test_maps": len(test_items),
              "target_111": os.path.basename(test_paths[target]), "representations": {}}

    for kind in ("l2", "lograw", "hybrid"):
        Xtr, Xte = transform(Xtr0, Xte0, kind)
        runs = []
        for seed in seeds:
            model = train_composition(Xtr, Ytr, len(subs), pretrain=None, seed=seed,
                                      epochs_ft=args.epochs)
            pred = predict_composition(model, Xte)
            err = 0.5 * np.abs(pred - Yte).sum(axis=1)
            runs.append({"seed": seed, "mean_error": float(err.mean()),
                         "error_111": float(err[target]),
                         "pred_111": pred[target].astype(float).tolist()})
            print(kind, seed, "mean", round(float(err.mean()), 4), "1:1:1",
                  np.round(pred[target], 3), "error", round(float(err[target]), 4), flush=True)
        result["representations"][kind] = {
            "mean_error": float(np.mean([r["mean_error"] for r in runs])),
            "mean_error_111": float(np.mean([r["error_111"] for r in runs])),
            "sd_error_111": float(np.std([r["error_111"] for r in runs])),
            "mean_pred_111": np.mean([r["pred_111"] for r in runs], axis=0).tolist(),
            "runs": runs,
        }
    result["train_hit_fraction_range"] = [float(min(train_hits)), float(max(train_hits))]
    result["test_hit_fraction_range"] = [float(min(test_hits)), float(max(test_hits))]
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["representations"], indent=2), flush=True)


if __name__ == "__main__":
    main()
