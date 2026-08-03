"""Learn apparent SERS response -> input composition/concentration.

NNLS is only the dynamic hit gate.  The inverse head is trained on binary/ternary
maps with known solution concentrations, while Role=test maps remain independent.
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


def extract(data_dir, items):
    from unmix import unmix_map

    spectra, apparent, hit, y_um, paths = [], [], [], [], []
    for i, item in enumerate(items):
        print(f"extracting {i + 1}/{len(items)}: {os.path.basename(item[0])}", flush=True)
        r = unmix_map(data_dir, item[0], method="nnls", baseline=False,
                      trim=(500, 2500), min_frac=0.15)
        cube = np.asarray(r.spectra)[r.hit]
        w = cube.sum(1)
        spectra.append((w / (w.sum() + 1e-12)) @ cube)
        apparent.append(r.mean_ratio)
        hit.append(r.hit_frac)
        conc = item[2] if len(item) > 2 else {}
        names = [r.comps[j] for j in r.nonbg]
        stock = [float(conc.get(s, 0.0)) * 1e6 for s in names]
        dilution = max(1, sum(v > 0 for v in stock))
        y_um.append([v / dilution for v in stock])
        paths.append(item[0])
    return (np.asarray(spectra), np.asarray(apparent), np.asarray(hit)[:, None],
            np.asarray(y_um), paths)


def features(train, test):
    tr_s, tr_a, tr_h = train
    te_s, te_a, te_h = test
    eps = 1e-12
    tr_log, te_log = np.log1p(tr_s), np.log1p(te_s)
    mu, sd = tr_log.mean(0), tr_log.std(0) + 1e-6
    tr_log, te_log = (tr_log - mu) / sd, (te_log - mu) / sd
    tr_l2 = tr_s / (np.linalg.norm(tr_s, axis=1, keepdims=True) + eps)
    te_l2 = te_s / (np.linalg.norm(te_s, axis=1, keepdims=True) + eps)
    tr_scale = np.log(np.linalg.norm(tr_s, axis=1, keepdims=True) + eps)
    te_scale = np.log(np.linalg.norm(te_s, axis=1, keepdims=True) + eps)
    sm, ss = tr_scale.mean(0), tr_scale.std(0) + 1e-6
    tr_scale, te_scale = (tr_scale - sm) / ss, (te_scale - sm) / ss
    # Keep the apparent response explicit so the inverse model can learn that, e.g.,
    # weak TBZ under THI dominance may still correspond to substantial input TBZ.
    return (np.hstack([tr_log, tr_l2, tr_scale, tr_a, tr_h]),
            np.hstack([te_log, te_l2, te_scale, te_a, te_h]))


def normalise(x):
    x = np.clip(np.asarray(x, float), 0, None)
    return x / (x.sum(1, keepdims=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output", default="output/inverse_competition_benchmark.json")
    args = ap.parse_args()

    from dataset import load_mixture_list
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline

    train_items = load_mixture_list(args.data_dir, "train")
    test_items = load_mixture_list(args.data_dir, "test")
    tr_s, tr_a, tr_h, tr_um, tr_paths = extract(args.data_dir, train_items)
    te_s, te_a, te_h, te_um, te_paths = extract(args.data_dir, test_items)
    Xtr, Xte = features((tr_s, tr_a, tr_h), (te_s, te_a, te_h))
    y_frac_tr, y_frac_te = normalise(tr_um), normalise(te_um)
    y_log_tr = np.log10(tr_um + 1.0)
    target = next(i for i, p in enumerate(te_paths) if "1000DQ1000THI1000" in p)

    models = {
        "ridge": Ridge(alpha=100.0),
        "pca_ridge": make_pipeline(PCA(n_components=12, random_state=0), Ridge(alpha=10.0)),
        "pls": PLSRegression(n_components=8, scale=True),
        "rf": RandomForestRegressor(n_estimators=600, min_samples_leaf=2,
                                     max_features=0.5, random_state=0),
        "extra_trees": ExtraTreesRegressor(n_estimators=600, min_samples_leaf=2,
                                            max_features=0.7, random_state=0),
    }
    result = {"subs": ["DQ", "TBZ", "THI"], "target_111": os.path.basename(te_paths[target]),
              "true_111_uM": te_um[target].tolist(), "models": {}}
    for name, model in models.items():
        model.fit(Xtr, y_frac_tr)
        pf = normalise(model.predict(Xte))
        comp_err = 0.5 * np.abs(pf - y_frac_te).sum(1)

        # A second head predicts log10(uM + 1); it shares the representation but not
        # the simplex constraint, so total concentration information is retained.
        if name == "ridge":
            cm = Ridge(alpha=100.0)
        elif name == "pca_ridge":
            cm = make_pipeline(PCA(n_components=12, random_state=0), Ridge(alpha=10.0))
        elif name == "pls":
            cm = PLSRegression(n_components=8, scale=True)
        elif name == "rf":
            cm = RandomForestRegressor(n_estimators=600, min_samples_leaf=2,
                                       max_features=0.5, random_state=0)
        else:
            cm = ExtraTreesRegressor(n_estimators=600, min_samples_leaf=2,
                                      max_features=0.7, random_state=0)
        cm.fit(Xtr, y_log_tr)
        pu = np.maximum(10.0 ** np.asarray(cm.predict(Xte)) - 1.0, 0.0)
        log_mae = np.abs(np.log10(pu + 1.0) - np.log10(te_um + 1.0)).mean()
        result["models"][name] = {
            "mean_composition_error": float(comp_err.mean()),
            "composition_error_111": float(comp_err[target]),
            "pred_111_fraction": pf[target].tolist(),
            "pred_111_uM": pu[target].tolist(),
            "mean_log10_uM_mae": float(log_mae),
        }
        print(name, "fraction", np.round(pf[target], 3), "uM", np.round(pu[target], 1),
              "comp_err", round(float(comp_err[target]), 3),
              "logMAE", round(float(log_mae), 3), flush=True)

    # Training-map bootstrap for the selected ridge inverse head. This quantifies
    # model uncertainty, separately from the within-map hit-pixel P10-P90.
    rng = np.random.default_rng(20260803)
    boot_frac, boot_um = [], []
    for _ in range(500):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        fm = Ridge(alpha=100.0, solver="lsqr").fit(Xtr[idx], y_frac_tr[idx])
        cm = Ridge(alpha=100.0, solver="lsqr").fit(Xtr[idx], y_log_tr[idx])
        boot_frac.append(normalise(fm.predict(Xte[target:target + 1]))[0])
        boot_um.append(np.maximum(10.0 ** cm.predict(Xte[target:target + 1])[0] - 1.0, 0.0))
    boot_frac, boot_um = np.asarray(boot_frac), np.asarray(boot_um)
    result["ridge_bootstrap_111"] = {
        "n": 500,
        "fraction_median": np.median(boot_frac, axis=0).tolist(),
        "fraction_p10": np.percentile(boot_frac, 10, axis=0).tolist(),
        "fraction_p90": np.percentile(boot_frac, 90, axis=0).tolist(),
        "uM_median": np.median(boot_um, axis=0).tolist(),
        "uM_p10": np.percentile(boot_um, 10, axis=0).tolist(),
        "uM_p90": np.percentile(boot_um, 90, axis=0).tolist(),
        "uM_ci95_low": np.percentile(boot_um, 2.5, axis=0).tolist(),
        "uM_ci95_high": np.percentile(boot_um, 97.5, axis=0).tolist(),
    }

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
