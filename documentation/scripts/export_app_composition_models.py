"""Fit full-data MLP and PLS composition heads and export app-compatible DLM files."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression

ROOT = Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
sys.path.insert(0, str(ROOT))
from dl_model import _fit_torch_bag, apply_model_pixels, load_model, save_model

BUNDLE = ROOT / "documentation/results/composition_all_no100/v2_search/bundle_89maps_24px.npz"
PARTIAL = ROOT / "documentation/results/composition_all_no100/partial"
AXIS = 100.0 + 1.55 * np.arange(2001)


def norm_rows(p):
    p = np.clip(np.asarray(p, float), 0, None)
    return p / (p.sum(1, keepdims=True) + 1e-12)


def pool(pred, y, names):
    true, out, paths = [], [], []
    for name in dict.fromkeys(names.tolist()):
        idx = np.where(names == name)[0]
        true.append(np.asarray(y[idx[0]], float).tolist())
        out.append(np.asarray(pred[idx], float).mean(0).tolist())
        paths.append(str(name))
    return {"true": true, "pred": out, "paths": paths, "level": "map"}


def oof(method):
    d = json.loads((PARTIAL / f"{method}.json").read_text(encoding="utf-8"))
    return {"true": d["true"], "pred": d["pred"], "paths": d["condition"],
            "fold": d["fold"], "level": "ratio-condition-grouped-5-fold"}


def common(z, method, train_eval, loo_eval, epochs=None):
    subs = z["subs"].astype(str).tolist()
    model = {
        "subs": subs,
        "lo": float(AXIS[0]), "hi": float(AXIS[-1]), "n_feat": int(len(AXIS)),
        "P": z["P_ref"].astype(float), "feature_mode": "log1p_raw",
        "calib_csv_text": None, "calib_csv_name": None,
        "uM": None, "has_uM": False,
        "n_train": int(len(z["X"])), "n_maps": int(len(set(z["names"].tolist()))),
        "px_per_map": 24, "pixel_sampling": "cached_nnls_screened_24px",
        "sampling_seed": 0, "training_level": "map_pooled_pixels",
        "nnls_screen": True, "screen_min_frac": 0.15,
        "equal_volume_mix": False, "concentration_basis": "final concentrations in filenames",
        "screen_stats": None, "data_dir": str(BUNDLE),
        "baseline": False, "trim": (float(AXIS[0]), float(AXIS[-1])),
        "train_eval": train_eval, "loo_eval": loo_eval, "test_eval": None,
        "blank": None, "method": method,
        "model_scope": "89 maps; nine >=100-fold imbalance conditions excluded",
        "protocol_version": "app_export_20260825",
    }
    if epochs is not None:
        model.update({"selected_epochs": int(epochs), "epoch_rule": "fixed-map-pooled",
                      "selection_level": "map", "comp_hidden": (256, 64)})
    return model


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(outdir, epochs):
    outdir.mkdir(parents=True, exist_ok=True)
    z = np.load(BUNDLE, allow_pickle=True)
    X, Y = z["X"].astype(np.float32), z["Y"].astype(np.float32)
    names = z["names"].astype(object)

    print(f"training MLP on {len(set(names.tolist()))} maps / {len(X)} pixels", flush=True)
    pred_mlp, net = _fit_torch_bag("mlp", X, Y, names, X, epochs=epochs,
                                   seed=170, return_net=True,
                                   progress=lambda s: print(s, flush=True))
    mlp = common(z, "mlp", pool(pred_mlp, Y, names), oof("mlp"), epochs)
    mlp["comp_state"] = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    mlp_path = outdir / "UNMIXR_MLP_89maps_no100x_20260825.dlm"
    save_model(mlp, mlp_path)

    print("training PLS(8) on identical rows", flush=True)
    sk = PLSRegression(n_components=8).fit(X.astype(float), Y.astype(float))
    pred_pls = norm_rows(sk.predict(X.astype(float)))
    pls = common(z, "pls", pool(pred_pls, Y, names), oof("pls"))
    pls.update({"sk": sk, "n_components": 8})
    pls_path = outdir / "UNMIXR_PLS8_89maps_no100x_20260825.dlm"
    save_model(pls, pls_path)

    checks = []
    raw = np.expm1(np.clip(X[:12], 0, None))
    for path in (mlp_path, pls_path):
        model = load_model(path)
        pp = apply_model_pixels(model, AXIS, raw)
        checks.append({"file": str(path), "method": model["method"], "shape": list(pp.shape),
                       "row_sum_min": float(pp.sum(1).min()), "row_sum_max": float(pp.sum(1).max()),
                       "sha256": sha(path), "bytes": path.stat().st_size})
    manifest = {"bundle": str(BUNDLE), "epochs_mlp": int(epochs), "seed_mlp": 170,
                "pls_components": 8, "models": checks,
                "note": "Composition-only DLM files. Concentration correction is intentionally not embedded."}
    (outdir / "UNMIXR_app_models_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for row in checks:
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/models_20260825")
    ap.add_argument("--epochs", type=int, default=120)
    a = ap.parse_args(); run(a.out, a.epochs)
