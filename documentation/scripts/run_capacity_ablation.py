# -*- coding: utf-8 -*-
"""용량 스윕 + 학습 곡선 — "얕은 MLP가 최적점"을 서사가 아니라 측정으로.

같은 ratio-grouped 5-fold(통합 프레임과 동일 fold)에서
  (A) MLP 은닉 폭 (32,16)…(1024,256) 스윕 → 조성 오차·검출 AUC vs 용량
  (B) 학습 조건 25/50/75/100% 서브샘플 × {MLP(256,64), PLS} → 학습 곡선
을 채점한다. dl_quantify._spec_net 의 기본 hidden 을 런마다 몽키패치.

실행:  python -u run_capacity_ablation.py
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
MAIN = Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(MAIN / "documentation" / "scripts"))
import run_integrated_reevaluation as R  # noqa: E402
import dl_model  # noqa: E402
import dl_quantify  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "capacity_ablation_20260831"
OUT.mkdir(parents=True, exist_ok=True)
_orig_spec = dl_quantify._spec_net


def set_hidden(h):
    dl_quantify._spec_net = (lambda n_feat, n_comp, hidden=h: _orig_spec(n_feat, n_comp, h))


def score(rows):
    from sklearn.metrics import roc_auc_score
    T = np.stack([r["true"] for r in rows]); P = np.stack([r["pred"] for r in rows])
    dev = 0.5 * np.abs(P - T).sum(1) * 100
    k = (T > 0).sum(1); b = k == 2
    auc = roc_auc_score((T[b] > 0).ravel().astype(int), P[b].ravel()) if b.any() else float("nan")
    return {"mean_dev_pp": float(dev.mean()), "binary_dev_pp": float(dev[b].mean()),
            "ternary_dev_pp": float(dev[k >= 3].mean()), "det_auc": float(auc)}


def run_mlp(X, Y, maps, names, fold_px, tr_extra=None, seed0=170, epochs=120):
    rows = []
    for fi in range(1, 6):
        te = np.where(fold_px == fi)[0]; tr = np.where(fold_px != fi)[0]
        if tr_extra is not None:
            tr = tr[np.isin(tr, tr_extra)]
        pred = dl_model._fit_predict("mlp", X[tr], Y[tr], X[te], epochs=epochs,
                                     seed=seed0 + fi, train_maps=maps[tr])
        rows.extend({"true": t, "pred": p} for t, p in
                    zip(*_pool(pred, Y[te], names[te])))
    return rows


def run_pls(X, Y, names, fold_px, tr_extra=None):
    rows = []
    for fi in range(1, 6):
        te = np.where(fold_px == fi)[0]; tr = np.where(fold_px != fi)[0]
        if tr_extra is not None:
            tr = tr[np.isin(tr, tr_extra)]
        pred = dl_model._fit_predict("pls", X[tr], Y[tr], X[te], n_components=8)
        rows.extend({"true": t, "pred": p} for t, p in
                    zip(*_pool(pred, Y[te], names[te])))
    return rows


def _pool(pred, Yte, nte):
    ts, ps = [], []
    for name, idx in R.map_rows(nte):
        ts.append(np.asarray(Yte[idx[0]], float))
        ps.append(np.asarray(pred[idx], float).mean(0))
    return ts, ps


def main():
    t0 = time.time()
    z = np.load(R.BUNDLE, allow_pickle=True)
    X, Y = z["X"].astype(np.float32), z["Y"].astype(np.float32)
    names, maps = z["names"].astype(object), z["mapkey"].astype(object)
    old = json.loads(R.OLD_FOLDS.read_text(encoding="utf-8"))
    fold_by = {str(n): int(f) for n, f in zip(old["condition"], old["fold"])}
    fold_px = np.array([fold_by[str(n)] for n in names], int)

    results = {"capacity": [], "learning": []}
    # (A) 용량 스윕
    for h in [(32, 16), (64, 32), (128, 64), (256, 64), (512, 128), (1024, 256)]:
        set_hidden(h)
        s = score(run_mlp(X, Y, maps, names, fold_px))
        n_par = 1290 * h[0] + h[0] * h[1] + h[1] * 4
        results["capacity"].append({"hidden": str(h), "params_approx": n_par, **s})
        print(f"[A] hidden {str(h):>11}  dev {s['mean_dev_pp']:.2f}pp  binary {s['binary_dev_pp']:.2f}  AUC {s['det_auc']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    set_hidden((256, 64))
    # (B) 학습 곡선 — ratio-condition 단위 서브샘플 (fold 구조 유지)
    rng = np.random.default_rng(7)
    conds = np.asarray([str(n) for n in names], object)
    uniq = list(dict.fromkeys(conds.tolist()))
    for frac in (0.25, 0.5, 0.75, 1.0):
        keep = set(rng.choice(uniq, size=max(8, int(len(uniq) * frac)), replace=False)) \
            if frac < 1.0 else set(uniq)
        tr_extra = np.where(np.isin(conds, list(keep)))[0]
        for tag, fn in (("mlp", lambda: run_mlp(X, Y, maps, names, fold_px, tr_extra)),
                        ("pls", lambda: run_pls(X, Y, names, fold_px, tr_extra))):
            s = score(fn())
            results["learning"].append({"frac": frac, "model": tag, **s})
            print(f"[B] {tag} @ {frac:.0%}  dev {s['mean_dev_pp']:.2f}pp  AUC {s['det_auc']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    R.dump_json(OUT / "results.json", results)
    import csv
    for key in ("capacity", "learning"):
        with (OUT / f"{key}.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(results[key][0]))
            w.writeheader(); w.writerows(results[key])
    print(f"done {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
