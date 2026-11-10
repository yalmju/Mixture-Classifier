"""Leakage-aware integrated composition/concentration re-evaluation.

Uses the cached 89-map real-data bundle (24 NNLS-screened pixels per map), excluding
the nine >=100-fold imbalance conditions. Composition is evaluated by ratio-grouped
held-out folds. Concentration uses absolute-condition outer folds; in every fold the
composition MLP and the calibration-guided residual head are fit only on training maps.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[6] / "Google Drive" / "내 드라이브" / "github" / "Mixture Classifier"
if not ROOT.exists():
    ROOT = Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")

BUNDLE = ROOT / "documentation/results/composition_all_no100/v2_search/bundle_89maps_24px.npz"
OLD_FOLDS = ROOT / "documentation/results/composition_all_no100/partial/mlp.json"
OLD_CONC = ROOT / "documentation/results/v2_concentration/concentration_residual_results.json"

SUBS = ["DQ", "TBZ", "THI"]
BAND_CM = np.array([1570.0, 1270.0, 1367.0], float)
AXIS_CM = 100.0 + 1.55 * np.arange(2001)


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")


def jfloat(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    raise TypeError(type(x).__name__)


def map_rows(names):
    names = np.asarray(names, object)
    return [(str(n), np.where(names == n)[0]) for n in dict.fromkeys(names.tolist())]


def pool_maps(pred_px, truth_px, names, conc_px=None):
    out = []
    for name, idx in map_rows(names):
        row = {
            "map": name,
            "true": np.asarray(truth_px[idx[0]], float),
            "pred": np.asarray(pred_px[idx], float).mean(0),
            "n_pixels": int(len(idx)),
        }
        if conc_px is not None:
            row["concentration_uM"] = np.asarray(conc_px[idx[0]], float)
        out.append(row)
    return out


def grouped_folds(groups, n_folds=5, seed=701):
    groups = np.asarray(groups, object)
    uniq = np.asarray(list(dict.fromkeys(groups.tolist())), object)
    rng = np.random.default_rng(seed)
    order = uniq[rng.permutation(len(uniq))]
    bins = [set() for _ in range(n_folds)]
    loads = np.zeros(n_folds, int)
    counts = {g: int(np.sum(groups == g)) for g in uniq}
    for g in sorted(order.tolist(), key=lambda q: -counts[q]):
        k = int(np.argmin(loads))
        bins[k].add(g)
        loads[k] += counts[g]
    return bins


def safe_auc(y, score):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, int)
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def composition_stats(rows):
    T = np.stack([r["true"] for r in rows])
    P = np.stack([r["pred"] for r in rows])
    k = (T > 0).sum(1)
    dev = 0.5 * np.abs(P - T).sum(1) * 100.0
    def block(sel):
        d = dev[sel]
        return {
            "n": int(sel.sum()),
            "mean_deviation_pp": float(d.mean()),
            "se_deviation_pp": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
            "median_deviation_pp": float(np.median(d)),
            "accuracy_at_10pp": float(np.mean(d <= 10.0)),
        }
    out = {"all": block(np.ones(len(T), bool))}
    if np.any(k == 2): out["binary"] = block(k == 2)
    if np.any(k >= 3): out["ternary"] = block(k >= 3)
    rec = {}
    for j, s in enumerate(SUBS):
        present = T[:, j] > 0
        q = P[present, j] / T[present, j] * 100.0
        rec[s] = {
            "n": int(len(q)), "mean_pct": float(q.mean()), "sd_pct": float(q.std(ddof=1)),
            "median_pct": float(np.median(q)), "within_80_120": float(np.mean((q >= 80) & (q <= 120))),
        }
    out["all"]["recovery"] = rec
    binary = k == 2
    out["binary_detection_auc"] = safe_auc((T[binary] > 0).ravel(), P[binary].ravel())
    return out


def roc_points(rows, method):
    T = np.stack([r["true"] for r in rows])
    P = np.stack([r["pred"] for r in rows])
    sel = (T > 0).sum(1) == 2
    y = (T[sel] > 0).ravel().astype(int)
    score = P[sel].ravel()
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, score)
    return [{"method": method, "fpr": float(a), "tpr": float(b), "threshold": float(c)}
            for a, b, c in zip(fpr, tpr, thr)]


def characteristic_signal(raw_px, prob_px):
    """Composition-weighted band signal per map, one value per component.

    Each pixel contributes in proportion to the held-out composition probability for
    that component. Within a ±10 cm-1 band, the local maximum is used to tolerate the
    instrument's 1.55 cm-1 grid and small peak drift.
    """
    raw_px = np.clip(np.asarray(raw_px, float), 0, None)
    prob_px = np.clip(np.asarray(prob_px, float), 0, None)
    signals = []
    band_p90 = []
    for j, centre in enumerate(BAND_CM):
        mask = np.abs(AXIS_CM - centre) <= 10.0
        band = raw_px[:, mask].max(1)
        w = prob_px[:, j] + 1e-6
        signals.append(float(np.sum(w * band) / np.sum(w)))
        band_p90.append(float(np.percentile(band, 90)))
    return np.asarray(signals), np.asarray(band_p90)


def calibration_context(X_px, prob_px, names, ab):
    raw = np.expm1(np.clip(np.asarray(X_px, float), 0, None))
    rows = []
    for name, idx in map_rows(names):
        p = np.asarray(prob_px[idx], float)
        ratio = p.mean(0); ratio /= ratio.sum() + 1e-12
        ieq, bp90 = characteristic_signal(raw[idx], p)
        c_unclipped = 10.0 ** ((ieq - ab[:, 1]) / ab[:, 0])
        ccal = np.clip(c_unclipped, 0.1, 100.0)
        total = raw[idx].sum(1)
        iq = np.percentile(np.log1p(total), [10, 50, 90])
        feat = np.concatenate([np.log10(ccal), ratio, np.log1p(ieq), iq])
        rows.append({"map": name, "feature": feat, "ratio": ratio, "Ieq": ieq,
                     "band_p90": bp90, "Ccal": ccal, "Ccal_unclipped": c_unclipped})
    return rows


class ResidualNet:
    def __init__(self, n_in, seed):
        import torch
        import torch.nn as nn
        torch.manual_seed(int(seed))
        self.net = nn.Sequential(nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(),
                                 nn.Dropout(0.25), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 3))


def masked_huber(pred, target, mask):
    import torch
    import torch.nn.functional as F
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=0.25)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def train_residual(X, Ccal, truth, seed=0, max_epochs=800, smoke=False):
    import torch
    X = np.asarray(X, np.float32); Ccal = np.asarray(Ccal, float); truth = np.asarray(truth, float)
    target = np.log10(np.clip(truth, 0.05, None)) - np.log10(np.clip(Ccal, 0.05, None))
    mask = (truth > 0).astype(np.float32)
    n = len(X); rng = np.random.default_rng(seed)
    perm = rng.permutation(n); nval = max(4, int(round(n * 0.2))) if n >= 20 else max(1, n // 5)
    va, tr = perm[:nval], perm[nval:]
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
    A = torch.tensor(((X - mu) / sd).astype(np.float32)); Y = torch.tensor(target.astype(np.float32)); M = torch.tensor(mask)
    model = ResidualNet(X.shape[1], seed).net
    op = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=3e-3)
    best, best_ep, stale = float("inf"), 20, 0
    budget = 40 if smoke else int(max_epochs)
    for ep in range(budget):
        model.train(); op.zero_grad(); loss = masked_huber(model(A[tr]), Y[tr], M[tr]); loss.backward(); op.step()
        model.eval()
        with torch.no_grad(): val = float(masked_huber(model(A[va]), Y[va], M[va]))
        if val < best - 1e-4: best, best_ep, stale = val, ep + 1, 0
        else: stale += 1
        if stale >= 80 and ep >= 100: break
    # Refit on every outer-training map for the validation-selected epoch count.
    mu = X.mean(0); sd = X.std(0) + 1e-6
    A = torch.tensor(((X - mu) / sd).astype(np.float32))
    model = ResidualNet(X.shape[1], seed + 1000).net
    op = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=3e-3)
    for _ in range(max(5, best_ep)):
        model.train(); op.zero_grad(); loss = masked_huber(model(A), Y, M); loss.backward(); op.step()
    model.eval()
    return model, mu, sd, int(best_ep), float(best)


def predict_residual(model, X, mu, sd, Ccal):
    import torch
    A = torch.tensor(((np.asarray(X, np.float32) - mu) / sd).astype(np.float32))
    model.eval()
    with torch.no_grad(): delta = model(A).numpy()
    pred = np.asarray(Ccal, float) * (10.0 ** np.clip(delta, -2.0, 2.0))
    return np.clip(pred, 0.001, 5000.0), delta


def concentration_stats(T, P, subset=None):
    T = np.asarray(T, float); P = np.asarray(P, float)
    if subset is None: subset = np.ones(len(T), bool)
    T, P = T[subset], P[subset]
    present = T > 0
    t, p = T[present], P[present]
    fold = p / t
    loge = np.log10(np.clip(fold, 1e-8, None))
    euclid = np.sqrt(np.sum((P - T) ** 2, axis=1))
    out = {
        "n_maps": int(len(T)), "n_present_components": int(len(t)),
        "mean_recovery_pct": float(fold.mean() * 100),
        "median_recovery_pct": float(np.median(fold) * 100),
        "mae_uM": float(np.mean(np.abs(p - t))),
        "rmse_log10": float(np.sqrt(np.mean(loge ** 2))),
        "within_1_5x_pct": float(np.mean((fold >= 1/1.5) & (fold <= 1.5)) * 100),
        "within_2x_pct": float(np.mean((fold >= 0.5) & (fold <= 2.0)) * 100),
        "euclidean_mean_uM": float(euclid.mean()), "euclidean_median_uM": float(np.median(euclid)),
    }
    out["components"] = {}
    for j, s in enumerate(SUBS):
        ok = T[:, j] > 0; q = P[ok, j] / T[ok, j] * 100
        out["components"][s] = {"n": int(ok.sum()), "mean_recovery_pct": float(q.mean()),
                                  "median_recovery_pct": float(np.median(q)),
                                  "mae_uM": float(np.mean(np.abs(P[ok, j] - T[ok, j])))}
    return out


def run(args):
    t0 = time.time(); outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    z = np.load(BUNDLE, allow_pickle=True)
    X, Y = z["X"].astype(np.float32), z["Y"].astype(np.float32)
    names, maps, C = z["names"].astype(object), z["mapkey"].astype(object), z["concentration"].astype(float)
    P_ref = z["P_ref"].astype(float)
    old = json.loads(OLD_FOLDS.read_text(encoding="utf-8"))
    fold_by_name = {str(n): int(f) for n, f in zip(old["condition"], old["fold"])}
    fold_px = np.array([fold_by_name[str(n)] for n in names], int)
    oldc = json.loads(OLD_CONC.read_text(encoding="utf-8"))
    ab = np.asarray(oldc["excel_loglinear_ab"], float)
    methods = ["nnls", "pls"] if args.smoke else ["nnls", "pls", "rf", "cnn", "mlp"]
    epochs = 3 if args.smoke else int(args.epochs)
    nfolds = 1 if args.smoke else 5
    from dl_model import _fit_predict

    composition = {}
    pixel_oof = {}
    for method in methods:
        rows, pred_all = [], np.full_like(Y, np.nan, dtype=float)
        for fi in range(1, nfolds + 1):
            te = np.where(fold_px == fi)[0]; tr = np.where(fold_px != fi)[0]
            pred = _fit_predict(method, X[tr], Y[tr], X[te], epochs=epochs, seed=170 + fi,
                                n_components=8, n_trees=(20 if args.smoke else 300), P_ref=P_ref,
                                rf_max_features="sqrt", train_maps=maps[tr])
            pred_all[te] = pred
            pooled = pool_maps(pred, Y[te], names[te], C[te])
            for r in pooled: r["fold"] = fi
            rows.extend(pooled)
            print(f"composition {method} fold {fi}/{nfolds} done", flush=True)
        composition[method] = rows
        pixel_oof[method] = pred_all
        dump_json(outdir / f"partial_{method}.json", {"rows": rows},)

    comp_summary = {m: composition_stats(r) for m, r in composition.items()}
    dump_json(outdir / "composition_results.json", {"protocol": {
        "n_maps": 89, "excluded_100x": 9, "folds": 5, "pixels_per_map": 24,
        "split": "ratio-condition-grouped", "epochs": int(args.epochs), "methods": methods},
        "summary": comp_summary, "rows": composition},)
    with (outdir / "composition_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["method","n","mean_deviation_pp","se_deviation_pp","median_deviation_pp","accuracy_at_10pp","binary_auc","DQ_mean_recovery_pct","TBZ_mean_recovery_pct","THI_mean_recovery_pct"])
        for m in methods:
            a=comp_summary[m]["all"]; rec=a["recovery"]
            w.writerow([m,a["n"],a["mean_deviation_pp"],a["se_deviation_pp"],a["median_deviation_pp"],a["accuracy_at_10pp"],comp_summary[m]["binary_detection_auc"],rec["DQ"]["mean_pct"],rec["TBZ"]["mean_pct"],rec["THI"]["mean_pct"]])
    with (outdir / "composition_roc.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields=["method","fpr","tpr","threshold"]; w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for m in methods: w.writerows(roc_points(composition[m],m))

    if args.smoke:
        dump_json(outdir / "smoke_complete.json", {"ok": True, "elapsed_s": time.time()-t0})
        print("SMOKE COMPLETE", flush=True); return

    # Absolute-condition grouped outer folds for the integrated concentration pipeline.
    abskey_px = np.array(["c:" + ",".join(f"{v:.8g}" for v in row) for row in C], object)
    cfolds = grouped_folds(abskey_px, 5, 701)
    conc_records = []
    for fi, held in enumerate(cfolds, 1):
        te = np.where(np.array([g in held for g in abskey_px], bool))[0]
        tr = np.where(np.array([g not in held for g in abskey_px], bool))[0]
        # The composition head is refit without the held-out absolute conditions.
        pred_te, net = __import__("dl_model")._fit_torch_bag(
            "mlp", X[tr], Y[tr], maps[tr], X[te], epochs=int(args.epochs), seed=1200+fi,
            return_net=True)
        import torch
        net.eval(); chunks=[]
        with torch.no_grad():
            for st in range(0,len(tr),256): chunks.append(torch.softmax(net(torch.tensor(X[tr][st:st+256])),1).numpy())
        pred_tr = np.vstack(chunks)
        ctx_tr = calibration_context(X[tr], pred_tr, names[tr], ab)
        ctx_te = calibration_context(X[te], pred_te, names[te], ab)
        truth_by_name = {str(n): C[np.where(names == n)[0][0]] for n in dict.fromkeys(names.tolist())}
        Ftr=np.stack([r["feature"] for r in ctx_tr]); Caltr=np.stack([r["Ccal"] for r in ctx_tr]); Ttr=np.stack([truth_by_name[r["map"]] for r in ctx_tr])
        Fte=np.stack([r["feature"] for r in ctx_te]); Calte=np.stack([r["Ccal"] for r in ctx_te]); Tte=np.stack([truth_by_name[r["map"]] for r in ctx_te])
        model,mu,sd,best_ep,best_val=train_residual(Ftr,Caltr,Ttr,seed=2200+fi,max_epochs=800)
        neural,delta=predict_residual(model,Fte,mu,sd,Calte)
        # A transparent linear residual comparator, fit component-wise on present training labels.
        from sklearn.linear_model import Ridge
        ridge=np.empty_like(Tte)
        for j in range(3):
            ok=Ttr[:,j]>0; target=np.log10(Ttr[ok,j])-np.log10(Caltr[ok,j])
            rr=Ridge(alpha=1.0).fit(Ftr[ok],target)
            ridge[:,j]=Calte[:,j]*(10.0**np.clip(rr.predict(Fte),-2,2))
        for i,r in enumerate(ctx_te):
            conc_records.append({"map":r["map"],"fold":fi,"true":Tte[i],"ratio":r["ratio"],"Ieq":r["Ieq"],"Ccal":Calte[i],"ridge":ridge[i],"neural":neural[i],"delta":delta[i],"best_epoch":best_ep,"best_val":best_val})
        dump_json(outdir/f"concentration_fold_{fi}.json", {"records":conc_records,"elapsed_s":time.time()-t0})
        print(f"concentration fold {fi}/5 done; residual epoch={best_ep}", flush=True)

    T=np.stack([r["true"] for r in conc_records]); names_c=np.array([r["map"] for r in conc_records],object)
    pred_methods={"pure_calibration":np.stack([r["Ccal"] for r in conc_records]),"ridge_residual":np.stack([r["ridge"] for r in conc_records]),"neural_residual":np.stack([r["neural"] for r in conc_records])}
    kval=(T>0).sum(1); low=np.all(np.isin(T,[3,6,12,24]),axis=1); inrange=np.max(T,axis=1)<=100
    subsets={"all_no100":np.ones(len(T),bool),"binary":kval==2,"ternary":kval>=3,"low_grid64":low,"calibration_range_max100":inrange}
    conc_summary={m:{g:concentration_stats(T,P,sel) for g,sel in subsets.items()} for m,P in pred_methods.items()}
    dump_json(outdir/"concentration_results.json", {"protocol":{"n_maps":int(len(T)),"split":"absolute-condition-grouped-5-fold","calibration_bands_cm":BAND_CM.tolist(),"calibration_ab":ab.tolist(),"signal_pool":"composition-probability-weighted band mean","residual_hidden":[128,32]},"summary":conc_summary,"records":conc_records})
    with (outdir/"concentration_predictions.csv").open("w",newline="",encoding="utf-8-sig") as f:
        fields=["map","fold","sample_type","component","true_uM","ratio_pred_pct","Ieq_au","Ccal_uM","ridge_uM","neural_uM","neural_recovery_pct","neural_abs_error_uM"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in conc_records:
            typ="binary" if np.sum(np.asarray(r["true"])>0)==2 else "ternary"
            for j,s in enumerate(SUBS):
                t=float(r["true"][j]); p=float(r["neural"][j])
                w.writerow({"map":r["map"],"fold":r["fold"],"sample_type":typ,"component":s,"true_uM":t,"ratio_pred_pct":float(r["ratio"][j]*100),"Ieq_au":float(r["Ieq"][j]),"Ccal_uM":float(r["Ccal"][j]),"ridge_uM":float(r["ridge"][j]),"neural_uM":p,"neural_recovery_pct":(p/t*100 if t>0 else ""),"neural_abs_error_uM":(abs(p-t) if t>0 else "")})
    with (outdir/"concentration_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        fields=["method","subset","n_maps","n_present_components","mean_recovery_pct","median_recovery_pct","mae_uM","rmse_log10","within_1_5x_pct","within_2x_pct","euclidean_mean_uM","euclidean_median_uM"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for m,groups in conc_summary.items():
            for g,v in groups.items(): w.writerow({k:(m if k=="method" else g if k=="subset" else v[k]) for k in fields})
    dump_json(outdir/"run_manifest.json", {"status":"complete","elapsed_s":time.time()-t0,"bundle":str(BUNDLE),"output":str(outdir),"protocol_version":"integrated_v1_20260824"})
    print(f"FULL COMPLETE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); ap.add_argument("--epochs",type=int,default=120); ap.add_argument("--smoke",action="store_true")
    run(ap.parse_args())
