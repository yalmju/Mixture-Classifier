"""Residual 학습기 교체 A/B — "더 잘하는 녀석" 탐색 (통합 5-fold 프로토타입).

같은 abs-condition 5-fold(seed 701, MLP ratio 헤드)에서 Δlog10(C/Ccal) 회귀기를
교체 채점한다. 89맵×12피처 스케일에서는 작은 신경망보다 부스팅/GP 가 나을 수 있다.

  neural   현행 128→32 net (기준)
  hgb      HistGradientBoosting, 성분별
  rf       RandomForest(500), 성분별
  gp       GaussianProcess(RBF+White), 성분별, 표준화 피처
  stack    neural·hgb 의 Δ 평균
  ktotal   (참고 상한) 총농도 참값 선언 시 Cᵢ = ratioᵢ × C_total — 앱 known-total 모드

실행:  python -u run_residual_model_zoo.py [--epochs 120]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

MAIN = Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(MAIN / "documentation" / "scripts"))
import run_integrated_reevaluation as R  # noqa: E402

SUBS = R.SUBS


def fit_percomp(maker, Ftr, Caltr, Ttr, Fte, Calte):
    """성분별로 Δlog10 을 회귀하고 µM 예측을 돌려준다 (present 라벨만 학습)."""
    pred = np.empty((len(Fte), 3))
    for j in range(3):
        ok = Ttr[:, j] > 0
        y = np.log10(np.clip(Ttr[ok, j], 0.05, None)) - np.log10(np.clip(Caltr[ok, j], 0.05, None))
        est = maker()
        est.fit(Ftr[ok], y)
        d = np.clip(est.predict(Fte), -2, 2)
        pred[:, j] = Calte[:, j] * 10.0 ** d
    return np.clip(pred, 0.001, 5000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "results" / "residual_model_zoo_20260831"))
    args = ap.parse_args()
    t0 = time.time()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    z = np.load(R.BUNDLE, allow_pickle=True)
    X, Y = z["X"].astype(np.float32), z["Y"].astype(np.float32)
    names, maps, C = z["names"].astype(object), z["mapkey"].astype(object), z["concentration"].astype(float)
    ab = np.asarray(json.loads(R.OLD_CONC.read_text(encoding="utf-8"))["excel_loglinear_ab"], float)

    import dl_model, torch
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    makers = {
        "hgb": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                                     max_leaf_nodes=15, min_samples_leaf=4,
                                                     l2_regularization=1.0, random_state=0),
        "rf": lambda: RandomForestRegressor(500, min_samples_leaf=2, random_state=0, n_jobs=-1),
        "gp": lambda: make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(kernel=ConstantKernel() * RBF(length_scale=3.0)
                                     + WhiteKernel(1e-2), normalize_y=True, random_state=0)),
    }

    abskey_px = np.array(["c:" + ",".join(f"{v:.8g}" for v in row) for row in C], object)
    cfolds = R.grouped_folds(abskey_px, 5, 701)
    records = []
    for fi, held in enumerate(cfolds, 1):
        te = np.where(np.array([g in held for g in abskey_px], bool))[0]
        tr = np.where(np.array([g not in held for g in abskey_px], bool))[0]
        pred_te, net = dl_model._fit_torch_bag("mlp", X[tr], Y[tr], maps[tr], X[te],
                                               epochs=int(args.epochs), seed=1200 + fi,
                                               return_net=True)
        net.eval(); chunks = []
        with torch.no_grad():
            for st in range(0, len(tr), 256):
                chunks.append(torch.softmax(net(torch.tensor(X[tr][st:st + 256])), 1).numpy())
        pred_tr = np.vstack(chunks)
        ctx_tr = R.calibration_context(X[tr], pred_tr, names[tr], ab)
        ctx_te = R.calibration_context(X[te], pred_te, names[te], ab)
        truth_by_name = {str(n): C[np.where(names == n)[0][0]]
                         for n in dict.fromkeys(names.tolist())}
        Ftr = np.stack([r["feature"] for r in ctx_tr]); Caltr = np.stack([r["Ccal"] for r in ctx_tr])
        Ttr = np.stack([truth_by_name[r["map"]] for r in ctx_tr])
        Fte = np.stack([r["feature"] for r in ctx_te]); Calte = np.stack([r["Ccal"] for r in ctx_te])
        Tte = np.stack([truth_by_name[r["map"]] for r in ctx_te])

        model, mu, sd, best_ep, _ = R.train_residual(Ftr, Caltr, Ttr, seed=2200 + fi, max_epochs=800)
        neural, delta_n = R.predict_residual(model, Fte, mu, sd, Calte)
        preds = {"neural": np.asarray(neural)}
        for name, mk in makers.items():
            preds[name] = fit_percomp(mk, Ftr, Caltr, Ttr, Fte, Calte)
        d_hgb = np.log10(np.clip(preds["hgb"], 1e-6, None) / np.clip(Calte, 1e-6, None))
        d_stack = (np.clip(delta_n, -2, 2) + d_hgb) / 2
        preds["stack"] = np.clip(Calte * 10.0 ** d_stack, 0.001, 5000.0)
        ratio_te = np.stack([r["ratio"] for r in ctx_te])
        preds["ktotal"] = ratio_te * Tte.sum(1, keepdims=True)   # 참고 상한 (총농도 참값 선언)
        for i, r in enumerate(ctx_te):
            rec = {"map": r["map"], "fold": fi, "true": Tte[i].tolist()}
            for k, P in preds.items():
                rec[k] = np.asarray(P)[i].tolist()
            records.append(rec)
        print(f"fold {fi}/5 done ({time.time() - t0:.0f}s)", flush=True)

    R.dump_json(outdir / "records.json", {"records": records})
    T = np.array([r["true"] for r in records])
    kk = (T > 0).sum(1); low = np.all(np.isin(T, [3, 6, 12, 24]), axis=1)
    subsets = {"grid64": low, "binary": kk == 2, "all_no100": np.ones(len(T), bool)}
    variants = ["neural", "hgb", "rf", "gp", "stack", "ktotal"]
    rows = []
    for v in variants:
        P = np.array([r[v] for r in records])
        for g, sel in subsets.items():
            t, p = T[sel], P[sel]; ok = t > 0
            e = p[ok] - t[ok]; fold = p[ok] / t[ok]
            worst = np.array([np.max(np.maximum(p[i] / np.maximum(t[i], 1e-9),
                                                t[i] / np.maximum(p[i], 1e-9))[t[i] > 0])
                              for i in range(len(t))])
            rows.append({"variant": v, "subset": g,
                         "rmse_uM": float(np.sqrt(np.mean(e ** 2))),
                         "mae_uM": float(np.mean(np.abs(e))),
                         "rmse_log10": float(np.sqrt(np.mean(np.log10(np.clip(fold, 1e-8, None)) ** 2))),
                         "within_2x_pct": float(np.mean((fold >= 0.5) & (fold <= 2.0)) * 100),
                         "allcomp_within_2x_pct": float(np.mean(worst <= 2) * 100),
                         "maps_gt3x": int(np.sum(worst > 3))})
    with (outdir / "variant_scores.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print("%-7s %-9s RMSE %7.2f  MAE %6.2f  RMSElog %.3f  w2x %5.1f%%  all-2x %5.1f%%  >3x %2d"
              % (r["variant"], r["subset"], r["rmse_uM"], r["mae_uM"], r["rmse_log10"],
                 r["within_2x_pct"], r["allcomp_within_2x_pct"], r["maps_gt3x"]), flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
