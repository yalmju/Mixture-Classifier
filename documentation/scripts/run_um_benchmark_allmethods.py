"""8-방법 µM 벤치마크 — 조성 헤드만 교체, 검량선-residual 정량 스테이지는 공유.

조성 벤치마크(v9)와 같은 방법 축(band/nnls/nnls_rf/mcr/pls/rf/cnn/mlp)으로
농도 패널을 만들기 위해, 통합 5-fold(abs-condition, seed 701) 프로토콜에서
ratio 헤드만 바꿔 가며 µM 을 채점한다. 각 방법의 픽셀 조성확률이
밴드신호 가중 → Ccal → 12-feature → neural residual(seed 2200+fold) 로 흐른다.

방법별 npz 체크포인트 — CNN 이 오래 걸리므로 끝난 방법은 재실행 시 건너뛴다.
실행:  python -u run_um_benchmark_allmethods.py [--epochs 120] [--cnn-epochs 120]
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
METHODS = ["nnls", "band", "nnls_rf", "mcr", "pls", "rf", "mlp", "cnn"]  # cnn 마지막


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--cnn-epochs", type=int, default=120)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "results" / "um_benchmark_allmethods_20260831"))
    args = ap.parse_args()
    t0 = time.time()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    z = np.load(R.BUNDLE, allow_pickle=True)
    X, Y = z["X"].astype(np.float32), z["Y"].astype(np.float32)
    names, maps, C = z["names"].astype(object), z["mapkey"].astype(object), z["concentration"].astype(float)
    P_ref = z["P_ref"].astype(float)
    ab = np.asarray(json.loads(R.OLD_CONC.read_text(encoding="utf-8"))["excel_loglinear_ab"], float)

    import dl_model
    from unmix import vip_bands, _vip_fit_mask
    band_mask = _vip_fit_mask(R.AXIS_CM, vip_bands(R.AXIS_CM, P_ref, SUBS), 10.0, len(SUBS))
    print(f"VIP mask: {int(np.asarray(band_mask, bool).sum())}/{len(R.AXIS_CM)} points", flush=True)

    abskey_px = np.array(["c:" + ",".join(f"{v:.8g}" for v in row) for row in C], object)
    cfolds = R.grouped_folds(abskey_px, 5, 701)
    truth_by_name = {str(n): C[np.where(names == n)[0][0]]
                     for n in dict.fromkeys(names.tolist())}

    all_records = {}
    for method in METHODS:
        ck = outdir / f"{method}.npz"
        if ck.exists():
            zz = np.load(ck, allow_pickle=True)
            all_records[method] = json.loads(str(zz["records"]))
            print(f"[skip] {method} — checkpoint exists", flush=True)
            continue
        records = []
        for fi, held in enumerate(cfolds, 1):
            te = np.where(np.array([g in held for g in abskey_px], bool))[0]
            tr = np.where(np.array([g not in held for g in abskey_px], bool))[0]
            ep = int(args.cnn_epochs) if method == "cnn" else int(args.epochs)
            pred_both = dl_model._fit_predict(method, X[tr], Y[tr], np.vstack([X[tr], X[te]]),
                                              epochs=ep, seed=1200 + fi, n_components=8,
                                              n_trees=300, rf_max_features="sqrt",
                                              P_ref=P_ref, band_mask=band_mask,
                                              train_maps=maps[tr])
            pred_tr, pred_te = pred_both[: len(tr)], pred_both[len(tr):]
            ctx_tr = R.calibration_context(X[tr], pred_tr, names[tr], ab)
            ctx_te = R.calibration_context(X[te], pred_te, names[te], ab)
            Ftr = np.stack([r["feature"] for r in ctx_tr]); Caltr = np.stack([r["Ccal"] for r in ctx_tr])
            Ttr = np.stack([truth_by_name[r["map"]] for r in ctx_tr])
            Fte = np.stack([r["feature"] for r in ctx_te]); Calte = np.stack([r["Ccal"] for r in ctx_te])
            model, mu, sd, best_ep, _ = R.train_residual(Ftr, Caltr, Ttr,
                                                         seed=2200 + fi, max_epochs=800)
            neural, _ = R.predict_residual(model, Fte, mu, sd, Calte)
            for i, r in enumerate(ctx_te):
                records.append({"map": r["map"], "fold": fi,
                                "true": truth_by_name[r["map"]].tolist(),
                                "ratio": np.asarray(r["ratio"]).tolist(),
                                "Ccal": Calte[i].tolist(),
                                "uM": np.asarray(neural)[i].tolist()})
            print(f"{method} fold {fi}/5 done ({time.time() - t0:.0f}s)", flush=True)
        np.savez(ck, records=json.dumps(records))
        all_records[method] = records
        print(f"[saved] {method}", flush=True)

    # ---- 채점: grid64, 부트스트랩 CI ------------------------------------------------
    rng = np.random.default_rng(0)
    rows = []
    for method in METHODS:
        recs = all_records[method]
        T = np.array([r["true"] for r in recs]); P = np.array([r["uM"] for r in recs])
        low = np.all(np.isin(T, [3, 6, 12, 24]), axis=1)
        Tg, Pg = T[low], P[low]
        n = len(Tg)

        def met(idx):
            t = Tg[idx].ravel(); p = Pg[idx].ravel()
            fold = p / t
            return (float(np.sqrt(np.mean((p - t) ** 2))),
                    float(np.mean((fold >= 0.5) & (fold <= 2.0)) * 100),
                    float(np.sqrt(np.mean(np.log10(np.clip(fold, 1e-8, None)) ** 2))),
                    float(np.median(fold) * 100))

        pt = met(np.arange(n))
        boot = np.array([met(rng.integers(0, n, n)) for _ in range(2000)])
        lo = np.percentile(boot, 2.5, axis=0); hi = np.percentile(boot, 97.5, axis=0)
        rec_stats = {}
        for j, s in enumerate(SUBS):
            q = Pg[:, j] / Tg[:, j] * 100
            rec_stats[s] = (float(np.mean(q)), float(np.std(q, ddof=1)),
                            float(np.std(q, ddof=1) / np.sqrt(len(q))))
        rows.append({"method": method, "rmse_uM": pt[0], "rmse_lo": lo[0], "rmse_hi": hi[0],
                     "within2x_pct": pt[1], "w2x_lo": lo[1], "w2x_hi": hi[1],
                     "rmse_log10": pt[2], "median_recovery_pct": pt[3],
                     **{f"recovery_{s}_mean": rec_stats[s][0] for s in SUBS},
                     **{f"recovery_{s}_sd": rec_stats[s][1] for s in SUBS},
                     **{f"recovery_{s}_se": rec_stats[s][2] for s in SUBS}})
        print("%-8s RMSE %6.2f [%5.2f-%5.2f]  w2x %5.1f%% [%4.1f-%4.1f]  RMSElog %.3f  "
              "rec DQ %.0f±%.0f TBZ %.0f±%.0f THI %.0f±%.0f"
              % (method, pt[0], lo[0], hi[0], pt[1], lo[1], hi[1], pt[2],
                 rec_stats["DQ"][0], rec_stats["DQ"][1], rec_stats["TBZ"][0],
                 rec_stats["TBZ"][1], rec_stats["THI"][0], rec_stats["THI"][1]), flush=True)

    with (outdir / "um_benchmark_scores.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
