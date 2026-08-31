"""PLS-ratio variant of the integrated concentration pipeline (figure1d 대응).

figure1d_calibration_mae_heatmap.csv (2026-08-24)는 integrated_final_20260824/full
의 neural_uM — 즉 nnls_screen → **MLP** composition ratio → 검량선 Ccal → neural
residual — 을 grid64 64조건으로 잘라 조건별 |오차| 평균을 낸 것이다. 여기서는
농도 스테이지의 ratio 헤드만 MLP → PLS(_fit_predict "pls", n_components=8)로
바꾸고 나머지(절대조건 grouped 5-fold, seed 701 동일 fold, 검량선 ab, residual
넷 구조·seed)는 그대로 둔다. Ccal 도 ratio 확률로 가중되므로 함께 달라진다.

실행:  python -u run_integrated_pls_concentration.py [--epochs 120] [--out DIR]
출력:  figure1d 와 같은 포맷의 per-map CSV (single/neural 오차, MAE·RMSE·euclid)
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
sys.path.insert(0, str(MAIN))                            # dl_model
sys.path.insert(0, str(MAIN / "documentation" / "scripts"))
import run_integrated_reevaluation as R                  # noqa: E402  (함수 재사용)

SUBS = R.SUBS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)   # ratio 헤드가 PLS라 미사용, 기록용
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "results" / "integrated_pls_20260831"))
    args = ap.parse_args()
    t0 = time.time()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    z = np.load(R.BUNDLE, allow_pickle=True)
    X, Y = z["X"].astype(np.float32), z["Y"].astype(np.float32)
    names, maps, C = z["names"].astype(object), z["mapkey"].astype(object), z["concentration"].astype(float)
    P_ref = z["P_ref"].astype(float)
    ab = np.asarray(json.loads(R.OLD_CONC.read_text(encoding="utf-8"))["excel_loglinear_ab"], float)

    from dl_model import _fit_predict

    abskey_px = np.array(["c:" + ",".join(f"{v:.8g}" for v in row) for row in C], object)
    cfolds = R.grouped_folds(abskey_px, 5, 701)          # 원 실행과 동일한 fold
    conc_records = []
    for fi, held in enumerate(cfolds, 1):
        te = np.where(np.array([g in held for g in abskey_px], bool))[0]
        tr = np.where(np.array([g not in held for g in abskey_px], bool))[0]
        # ratio 헤드: PLS. 학습픽셀 in-sample 예측 + held-out 예측을 한 번에 뽑는다
        # (MLP 갈래도 학습 net 을 자기 학습픽셀에 다시 적용했으므로 같은 취급이다).
        pred_both = _fit_predict("pls", X[tr], Y[tr], np.vstack([X[tr], X[te]]),
                                 n_components=8, seed=1200 + fi)
        pred_tr, pred_te = pred_both[: len(tr)], pred_both[len(tr):]
        ctx_tr = R.calibration_context(X[tr], pred_tr, names[tr], ab)
        ctx_te = R.calibration_context(X[te], pred_te, names[te], ab)
        truth_by_name = {str(n): C[np.where(names == n)[0][0]]
                         for n in dict.fromkeys(names.tolist())}
        Ftr = np.stack([r["feature"] for r in ctx_tr]); Caltr = np.stack([r["Ccal"] for r in ctx_tr])
        Ttr = np.stack([truth_by_name[r["map"]] for r in ctx_tr])
        Fte = np.stack([r["feature"] for r in ctx_te]); Calte = np.stack([r["Ccal"] for r in ctx_te])
        Tte = np.stack([truth_by_name[r["map"]] for r in ctx_te])
        model, mu, sd, best_ep, best_val = R.train_residual(Ftr, Caltr, Ttr,
                                                            seed=2200 + fi, max_epochs=800)
        neural, delta = R.predict_residual(model, Fte, mu, sd, Calte)
        for i, r in enumerate(ctx_te):
            conc_records.append({"map": r["map"], "fold": fi, "true": Tte[i],
                                 "ratio": r["ratio"], "Ccal": Calte[i], "neural": neural[i]})
        print(f"concentration fold {fi}/5 done; residual epoch={best_ep}", flush=True)

    T = np.stack([r["true"] for r in conc_records])
    low = np.all(np.isin(T, [3, 6, 12, 24]), axis=1)
    summary = {m: {g: R.concentration_stats(T, P, sel)
                   for g, sel in {"all_no100": np.ones(len(T), bool), "low_grid64": low}.items()}
               for m, P in {"pure_calibration": np.stack([r["Ccal"] for r in conc_records]),
                            "neural_residual": np.stack([r["neural"] for r in conc_records])}.items()}
    R.dump_json(outdir / "concentration_results_pls.json",
                {"protocol": {"ratio_head": "pls_n_components_8",
                              "split": "absolute-condition-grouped-5-fold (seed 701, 원 실행과 동일)",
                              "residual": "동일 neural residual (hidden 128/32)"},
                 "summary": summary})

    # figure1d 와 같은 포맷 — grid64 64조건, 조건별 오차 (MAE / RMSE / euclid)
    import re
    pat = re.compile(r"DQ(\d+)-TB(\d+)-TH(\d+)")
    with (outdir / "figure1d_pls_calibration_mae_heatmap.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["map", "single_abs_error_uM", "neural_abs_error_uM",
                    "single_rmse_uM", "neural_rmse_uM",
                    "single_euclidean_uM", "neural_euclidean_uM", "DQ", "TBZ", "THI"])
        for r in conc_records:
            m = pat.match(str(r["map"]))
            if not m:
                continue
            k = tuple(int(x) for x in m.groups())
            if not all(v in (3, 6, 12, 24) for v in k):
                continue
            t = np.asarray(r["true"], float)
            e_cal = np.abs(np.asarray(r["Ccal"], float) - t)
            e_neu = np.abs(np.asarray(r["neural"], float) - t)
            w.writerow([r["map"], float(e_cal.mean()), float(e_neu.mean()),
                        float(np.sqrt((e_cal ** 2).mean())), float(np.sqrt((e_neu ** 2).mean())),
                        float(np.sqrt((e_cal ** 2).sum())), float(np.sqrt((e_neu ** 2).sum())),
                        *k])
    with (outdir / "concentration_predictions_pls.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["map", "fold", "component", "true_uM", "ratio_pred_pct", "Ccal_uM",
                  "neural_uM", "neural_abs_error_uM"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in conc_records:
            for j, s in enumerate(SUBS):
                t = float(r["true"][j]); p = float(r["neural"][j])
                w.writerow({"map": r["map"], "fold": r["fold"], "component": s,
                            "true_uM": t, "ratio_pred_pct": float(r["ratio"][j] * 100),
                            "Ccal_uM": float(r["Ccal"][j]), "neural_uM": p,
                            "neural_abs_error_uM": (abs(p - t) if t > 0 else "")})
    for m, groups in summary.items():
        for g, v in groups.items():
            print(f"{m:>18} {g:<12} MAE {v['mae_uM']:.3f} µM  within-2x {v['within_2x_pct']:.1f}%")
    print(f"done in {time.time() - t0:.0f}s -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
