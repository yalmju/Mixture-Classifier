"""TBZ 폭주(과대 Δ 보정) 억제 대책 A/B — 통합 5-fold 프로토콜에서 프로토타입.

배경: TB24 계열에서 TBZ 1270 밴드가 경쟁흡착으로 눌려 Ccal 이 수 배 과소 →
residual 이 +1 decade 급 보정을 해야 하고, 지수형(C=Ccal·10^Δ)이라 held-out 에서
Δ 를 과하게 잡으면 93~129 µM 로 폭주한다 (09b 관찰). 제약은 전부 범위 제약이라
안 걸렸다. 여기서는 배포 구조를 건드리기 전에 같은 abs-condition 5-fold(seed 701,
MLP ratio 헤드 = 배포와 동일 구성)에서 다음을 채점한다:

  base      Δ clip ±2.0 (현행)          — integrated_final full 재현 확인용
  clip1     Δ clip ±1.0
  clip07    Δ clip ±0.7
  ens       3-seed residual median Δ, clip ±2.0
  ens_clip1 3-seed median Δ, clip ±1.0

Δ 클립은 예측 시에만 적용되므로 net 은 seed 당 한 번만 학습하면 된다.
실행:  python -u run_residual_blowup_fix.py [--epochs 120]
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
SEEDS = (0, 1, 2)                        # residual 앙상블 seed 오프셋
CLIPS = {"base": 2.0, "clip1": 1.0, "clip07": 0.7}


def predict_delta(model, X, mu, sd):
    import torch
    A = torch.tensor(((np.asarray(X, np.float32) - mu) / sd).astype(np.float32))
    model.eval()
    with torch.no_grad():
        return model(A).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "results" / "residual_blowup_fix_20260831"))
    args = ap.parse_args()
    t0 = time.time()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    z = np.load(R.BUNDLE, allow_pickle=True)
    X, Y = z["X"].astype(np.float32), z["Y"].astype(np.float32)
    names, maps, C = z["names"].astype(object), z["mapkey"].astype(object), z["concentration"].astype(float)
    ab = np.asarray(json.loads(R.OLD_CONC.read_text(encoding="utf-8"))["excel_loglinear_ab"], float)

    import dl_model

    abskey_px = np.array(["c:" + ",".join(f"{v:.8g}" for v in row) for row in C], object)
    cfolds = R.grouped_folds(abskey_px, 5, 701)
    records = []
    for fi, held in enumerate(cfolds, 1):
        te = np.where(np.array([g in held for g in abskey_px], bool))[0]
        tr = np.where(np.array([g not in held for g in abskey_px], bool))[0]
        pred_te, net = dl_model._fit_torch_bag("mlp", X[tr], Y[tr], maps[tr], X[te],
                                               epochs=int(args.epochs), seed=1200 + fi,
                                               return_net=True)
        import torch
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
        deltas = []
        for k in SEEDS:
            model, mu, sd, best_ep, _ = R.train_residual(Ftr, Caltr, Ttr,
                                                         seed=2200 + fi + 1000 * k,
                                                         max_epochs=800)
            deltas.append(predict_delta(model, Fte, mu, sd))
            print(f"fold {fi}/5 seed {k} residual epoch={best_ep}", flush=True)
        d0 = deltas[0]
        dmed = np.median(np.stack(deltas), axis=0)
        for i, r in enumerate(ctx_te):
            rec = {"map": r["map"], "fold": fi, "true": Tte[i].tolist(),
                   "Ccal": Calte[i].tolist()}
            for name, clip in CLIPS.items():
                rec[name] = (Calte[i] * 10.0 ** np.clip(d0[i], -clip, clip)).tolist()
            rec["ens"] = (Calte[i] * 10.0 ** np.clip(dmed[i], -2, 2)).tolist()
            rec["ens_clip1"] = (Calte[i] * 10.0 ** np.clip(dmed[i], -1, 1)).tolist()
            rec["deltas"] = [d[i].tolist() for d in deltas]
            records.append(rec)
        print(f"fold {fi}/5 done  ({time.time() - t0:.0f}s)", flush=True)

    R.dump_json(outdir / "records.json", {"records": records})

    T = np.array([r["true"] for r in records])
    k = (T > 0).sum(1); low = np.all(np.isin(T, [3, 6, 12, 24]), axis=1)
    subsets = {"grid64": low, "binary": k == 2, "all_no100": np.ones(len(T), bool)}
    variants = ["base", "clip1", "clip07", "ens", "ens_clip1"]
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
                         "maps_gt3x": int(np.sum(worst > 3)),
                         "worst_fold_max": float(worst.max())})
    with (outdir / "variant_scores.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print("%-9s %-9s RMSE %7.2f  MAE %6.2f  RMSElog %.3f  w2x %5.1f%%  >3x %2d  worstmax %5.1fx"
              % (r["variant"], r["subset"], r["rmse_uM"], r["mae_uM"], r["rmse_log10"],
                 r["within_2x_pct"], r["maps_gt3x"], r["worst_fold_max"]), flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
