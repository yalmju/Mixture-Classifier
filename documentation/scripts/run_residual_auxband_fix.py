"""TBZ 폭주 대책 2차 — 보조 마커 밴드 feature 주입 (통합 5-fold 프로토타입).

클립 축소는 binary 를 파괴했고 앙상블은 폭주를 못 줄였다 (run_residual_blowup_fix).
폭주의 원인은 주 밴드(TBZ 1270)가 경쟁흡착으로 눌렸을 때 억제량을 추정할 단서가
feature 에 없다는 것 — 그래서 순물질 템플릿에서 성분별 **보조 밴드**(주 밴드에서
떨어져 있고 교차간섭이 낮은 제2 마커)를 자동 선정해 log1p 신호를 feature 에 추가한다
(12 → 15). 나머지 프로토콜은 동일 (abs-condition 5-fold seed 701, MLP ratio 헤드,
3-seed median residual).

실행:  python -u run_residual_auxband_fix.py [--epochs 120]
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
SEEDS = (0, 1, 2)


def pick_aux_bands(P_ref):
    """성분별 보조 밴드: 자기 템플릿 상위 피크 중 주 밴드들에서 ≥25 cm⁻¹ 떨어져
    있고 (자기 신호)/(타 성분 신호 합) 이 가장 큰 채널."""
    P = np.clip(np.asarray(P_ref, float), 0, None)
    aux = []
    for j in range(P.shape[0]):
        own = P[j]
        others = P.sum(0) - own
        ok = np.ones(P.shape[1], bool)
        for c in R.BAND_CM:
            ok &= np.abs(R.AXIS_CM - c) > 25.0
        cand = np.argsort(-own)[:200]
        cand = [i for i in cand if ok[i] and own[i] > 0.2 * own.max()]
        best = max(cand, key=lambda i: own[i] / (others[i] + 1e-9))
        aux.append(float(R.AXIS_CM[best]))
    return np.asarray(aux)


def aux_signal(raw_px, prob_px, aux_cm):
    sig = []
    for j, centre in enumerate(aux_cm):
        mask = np.abs(R.AXIS_CM - centre) <= 10.0
        band = raw_px[:, mask].max(1)
        w = np.clip(prob_px[:, j], 0, None) + 1e-6
        sig.append(float(np.sum(w * band) / np.sum(w)))
    return np.asarray(sig)


def context_aux(X_px, prob_px, names, ab, aux_cm):
    rows = R.calibration_context(X_px, prob_px, names, ab)
    raw = np.expm1(np.clip(np.asarray(X_px, float), 0, None))
    for r, (name, idx) in zip(rows, R.map_rows(names)):
        assert r["map"] == name
        s = aux_signal(raw[idx], np.asarray(prob_px[idx], float), aux_cm)
        r["feature"] = np.concatenate([r["feature"], np.log1p(s)])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "results" / "residual_auxband_fix_20260831"))
    args = ap.parse_args()
    t0 = time.time()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    z = np.load(R.BUNDLE, allow_pickle=True)
    X, Y = z["X"].astype(np.float32), z["Y"].astype(np.float32)
    names, maps, C = z["names"].astype(object), z["mapkey"].astype(object), z["concentration"].astype(float)
    P_ref = z["P_ref"].astype(float)
    ab = np.asarray(json.loads(R.OLD_CONC.read_text(encoding="utf-8"))["excel_loglinear_ab"], float)
    aux_cm = pick_aux_bands(P_ref)
    print("primary bands:", R.BAND_CM.tolist(), " aux bands:", aux_cm.tolist(), flush=True)

    import dl_model, torch

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
        ctx_tr = context_aux(X[tr], pred_tr, names[tr], ab, aux_cm)
        ctx_te = context_aux(X[te], pred_te, names[te], ab, aux_cm)
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
            import torch as _t
            A = _t.tensor(((Fte - mu) / sd).astype(np.float32))
            model.eval()
            with _t.no_grad():
                deltas.append(model(A).numpy())
        dmed = np.median(np.stack(deltas), axis=0)
        for i, r in enumerate(ctx_te):
            records.append({"map": r["map"], "fold": fi, "true": Tte[i].tolist(),
                            "Ccal": Calte[i].tolist(),
                            "aux": (Calte[i] * 10.0 ** np.clip(dmed[i], -2, 2)).tolist(),
                            "aux_seed0": (Calte[i] * 10.0 ** np.clip(deltas[0][i], -2, 2)).tolist()})
        print(f"fold {fi}/5 done ({time.time() - t0:.0f}s)", flush=True)

    R.dump_json(outdir / "records.json", {"aux_bands_cm": aux_cm.tolist(), "records": records})

    T = np.array([r["true"] for r in records])
    k = (T > 0).sum(1); low = np.all(np.isin(T, [3, 6, 12, 24]), axis=1)
    subsets = {"grid64": low, "binary": k == 2, "all_no100": np.ones(len(T), bool)}
    rows = []
    for v in ("aux_seed0", "aux"):
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
