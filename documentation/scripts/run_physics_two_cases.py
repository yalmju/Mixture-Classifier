"""Paired MLP ablation for the max-rescue and THI-overestimate cases."""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

import physics_max_rescue_experiment as exp
from run_physics_max_rescue_fixed import physics_pretrain_fixed

CASES = [(12, 6, 3), (24, 3, 12)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=120)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    folder = os.path.join(a.db, "Ratio", "260814_mixture_final")
    pure = os.path.join(a.db, "Pure")
    low, high = exp.parse_folder(folder)
    keys = sorted(c for _, c in low)
    order = np.random.default_rng(0).permutation(len(keys))
    folds = [[keys[i] for i in order[f::4]] for f in range(4)]
    target_fold = next(i for i, fold in enumerate(folds) if CASES[0] in fold)
    held = set(folds[target_fold])
    train = high + [(p, c) for p, c in low if c not in held]
    print(f"fold={target_fold + 1} train={len(train)} held={len(held)}", flush=True)
    _, _, mask, P, _, _ = exp._refs(pure, True, None)
    X, Y, C, S, _ = exp.gather_rows(pure, train, P, 20)
    preX, preY, cal, m = physics_pretrain_fixed(
        pure, exp.calibration_path(a.db), P, mask
    )
    models = exp.fit_models(X, Y, C, S, (preX, preY), cal, m, a.epochs)
    rows = []
    for conc in CASES:
        path = next(p for p, c in low if c == conc)
        true = exp._ratio(np.asarray(conc, float))
        _, _, _, _, _, nnls = exp.nnls_and_spectrum(pure, path)
        Xt = exp.target_features(pure, path)
        mlp = exp.predict(models[0.0][0], Xt)
        phys = exp.predict(models[.03][0], Xt)
        for method, pred in [("True", true), ("NNLS", nnls),
                             ("MLP", mlp), ("MLP + Physics", phys)]:
            error = 0.0 if method == "True" else float(
                0.5 * np.abs(pred - true).sum() * 100
            )
            rows.append({
                "case": "max_rescue" if conc == CASES[0] else "THI_overestimate",
                "condition": f"DQ{conc[0]}-TBZ{conc[1]}-THI{conc[2]}",
                "method": method,
                "DQ_pct": pred[0] * 100,
                "TBZ_pct": pred[1] * 100,
                "THI_pct": pred[2] * 100,
                "composition_error_pct": error,
                "THI_bias_pct_point": (pred[2] - true[2]) * 100,
                "held_out_fold": target_fold + 1,
                "surface_loss_weight": .03 if method == "MLP + Physics" else 0,
            })
        wn, raw, y, P2, B, _ = exp.nnls_and_spectrum(pure, path)
        dom = int(np.argmax(B)); residual = np.clip(y - B[dom] * P2[dom], 0, None)
        spath = os.path.join(a.out, f"panel_h_{rows[-1]['case']}_spectrum.csv")
        with open(spath, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["raman_shift_cm-1", "measured_raw", "preprocessed",
                        "dominant_component_fit", "residual"])
            w.writerows(zip(wn, raw, y, B[dom] * P2[dom], residual))
    out = os.path.join(a.out, "panel_h_two_case_comparison.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(out, flush=True)
    for row in rows: print(row, flush=True)


if __name__ == "__main__":
    main()
