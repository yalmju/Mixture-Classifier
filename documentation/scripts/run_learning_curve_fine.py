# -*- coding: utf-8 -*-
"""세분화 학습 곡선 — "조성 조건이 몇 개면 충분한가"를 포화 곡선 적합으로.

run_capacity_ablation 의 (B)를 확장: 비율 7단 × 서브샘플 3반복(제외 조건 추첨의
운을 오차로), MLP(256,64) + PLS. 끝에 dev = a·N^(-b) + c 를 적합해 목표 오차까지
필요한 조건 수를 외삽한다.

실행:  python -u run_learning_curve_fine.py
"""
from __future__ import annotations

import csv
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

OUT = Path(__file__).resolve().parents[1] / "results" / "learning_curve_fine_20260901"
OUT.mkdir(parents=True, exist_ok=True)
FRACS = (0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0)
REPS = 3


def _pool(pred, Yte, nte):
    ts, ps = [], []
    for name, idx in R.map_rows(nte):
        ts.append(np.asarray(Yte[idx[0]], float))
        ps.append(np.asarray(pred[idx], float).mean(0))
    return np.stack(ts), np.stack(ps)


def dev_of(rows):
    T, P = rows
    return float((0.5 * np.abs(P - T).sum(1) * 100).mean())


def main():
    t0 = time.time()
    z = np.load(R.BUNDLE, allow_pickle=True)
    X, Y = z["X"].astype(np.float32), z["Y"].astype(np.float32)
    names, maps = z["names"].astype(object), z["mapkey"].astype(object)
    old = json.loads(R.OLD_FOLDS.read_text(encoding="utf-8"))
    fold_by = {str(n): int(f) for n, f in zip(old["condition"], old["fold"])}
    fold_px = np.array([fold_by[str(n)] for n in names], int)
    conds = np.asarray([str(n) for n in names], object)
    uniq = list(dict.fromkeys(conds.tolist()))
    res = []
    for frac in FRACS:
        for rep in range(REPS if frac < 1.0 else 1):
            rng = np.random.default_rng(100 + rep)
            keep = set(rng.choice(uniq, size=max(8, int(len(uniq) * frac)),
                                  replace=False)) if frac < 1.0 else set(uniq)
            tr_extra = np.where(np.isin(conds, list(keep)))[0]
            for tag in ("mlp", "pls"):
                Ts, Ps = [], []
                for fi in range(1, 6):
                    te = np.where(fold_px == fi)[0]
                    tr = np.where(fold_px != fi)[0]
                    tr = tr[np.isin(tr, tr_extra)]
                    if tag == "mlp":
                        pred = dl_model._fit_predict("mlp", X[tr], Y[tr], X[te],
                                                     epochs=120, seed=170 + fi,
                                                     train_maps=maps[tr])
                    else:
                        pred = dl_model._fit_predict("pls", X[tr], Y[tr], X[te],
                                                     n_components=8)
                    t_, p_ = _pool(pred, Y[te], names[te])
                    Ts.append(t_); Ps.append(p_)
                d = dev_of((np.vstack(Ts), np.vstack(Ps)))
                n_cond = len(keep)
                res.append({"frac": frac, "n_conditions": n_cond, "rep": rep,
                            "model": tag, "dev_pp": round(d, 3)})
                print(f"{tag} frac {frac:.0%} rep {rep} n={n_cond}  dev {d:.2f}pp"
                      f"  ({time.time()-t0:.0f}s)", flush=True)
    with (OUT / "learning_fine.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader(); w.writerows(res)
    # ── 포화 적합: dev = a·N^-b + c (MLP) → 목표 오차 외삽 ──
    from scipy.optimize import curve_fit
    mlp = [(r["n_conditions"], r["dev_pp"]) for r in res if r["model"] == "mlp"]
    N = np.array([n for n, _ in mlp], float); D = np.array([d for _, d in mlp], float)
    try:
        (a, b, c), _ = curve_fit(lambda n, a, b, c: a * n**(-b) + c, N, D,
                                 p0=(50, 0.5, 10), maxfev=20000)
        print(f"fit: dev = {a:.1f}*N^(-{b:.2f}) + {c:.2f}  (asymptote {c:.1f}pp)")
        for target in (16.0, 15.0, 14.0, 13.0):
            if target <= c:
                print(f"  {target}pp: unreachable (asymptote {c:.1f})")
            else:
                need = (a / (target - c)) ** (1 / b)
                print(f"  {target}pp reached at ~{need:.0f} conditions")
        with (OUT / "fit.json").open("w", encoding="utf-8") as f:
            json.dump({"a": a, "b": b, "c": c}, f)
    except Exception as e:
        print("fit failed:", e)
    print(f"done {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
