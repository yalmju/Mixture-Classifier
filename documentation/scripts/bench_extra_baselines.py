"""bench_extra_baselines.py — add the two training-free control methods to a finished
benchmark, WITHOUT re-running it.

    python documentation/scripts/bench_extra_baselines.py <export folder>

Reads ``benchmark_predictions.csv`` from a Model-tab Export and derives two more
columns, then re-emits the whole table (existing methods + the two new ones) as
``benchmark_metrics_extended.csv`` and ``benchmark_predictions_extended.csv``.

Why these two, and why they need no maps
----------------------------------------
* **null** — always predicts the mean composition of the OTHER conditions. It is the
  floor every method has to clear: if a method cannot beat "guess the average
  mixture", it has learned nothing about the spectrum. Reviewers ask for this.

* **nnls+rf** — the NNLS composition corrected by one response factor per substance
  (``dl_quantify.fit_response_factors``: r_i from the mean log-ratio of observed to
  prepared). This is the honest control for the paper's claim. If three numbers close
  most of the gap between NNLS and the learned models, then what the learning mainly
  buys is response-factor correction, and the paper should say so; if a large gap
  survives, the learned models are doing something the linear correction cannot.

Both are functions of the per-condition NNLS prediction and the true composition, which
the export already contains — so they can be computed after the fact. Both are fitted
leave-one-condition-out, on the same folds as everything else: the response factors and
the null mean for a held-out condition are computed from the other conditions only.

MCR-ALS is the third control worth having and is NOT here: it re-estimates the component
spectra, so it needs the maps and a fresh pass over them.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dl_quantify import fit_response_factors, apply_response_factors  # noqa: E402
from dl_model import benchmark_summary                                # noqa: E402


def load_predictions(path):
    """benchmark_predictions.csv → (bench dict, subs). Rows are (method, mixture, true…, pred…)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    subs = [h[len("true_"):] for h in head if h.startswith("true_")]
    ti = [head.index(f"true_{s}") for s in subs]
    pi = [head.index(f"pred_{s}") for s in subs]
    bench = {"subs": subs}
    names = {}
    for r in body:
        m = r[0]
        e = bench.setdefault(m, {"true": [], "pred": []})
        e["true"].append([float(r[j]) for j in ti])
        e["pred"].append([float(r[j]) for j in pi])
        names.setdefault(m, []).append(r[1])
    return bench, subs, names


def null_predictions(T):
    """Leave-one-out mean composition: for each condition, the mean of all the others."""
    T = np.asarray(T, float)
    n = len(T)
    tot = T.sum(axis=0)
    out = np.array([(tot - T[i]) / max(n - 1, 1) for i in range(n)])
    s = out.sum(axis=1, keepdims=True)
    return np.divide(out, s, out=np.full_like(out, 1.0 / T.shape[1]), where=s > 0)


def response_corrected(T, P):
    """NNLS composition corrected by per-substance response factors, fitted leave-one-out:
    the factors for condition i come from every condition except i."""
    T = np.asarray(T, float); P = np.asarray(P, float)
    out = np.zeros_like(P)
    for i in range(len(T)):
        keep = np.arange(len(T)) != i
        r = fit_response_factors(P[keep], T[keep])
        out[i] = apply_response_factors(P[i][None, :], r)[0]
    return out


def main(folder):
    src = os.path.join(folder, "benchmark_predictions.csv")
    if not os.path.exists(src):
        raise SystemExit(f"no benchmark_predictions.csv in {folder} — export a benchmark first")
    bench, subs, names = load_predictions(src)
    if "nnls" not in bench:
        raise SystemExit("this export has no 'nnls' column; the response-factor control "
                         "is derived from it")

    T = np.asarray(bench["nnls"]["true"], float)
    Pn = np.asarray(bench["nnls"]["pred"], float)
    bench["null"] = {"true": T.tolist(), "pred": null_predictions(T).tolist()}
    bench["nnls_rf"] = {"true": T.tolist(), "pred": response_corrected(T, Pn).tolist()}
    names["null"] = names["nnls_rf"] = names["nnls"]

    # the cut has to match the one the run declared; read it off the metrics file if present
    cut = None
    mpath = os.path.join(folder, "benchmark_metrics.csv")
    if os.path.exists(mpath):
        with open(mpath, newline="", encoding="utf-8") as f:
            for h in next(csv.reader(f)):
                if h.startswith("accuracy_within_") and h.endswith("pp"):
                    cut = float(h[len("accuracy_within_"):-2])
    summ = benchmark_summary(bench, cut)

    order = [m for m in ("null", "band", "nnls", "nnls_rf", "pls", "rf", "cnn", "mlp")
             if m in summ] + [m for m in summ if m not in
                              ("null", "band", "nnls", "nnls_rf", "pls", "rf", "cnn", "mlp")]
    head = ["method", "n_conditions", "mean_deviation_pp", "rmse_pp", "rmse_se",
            "logratio_distance", "logratio_se", "median_delta_vs_nnls_pp",
            "wilcoxon_p_vs_nnls"]
    for s in subs:
        head += [f"recovery_pct_{s}", f"recovery_se_{s}", f"recovery_n_{s}"]
    if cut is not None:
        head.append(f"accuracy_within_{cut:g}pp")
    out_rows = []
    for m in order:
        e = summ[m]
        rm, rs, _ = e["rmse_pp"]; lm, ls, _ = e["logratio"]
        vr = e.get("vs_ref") or (float("nan"), float("nan"), 0)
        row = [m, e["n"], f"{e['mean_dev_pp']:.2f}", f"{rm:.2f}", f"{rs:.2f}",
               f"{lm:.4f}", f"{ls:.4f}", f"{vr[0]:.2f}", f"{vr[1]:.5f}"]
        for s in subs:
            mu, se, nn = e["recovery"][s]
            row += [f"{mu:.2f}", f"{se:.2f}", str(nn)]
        if cut is not None:
            row.append(f"{e['acc_at_cut']:.4f}")
        out_rows.append(row)

    dst = os.path.join(folder, "benchmark_metrics_extended.csv")
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(head); w.writerows(out_rows)

    dst2 = os.path.join(folder, "benchmark_predictions_extended.csv")
    with open(dst2, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "mixture"] + [f"true_{s}" for s in subs]
                   + [f"pred_{s}" for s in subs])
        for m in order:
            for nm, tv, pv in zip(names[m], bench[m]["true"], bench[m]["pred"]):
                w.writerow([m, nm] + [f"{v:.4f}" for v in tv] + [f"{v:.4f}" for v in pv])

    lab = {"null": "null (mean mix)", "band": "VIP band", "nnls": "NNLS",
           "nnls_rf": "NNLS + resp.factor", "pls": "PLS", "rf": "RF",
           "cnn": "1D-CNN", "mlp": "MLP"}
    print(f"{len(T)} conditions" + (f", declared cut ≤{cut:g} %p" if cut else ", no cut"))
    hdr = (f"{'method':20s} {'dev %p':>7s} {'RMSE %p':>8s} {'log-ratio':>10s} "
           f"{'vs NNLS':>16s}" + (f" {'acc':>5s}" if cut else ""))
    print(hdr); print("-" * len(hdr))
    for m in order:
        e = summ[m]
        vr = e.get("vs_ref")
        v = "reference" if m == "nnls" else (f"{vr[0]:+.1f}, p={vr[1]:.3f}" if vr else "—")
        line = (f"{lab.get(m, m):20s} {e['mean_dev_pp']:7.1f} {e['rmse_pp'][0]:8.1f} "
                f"{e['logratio'][0]:10.2f} {v:>16s}")
        if cut is not None:
            line += f" {e['acc_at_cut']*100:4.0f}%"
        print(line)
    print(f"\nwrote {os.path.basename(dst)} and {os.path.basename(dst2)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
