"""bench_by_concentration_arm.py — run the benchmark separately on the saturated and the
working-range conditions, because averaging the two hides the effect the paper is about.

    python documentation/scripts/bench_by_concentration_arm.py [--split 50] [--quick]

Why
---
The paper's panel (a) triangle shows NNLS badly overestimating THI, and the learned model
tight. The 9-method benchmark over all conditions shows NNLS 19.8 %p against MLP 16.2 %p
— dramatic in the figure, 3.6 %p in the table. Both are correct; they are different
condition sets. export_nnls.py draws panel (a) with `--set high`, and says why in its own
header: on the low-concentration grid NNLS is already green (composition error 15.1 %p,
recovery 109/120/112%), while the high-concentration arm is where THI monopolises the
surface and linear unmixing breaks.

So the honest comparison is stratified, not averaged. This runs the same
leave-one-condition-out benchmark twice — once on conditions whose largest component is
at or above ``--split`` uM (the saturated arm), once below (the working range) — and
prints both tables. The expected shape, if the paper's reading is right: the methods
separate in the saturated arm and converge in the working range.

Note on folds: conditions are grouped by RATIO, and one ratio (1:1:1) was measured in
both arms, so it is dropped from both to keep each arm clean. The script says how many.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataset import load_mixture_list, load_preprocess   # noqa: E402
from dl_model import benchmark_loo, benchmark_summary     # noqa: E402
from dl_quantify import _ratio                            # noqa: E402
from real_data import PEST_DEFAULT                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=PEST_DEFAULT)
    ap.add_argument("--split", type=float, default=50.0, help="uM boundary between arms")
    ap.add_argument("--cut", type=float, default=15.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cfg = load_preprocess(a.data_dir)
    items = load_mixture_list(a.data_dir)
    if not items or len(items[0]) < 3:
        raise SystemExit("mixtures.json has no per-substance uM — cannot split by arm")

    subs0 = sorted({s for _p, r, *_c in items for s in r})

    def key(r):
        return ",".join(f"{v:.4f}" for v in _ratio([float(r.get(s, 0.0)) for s in subs0]))

    def top_uM(it):
        return max([float(v) * 1e6 for v in it[2].values()] or [0.0])

    hi = [it for it in items if top_uM(it) >= a.split]
    lo = [it for it in items if top_uM(it) < a.split]
    kh, kl = {key(it[1]) for it in hi}, {key(it[1]) for it in lo}
    shared = kh & kl
    if shared:
        hi = [it for it in hi if key(it[1]) not in shared]
        lo = [it for it in lo if key(it[1]) not in shared]
    print(f"saturated arm (>= {a.split:g} uM): {len(hi)} maps / {len(kh - shared)} ratios")
    print(f"working range (<  {a.split:g} uM): {len(lo)} maps / {len(kl - shared)} ratios")
    print(f"dropped from both — ratios measured in BOTH arms: {len(shared)}\n")

    methods = (("pls", "mlp") if a.quick else
               ("null", "band", "nnls", "nnls_rf", "mcr", "pls", "rf", "cnn", "mlp"))
    out = {}
    for tag, sub in (("saturated", hi), ("working", lo)):
        if len(sub) < 4:
            print(f"[{tag}] too few conditions — skipped"); continue
        print(f"--- {tag} ---", flush=True)
        b = benchmark_loo(a.data_dir, sub, baseline=cfg["baseline"], trim=cfg["trim"],
                          methods=methods, epochs=120 if a.quick else 350,
                          rf_max_features="sqrt", px_per_map=0,
                          progress=lambda s: print("   ·", s, flush=True) if "1/" in s else None)
        out[tag] = benchmark_summary(b, a.cut)

    arms = list(out)
    hdr = (f"{'method':10s}" + "".join(f"{t + ' %p':>14s}" for t in arms)
           + "".join(f"{t + ' acc':>12s}" for t in arms))
    print("\n=== mean composition deviation, by concentration arm ===")
    print(hdr); print("-" * len(hdr))
    for m in methods:
        if not all(m in out[t] for t in arms):
            continue
        row = f"{m:10s}"
        row += "".join(f"{out[t][m]['mean_dev_pp']:13.1f} " for t in arms)
        row += "".join(f"{out[t][m].get('acc_at_cut', float('nan'))*100:11.0f}%" for t in arms)
        print(row)
    for t in arms:
        print(f"\n[{t}] n = {out[t][methods[0]]['n']} conditions")

    if a.out:
        import csv
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["arm", "method", "n", "mean_deviation_pp", "rmse_pp", "logratio",
                        f"accuracy_within_{a.cut:g}pp"]
                       + [f"recovery_pct_{s}" for s in subs0])
            for t in arms:
                for m in out[t]:
                    e = out[t][m]
                    w.writerow([t, m, e["n"], f"{e['mean_dev_pp']:.3f}",
                                f"{e['rmse_pp'][0]:.3f}", f"{e['logratio'][0]:.4f}",
                                f"{e.get('acc_at_cut', float('nan')):.4f}"]
                               + [f"{e['recovery'].get(s, (float('nan'),))[0]:.1f}" for s in subs0])
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
