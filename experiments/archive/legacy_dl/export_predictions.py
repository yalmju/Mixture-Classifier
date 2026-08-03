# -*- coding: utf-8 -*-
"""One clean prediction table: per mixture, true vs NNLS / linear-RF / DL composition.
Binary = leave-one-map-out (honest); ternary = trained on all 28 binary (generalization).
Writes docs/mixture_predictions.csv.
"""
import os, sys, glob, csv
sys.path.insert(0, os.getcwd())
import numpy as np

root = "S:/Google Drive"
acf = next(os.path.join(e.path, "ACF_PEST_DB") for e in os.scandir(root)
           if e.is_dir() and os.path.isdir(os.path.join(e.path, "ACF_PEST_DB")))
pure = os.path.join(acf, "Pure")
cal_csv = os.path.join(acf, "Ratio", "results", "calibration_spectra.csv")
d_bin = os.path.join(acf, "Ratio", "Binary")
d_tern = os.path.join(acf, "Ratio", "Tertiary")

from unmix import _templates, _baseline_removed, _l2
from real_data import load_map
from io_utils import load_calibration_csv
from dataset import is_blank
from calibration import calibrate
from sers_mixture import als_baseline
from validate import parse_mixture_label
from dl_quantify import (simulate_surface_solution, surface_composition,
                         fit_response_factors, apply_response_factors,
                         train_residual, predict_residual, _ratio)

LO, HI = 300.0, 1800.0
names, wn, means = _templates(pure, True, None)
m = (wn >= LO) & (wn <= HI)
nb = [names[i] for i in range(len(names)) if not is_blank(names[i])]
nonbg = [i for i in range(len(names)) if not is_blank(names[i])]
P = _l2(_baseline_removed(means[nonbg][:, m], True))
ax_c, names_c, dils = load_calibration_csv(cal_csv); mc = (ax_c >= LO) & (ax_c <= HI)
dil_al = [(dils[names_c.index(n)][0], np.asarray(dils[names_c.index(n)][1])[:, mc]) for n in nb]
calib = calibrate(dil_al, P, nb); K, gA = calib.K, calib.gA

def mix_spectrum(path):
    _w, c, _mn, _c = load_map(path); c = np.asarray(c, float)[:, m]
    w = c.sum(1); w = w / (w.sum() + 1e-12); mean = w @ c
    return np.clip(mean - als_baseline(mean), 0, None)

def dataset(folder):
    S, Y, L = [], [], []
    for p in sorted(glob.glob(os.path.join(folder, "*_corrected.csv"))):
        t = parse_mixture_label(os.path.basename(p), nb)
        if t and len(t) >= 2:
            S.append(surface_composition(mix_spectrum(p)[None, :], P)[0])
            Y.append(_ratio([t.get(n, 0) for n in nb]))
            L.append(os.path.basename(p).replace("_corrected.csv", ""))
    return np.array(S), np.array(Y), L

Sb, Yb, Lb = dataset(d_bin); St, Yt, Lt = dataset(d_tern)
rng = np.random.default_rng(0)
pt = simulate_surface_solution(P, K, gA, 5000, rng, noise=0.015, baseline=0.03,
                               gain_lo=0.9, gain_hi=1.1)

rows = []
def add(setname, label, true, nn, rf, dl):
    e = lambda a: 0.5 * float(np.abs(np.asarray(a) - true).sum())
    rows.append([setname, label]
                + [f"{v:.3f}" for v in true] + [f"{v:.3f}" for v in nn]
                + [f"{v:.3f}" for v in rf] + [f"{v:.3f}" for v in dl]
                + [f"{e(nn):.3f}", f"{e(rf):.3f}", f"{e(dl):.3f}"])

# binary: leave-one-map-out
for i in range(len(Sb)):
    tr = [j for j in range(len(Sb)) if j != i]
    r = fit_response_factors(Sb[tr], Yb[tr])
    mdl = train_residual(Sb[tr], Yb[tr], pretrain=pt, hidden=(32, 16), epochs=300, seed=i)
    add("binary_LOO", Lb[i], Yb[i], Sb[i],
        apply_response_factors(Sb[i][None, :], r)[0], predict_residual(mdl, Sb[i][None, :])[0])

# ternary: train on ALL binary, predict unseen ternary
r_all = fit_response_factors(Sb, Yb)
mdl_all = train_residual(Sb, Yb, pretrain=pt, hidden=(32, 16), epochs=400, seed=0)
for i in range(len(St)):
    add("ternary_generalization", Lt[i], Yt[i], St[i],
        apply_response_factors(St[i][None, :], r_all)[0], predict_residual(mdl_all, St[i][None, :])[0])

hdr = (["set", "mixture"]
       + [f"true_{n}" for n in nb] + [f"nnls_{n}" for n in nb]
       + [f"rf_{n}" for n in nb] + [f"dl_{n}" for n in nb]
       + ["err_nnls", "err_rf", "err_dl"])
out = os.path.join(os.getcwd(), "docs", "mixture_predictions.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
print("wrote", out, f"({len(rows)} mixtures)")
# summary
for s in ("binary_LOO", "ternary_generalization"):
    sub = [r for r in rows if r[0] == s]
    en = np.mean([float(r[-3]) for r in sub]); er = np.mean([float(r[-2]) for r in sub]); ed = np.mean([float(r[-1]) for r in sub])
    print(f"  {s:<24} n={len(sub):2d}  NNLS {en:.1%}  RF {er:.1%}  DL {ed:.1%}")
