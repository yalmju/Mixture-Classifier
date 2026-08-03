# -*- coding: utf-8 -*-
"""Strengthen the methods claim: compare physics baselines (NNLS, linear RF), classical
chemometrics (PLS, Random Forest — spectrum→composition), and the physics-informed DL,
on binary leave-one-map-out AND binary→ternary generalization.
"""
import os, sys, glob
sys.path.insert(0, os.getcwd())
import numpy as np

root = "S:/Google Drive"
acf = next(os.path.join(e.path, "ACF_PEST_DB") for e in os.scandir(root)
           if e.is_dir() and os.path.isdir(os.path.join(e.path, "ACF_PEST_DB")))
pure = os.path.join(acf, "Pure"); cal_csv = os.path.join(acf, "Ratio", "results", "calibration_spectra.csv")
d_bin = os.path.join(acf, "Ratio", "Binary"); d_tern = os.path.join(acf, "Ratio", "Tertiary")

from unmix import _templates, _baseline_removed, _l2
from real_data import load_map
from io_utils import load_calibration_csv
from dataset import is_blank
from calibration import calibrate
from sers_mixture import als_baseline
from validate import parse_mixture_label
from dl_quantify import (simulate_surface_solution, simulate_mixtures, surface_composition,
                         fit_response_factors, apply_response_factors,
                         train_residual, predict_residual, _ratio)
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor

LO, HI = 300.0, 1800.0
names, wn, means = _templates(pure, True, None); m = (wn >= LO) & (wn <= HI)
nb = [names[i] for i in range(len(names)) if not is_blank(names[i])]
nonbg = [i for i in range(len(names)) if not is_blank(names[i])]
P = _l2(_baseline_removed(means[nonbg][:, m], True))
ax_c, names_c, dils = load_calibration_csv(cal_csv); mc = (ax_c >= LO) & (ax_c <= HI)
dil_al = [(dils[names_c.index(n)][0], np.asarray(dils[names_c.index(n)][1])[:, mc]) for n in nb]
calib = calibrate(dil_al, P, nb); K, gA = calib.K, calib.gA

def mix_spec(p):
    _w, c, _mn, _c = load_map(p); c = np.asarray(c, float)[:, m]
    w = c.sum(1); w = w / (w.sum() + 1e-12); mean = w @ c
    y = np.clip(mean - als_baseline(mean), 0, None)
    return y / (np.linalg.norm(y) + 1e-12)

def dataset(folder):
    X, Sf, Y = [], [], []
    for p in sorted(glob.glob(os.path.join(folder, "*_corrected.csv"))):
        t = parse_mixture_label(os.path.basename(p), nb)
        if t and len(t) >= 2:
            y = mix_spec(p); X.append(y); Sf.append(surface_composition(y[None, :], P)[0])
            Y.append(_ratio([t.get(n, 0) for n in nb]))
    return np.array(X), np.array(Sf), np.array(Y)

Xb, Sb, Yb = dataset(d_bin); Xt, St, Yt = dataset(d_tern)
# physics-simulated pretrain (surface->solution) for the DL
rng = np.random.default_rng(0)
pt = simulate_surface_solution(P, K, gA, 5000, rng, noise=0.015, baseline=0.03, gain_lo=0.9, gain_hi=1.1)
# spectrum-level synthetic for PLS/RF pretrain option (optional; here train on real only)
def norm_rows(A): return np.array([_ratio(np.clip(a, 0, None)) for a in A])
def err(a, b): return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum()

def evaluate(train_idx_fn, Xtr, Str, Ytr, Xte, Ste, Yte):
    """Return dict method->list of per-sample errors on the test set."""
    out = {}
    out["NNLS"] = [err(Ste[i], Yte[i]) for i in range(len(Yte))]
    r = fit_response_factors(Str, Ytr)
    out["linear RF"] = [err(apply_response_factors(Ste[i][None, :], r)[0], Yte[i]) for i in range(len(Yte))]
    pls = PLSRegression(n_components=min(3, Xtr.shape[0] - 1)).fit(Xtr, Ytr)
    Ppls = norm_rows(pls.predict(Xte))
    out["PLS"] = [err(Ppls[i], Yte[i]) for i in range(len(Yte))]
    rf = RandomForestRegressor(n_estimators=300, random_state=0).fit(Xtr, Ytr)
    Prf = norm_rows(rf.predict(Xte))
    out["RandomForest"] = [err(Prf[i], Yte[i]) for i in range(len(Yte))]
    mdl = train_residual(Str, Ytr, pretrain=pt, hidden=(32, 16), epochs=400, seed=0)
    out["DL (ours)"] = [err(predict_residual(mdl, Ste[i][None, :])[0], Yte[i]) for i in range(len(Yte))]
    return out

# ---- binary leave-one-map-out ----
methods = ["NNLS", "linear RF", "PLS", "RandomForest", "DL (ours)"]
acc = {mth: [] for mth in methods}
for i in range(len(Xb)):
    tr = [j for j in range(len(Xb)) if j != i]
    res = evaluate(None, Xb[tr], Sb[tr], Yb[tr], Xb[i:i+1], Sb[i:i+1], Yb[i:i+1])
    for mth in methods: acc[mth] += res[mth]
print("Binary leave-one-map-out (mean composition error):")
for mth in methods: print(f"  {mth:<14} {np.mean(acc[mth]):.1%}")

# ---- binary -> ternary generalization ----
rest = evaluate(None, Xb, Sb, Yb, Xt, St, Yt)
print("\nBinary -> unseen TERNARY generalization:")
for mth in methods: print(f"  {mth:<14} {np.mean(rest[mth]):.1%}")
