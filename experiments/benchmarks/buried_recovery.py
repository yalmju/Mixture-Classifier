# -*- coding: utf-8 -*-
"""The real goal: recover the components BURIED under THI's dominant signal.
Metric focuses on the non-THI (buried) components. Classical NNLS collapses them to ~0
(reports near-pure THI); trained correctors (RF/PLS/DL) should un-bury them.
Evaluated on the 7 unseen ternary mixtures (trained on 28 binary).
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
from dl_quantify import (simulate_mixtures, simulate_surface_solution, surface_composition,
                         fit_response_factors, apply_response_factors,
                         train_residual, predict_residual, _ratio)
from sklearn.cross_decomposition import PLSRegression

LO, HI = 300.0, 1800.0
names, wn, means = _templates(pure, True, None); m = (wn >= LO) & (wn <= HI)
nb = [names[i] for i in range(len(names)) if not is_blank(names[i])]
THI = nb.index("THI"); nonTHI = [i for i in range(3) if i != THI]
nonbg = [i for i in range(len(names)) if not is_blank(names[i])]
P = _l2(_baseline_removed(means[nonbg][:, m], True))
ax_c, names_c, dils = load_calibration_csv(cal_csv); mc = (ax_c >= LO) & (ax_c <= HI)
dil_al = [(dils[names_c.index(n)][0], np.asarray(dils[names_c.index(n)][1])[:, mc]) for n in nb]
calib = calibrate(dil_al, P, nb); K, gA = calib.K, calib.gA

def mix_spec(p):
    _w, c, _mn, _c = load_map(p); c = np.asarray(c, float)[:, m]
    w = c.sum(1); w = w / (w.sum() + 1e-12); mean = w @ c
    y = np.clip(mean - als_baseline(mean), 0, None); return y / (np.linalg.norm(y) + 1e-12)

def dataset(folder):
    X, S, Y, L = [], [], [], []
    for p in sorted(glob.glob(os.path.join(folder, "*_corrected.csv"))):
        t = parse_mixture_label(os.path.basename(p), nb)
        if t and len(t) >= 2:
            y = mix_spec(p); X.append(y); S.append(surface_composition(y[None, :], P)[0])
            Y.append(_ratio([t.get(n, 0) for n in nb])); L.append(os.path.basename(p).replace("_corrected.csv",""))
    return np.array(X), np.array(S), np.array(Y), L

Xb, Sb, Yb, _ = dataset(d_bin); Xt, St, Yt, Lt = dataset(d_tern)
rng = np.random.default_rng(0)
pt = simulate_surface_solution(P, K, gA, 5000, rng, noise=0.015, baseline=0.03, gain_lo=0.9, gain_hi=1.1)

def norm_rows(A): return np.array([_ratio(np.clip(a, 0, None)) for a in A])
r = fit_response_factors(Sb, Yb)
pls = PLSRegression(n_components=3).fit(Xb, Yb)
mdl = train_residual(Sb, Yb, pretrain=pt, hidden=(32, 16), epochs=400, seed=0)

preds = {
    "NNLS":       St,
    "linear RF":  apply_response_factors(St, r),
    "PLS":        norm_rows(pls.predict(Xt)),
    "DL residual": np.array([predict_residual(mdl, St[i][None, :])[0] for i in range(len(St))]),
}

print("BURIED-COMPONENT RECOVERY on 7 unseen ternary  (all THI-dominated)\n")
print("per mixture: true[DQ/TBZ/THI] and each method's recovered NON-THI mass (DQ+TBZ)")
print(f"{'mixture':<16}{'true nonTHI':<13}" + "".join(f"{k:<13}" for k in preds))
for i in range(len(Lt)):
    tn = Yt[i][nonTHI].sum()
    row = f"{Lt[i]:<16}{tn:<13.2f}"
    for k, Pm in preds.items():
        row += f"{Pm[i][nonTHI].sum():<13.2f}"
    print(row)

print("\nsummary (mean over 7 ternary):")
print(f"  {'method':<14}{'nonTHI L1 err':<16}{'nonTHI recovered':<18}{'nonTHI detect rate'}")
for k, Pm in preds.items():
    err = np.mean([np.abs(Pm[i][nonTHI] - Yt[i][nonTHI]).sum() for i in range(len(Lt))])
    rec = np.mean([Pm[i][nonTHI].sum() for i in range(len(Lt))])
    # detection: fraction of truly-present non-THI components predicted > 0.05
    det = np.mean([np.mean([(Pm[i][j] > 0.05) for j in nonTHI if Yt[i][j] > 0.05] or [1.0])
                   for i in range(len(Lt))])
    print(f"  {k:<14}{err:<16.2f}{rec:<18.2f}{det:.0%}")
print(f"\n  (true mean non-THI mass = {np.mean([Yt[i][nonTHI].sum() for i in range(len(Lt))]):.2f})")
