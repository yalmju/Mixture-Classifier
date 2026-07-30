# -*- coding: utf-8 -*-
"""Validate the physics-informed DL quantifier (track A) on the REAL binary mixtures.

Trains on synthetic mixtures (from the real calibration's K/gA + pure templates), then
predicts composition on each real mixture map and compares the predicted CONCENTRATION
ratio to the true ratio in the filename. Raw NNLS surface ratio is the baseline.
"""
import os, sys, glob
sys.path.insert(0, os.getcwd())
import numpy as np

root = "S:/Google Drive"
acf = next(os.path.join(e.path, "ACF_PEST_DB") for e in os.scandir(root)
           if e.is_dir() and os.path.isdir(os.path.join(e.path, "ACF_PEST_DB")))
pure = os.path.join(acf, "Pure")
cal_csv = os.path.join(acf, "Ratio", "results", "calibration_spectra.csv")
binary = next((os.path.join(acf, "Ratio", d) for d in os.listdir(os.path.join(acf, "Ratio"))
               if d == "Binary"), None)

from unmix import _templates, _baseline_removed, _l2
from real_data import load_map
from io_utils import load_calibration_csv
from dataset import is_blank
from calibration import calibrate
from competitive import fit_B
from sers_mixture import als_baseline
from validate import parse_mixture_label
from dl_quantify import train_quantifier, predict_concentration

LO, HI = 300.0, 1800.0                                   # informative SERS window

names, wn, means = _templates(pure, True, None)
m = (wn >= LO) & (wn <= HI); wn_t = wn[m]
nonbg = [i for i in range(len(names)) if not is_blank(names[i])]
nb = [names[i] for i in nonbg]
P = _l2(_baseline_removed(means[nonbg][:, m], True))     # unit templates, trimmed
print("compounds:", nb, "| n_feat", len(wn_t))

# ---- calibration K/gA from the dilution spectra ----
ax_c, names_c, dils = load_calibration_csv(cal_csv)
mc = (ax_c >= LO) & (ax_c <= HI)
order = [names_c.index(n) for n in nb]                    # align to nb order
dil_al = [(dils[names_c.index(n)][0], np.asarray(dils[names_c.index(n)][1])[:, mc]) for n in nb]
calib = calibrate(dil_al, P, nb)
K, gA = calib.K, calib.gA
print("K :", np.array2string(K, precision=1))
print("gA:", np.array2string(gA, precision=1))

# ---- train DL on synthetic (gain fixed ~ calibration assumption) ----
model, hist = train_quantifier(P, K, gA, nb, n_train=16000, n_val=3000, epochs=45,
                               sim=dict(noise=0.015, baseline=0.03, gain_lo=0.9, gain_hi=1.1),
                               progress=lambda s: None)
print(f"DL trained — val log-MAE {hist['val_mae'][-1]:.3f} decades")

def mix_spectrum(path):
    """Intensity-weighted mean spectrum of a map, baseline-removed, trimmed."""
    _wm, cube, _mn, _c = load_map(path)
    cube = np.asarray(cube, float)[:, m]
    w = cube.sum(axis=1); w = w / (w.sum() + 1e-12)
    mean = w @ cube                                      # emphasise bright analyte pixels
    return np.clip(mean - als_baseline(mean), 0, None)

def ratio(v):
    v = np.clip(np.asarray(v, float), 0, None); s = v.sum()
    return v / s if s > 0 else v

maps = sorted(glob.glob(os.path.join(binary, "*_corrected.csv")))
rows, dl_err, nn_err = [], [], []
for p in maps:
    true = parse_mixture_label(os.path.basename(p), nb)
    if not true or len(true) < 2:
        continue
    t = ratio([true.get(n, 0) for n in nb])
    y = mix_spectrum(p)
    C = predict_concentration(model, y[None, :])[0]
    dl_r = ratio(C)
    B, _res = fit_B(y / (np.linalg.norm(y) + 1e-12), P)   # NNLS surface ratio (baseline)
    nn_r = ratio(B)
    dl_err.append(np.abs(dl_r - t).sum() / 2)
    nn_err.append(np.abs(nn_r - t).sum() / 2)
    rows.append((os.path.basename(p).replace("_corrected.csv", ""), t, nn_r, dl_r))

print(f"\n{len(rows)} mixtures validated   (error = ½·Σ|pred−true| composition L1)\n")
print(f"{'mixture':<14}{'true':<22}{'NNLS surf':<22}{'DL conc':<22}")
for nm, t, nn, dl in rows:
    f = lambda v: "/".join(f"{x:.2f}" for x in v)
    print(f"{nm:<14}{f(t):<22}{f(nn):<22}{f(dl):<22}")
print(f"\nMEAN composition error:  NNLS surface {np.mean(nn_err):.1%}   |   DL conc {np.mean(dl_err):.1%}")
