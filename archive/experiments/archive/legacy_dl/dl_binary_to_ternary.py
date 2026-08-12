# -*- coding: utf-8 -*-
"""Generalization test: train correctors on the 28 BINARY mixtures, predict the 7
TERNARY mixtures (all three components present — never seen in binary training).
NNLS / linear response factor / DL residual (synthetic-pretrained + binary fine-tune).
"""
import os, sys, glob
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
from competitive import fit_B
from validate import parse_mixture_label
from dl_quantify import (simulate_surface_solution, surface_composition,
                         fit_response_factors, apply_response_factors,
                         train_residual, predict_residual, _ratio)

LO, HI = 300.0, 1800.0
names, wn, means = _templates(pure, True, None)
m = (wn >= LO) & (wn <= HI)
nonbg = [i for i in range(len(names)) if not is_blank(names[i])]
nb = [names[i] for i in nonbg]
P = _l2(_baseline_removed(means[nonbg][:, m], True))

ax_c, names_c, dils = load_calibration_csv(cal_csv)
mc = (ax_c >= LO) & (ax_c <= HI)
dil_al = [(dils[names_c.index(n)][0], np.asarray(dils[names_c.index(n)][1])[:, mc]) for n in nb]
calib = calibrate(dil_al, P, nb); K, gA = calib.K, calib.gA

def mix_spectrum(path):
    _wm, cube, _mn, _c = load_map(path)
    cube = np.asarray(cube, float)[:, m]
    w = cube.sum(1); w = w / (w.sum() + 1e-12)
    mean = w @ cube
    return np.clip(mean - als_baseline(mean), 0, None)

def dataset(folder):
    S, Y, L = [], [], []
    for p in sorted(glob.glob(os.path.join(folder, "*_corrected.csv"))):
        t = parse_mixture_label(os.path.basename(p), nb)
        if not t or len(t) < 2:
            continue
        S.append(surface_composition(mix_spectrum(p)[None, :], P)[0])
        Y.append(_ratio([t.get(n, 0) for n in nb]))
        L.append(os.path.basename(p).replace("_corrected.csv", ""))
    return np.array(S), np.array(Y), L

Sb, Yb, Lb = dataset(d_bin)                              # train (binary)
St, Yt, Lt = dataset(d_tern)                             # test  (ternary)
print(f"train binary {len(Sb)} · test ternary {len(St)} · compounds {nb}\n")

# correctors fit on binary only
r = fit_response_factors(Sb, Yb)
rng = np.random.default_rng(0)
pt_s, pt_y = simulate_surface_solution(P, K, gA, 5000, rng,
                                       noise=0.015, baseline=0.03, gain_lo=0.9, gain_hi=1.1)
model = train_residual(Sb, Yb, pretrain=(pt_s, pt_y), hidden=(32, 16), epochs=400, seed=0)

def err(a, b): return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum()
en, er, ed = [], [], []
f = lambda v: "/".join(f"{x:.2f}" for x in v)
print(f"{'ternary mix':<16}{'true':<20}{'NNLS':<20}{'RF':<20}{'DL':<20}")
for i in range(len(St)):
    nn = St[i]
    rf = apply_response_factors(St[i][None, :], r)[0]
    dl = predict_residual(model, St[i][None, :])[0]
    en.append(err(nn, Yt[i])); er.append(err(rf, Yt[i])); ed.append(err(dl, Yt[i]))
    print(f"{Lt[i]:<16}{f(Yt[i]):<20}{f(nn):<20}{f(rf):<20}{f(dl):<20}")
print(f"\nMEAN error on unseen ternary:  NNLS {np.mean(en):.1%}  |  linear RF {np.mean(er):.1%}"
      f"  |  DL {np.mean(ed):.1%}")
