# -*- coding: utf-8 -*-
"""Track B validation — leave-one-out CV on the real binary mixtures.

Compares, on held-out real mixtures, three surface→solution correctors:
  1. NNLS (identity, no correction)          — the raw surface composition
  2. linear response factors (recovery panel) — fit on the training fold
  3. DL residual (synthetic pretrain + fine-tune on the training fold)
"""
import os, sys, glob
sys.path.insert(0, os.getcwd())
import numpy as np

root = "S:/Google Drive"
acf = next(os.path.join(e.path, "ACF_PEST_DB") for e in os.scandir(root)
           if e.is_dir() and os.path.isdir(os.path.join(e.path, "ACF_PEST_DB")))
pure = os.path.join(acf, "Pure")
cal_csv = os.path.join(acf, "Ratio", "results", "calibration_spectra.csv")
binary = os.path.join(acf, "Ratio", "Binary")

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

# ---- build the real (surface, true) dataset ----
maps = sorted(glob.glob(os.path.join(binary, "*_corrected.csv")))
surf, sol, labels = [], [], []
for p in maps:
    true = parse_mixture_label(os.path.basename(p), nb)
    if not true or len(true) < 2:
        continue
    y = mix_spectrum(p)
    surf.append(surface_composition(y[None, :], P)[0])
    sol.append(_ratio([true.get(n, 0) for n in nb]))
    labels.append(os.path.basename(p).replace("_corrected.csv", ""))
surf = np.array(surf); sol = np.array(sol)
N = len(surf)
print(f"{N} real mixtures | compounds {nb}\n")

# ---- physics-simulated pretrain pairs (fixed once) ----
rng = np.random.default_rng(0)
pt_surf, pt_sol = simulate_surface_solution(P, K, gA, 4000, rng,
                                            noise=0.015, baseline=0.03,
                                            gain_lo=0.9, gain_hi=1.1)

def comp_err(a, b):
    return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum()

e_nnls, e_rf, e_dl = [], [], []
for i in range(N):                                       # leave-one-out
    tr = [j for j in range(N) if j != i]
    Str, Ytr = surf[tr], sol[tr]
    # 1. NNLS identity
    e_nnls.append(comp_err(surf[i], sol[i]))
    # 2. linear response factors (fit on train fold)
    r = fit_response_factors(Str, Ytr)
    e_rf.append(comp_err(apply_response_factors(surf[i][None, :], r)[0], sol[i]))
    # 3. DL residual (synthetic pretrain + fine-tune on train fold)
    model = train_residual(Str, Ytr, pretrain=(pt_surf, pt_sol),
                           hidden=(16,), epochs=300, seed=i)
    e_dl.append(comp_err(predict_residual(model, surf[i][None, :])[0], sol[i]))
    print(f"  [{i+1:2d}/{N}] {labels[i]:<12} "
          f"NNLS {e_nnls[-1]:.2f}  RF {e_rf[-1]:.2f}  DL {e_dl[-1]:.2f}")

print(f"\nLOO-CV mean composition error (lower = better):")
print(f"  NNLS (no correction)      : {np.mean(e_nnls):.1%}")
print(f"  linear response factors   : {np.mean(e_rf):.1%}")
print(f"  DL residual (pretrain+ft) : {np.mean(e_dl):.1%}")
