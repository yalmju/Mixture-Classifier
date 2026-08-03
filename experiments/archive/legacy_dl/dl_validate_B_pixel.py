# -*- coding: utf-8 -*-
"""Track B, PER-PIXEL — leave-one-MAP-out CV (no pixel leakage between train/test).

Each map's ~hundreds of pixels share the map's true ratio → ~thousands of labelled
samples instead of 28 means. Tests whether that lets the DL residual beat linear RF.
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
from sers_mixture import als_baseline
from competitive import fit_B
from validate import parse_mixture_label
from dl_quantify import (fit_response_factors, apply_response_factors,
                         train_residual, predict_residual, _ratio)

LO, HI, KPX = 300.0, 1800.0, 120                          # window, pixels/map cap
names, wn, means = _templates(pure, True, None)
m = (wn >= LO) & (wn <= HI)
nonbg = [i for i in range(len(names)) if not is_blank(names[i])]
nb = [names[i] for i in nonbg]
P = _l2(_baseline_removed(means[nonbg][:, m], True))

def pixel_surfs(path, rng):
    """Top-KPX hit pixels → per-pixel baseline-removed surface composition (k, n_comp)."""
    _wm, cube, _mn, _c = load_map(path)
    cube = np.asarray(cube, float)[:, m]
    tot = cube.sum(1)
    idx = np.argsort(tot)[::-1][:KPX]                     # brightest = analyte, not blank
    out = []
    for j in idx:
        y = np.clip(cube[j] - als_baseline(cube[j]), 0, None)
        B, _ = fit_B(y / (np.linalg.norm(y) + 1e-12), P)
        out.append(_ratio(B))
    return np.array(out)

# ---- build per-pixel dataset (surface, true, map id) ----
rng = np.random.default_rng(0)
maps = sorted(glob.glob(os.path.join(binary, "*_corrected.csv")))
S, Y, G, labels = [], [], [], []
for gi, p in enumerate(maps):
    true = parse_mixture_label(os.path.basename(p), nb)
    if not true or len(true) < 2:
        continue
    t = _ratio([true.get(n, 0) for n in nb])
    ps = pixel_surfs(p, rng)
    S.append(ps); Y.append(np.tile(t, (len(ps), 1)))
    G.append(np.full(len(ps), len(labels))); labels.append((os.path.basename(p).replace("_corrected.csv", ""), t))
S = np.vstack(S); Y = np.vstack(Y); G = np.concatenate(G)
NM = len(labels)
print(f"{NM} maps · {len(S)} pixels total ({len(S)//NM}/map avg) · compounds {nb}\n")

def comp_err(a, b):
    return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum()

e_nnls, e_rf, e_dl = [], [], []
for gi in range(NM):                                      # leave-one-MAP-out
    te = G == gi; tr = ~te
    Ste, true = S[te], labels[gi][1]
    # NNLS identity: mean surface composition of the held-out map
    e_nnls.append(comp_err(Ste.mean(0), true))
    # linear response factors fit on train pixels
    r = fit_response_factors(S[tr], Y[tr])
    e_rf.append(comp_err(apply_response_factors(Ste, r).mean(0), true))
    # DL residual trained on train pixels (no synthetic pretrain — real pixels suffice)
    model = train_residual(S[tr], Y[tr], pretrain=None, hidden=(32, 16), epochs=120, seed=gi)
    e_dl.append(comp_err(predict_residual(model, Ste).mean(0), true))
    print(f"  [{gi+1:2d}/{NM}] {labels[gi][0]:<12} NNLS {e_nnls[-1]:.2f}  RF {e_rf[-1]:.2f}  DL {e_dl[-1]:.2f}")

print(f"\nLOO-MAP-out mean composition error (per-pixel training):")
print(f"  NNLS (no correction)    : {np.mean(e_nnls):.1%}")
print(f"  linear response factors : {np.mean(e_rf):.1%}")
print(f"  DL residual (per-pixel) : {np.mean(e_dl):.1%}")
