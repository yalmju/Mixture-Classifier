# -*- coding: utf-8 -*-
"""Absolute concentration (µM) prediction — the harder task the composition work set aside.
True µM = filename code x 10 (1->10µM, 10->100µM, 100->1mM, 0.1->1µM, 0.01->0.1µM).

Compares, per compound (35-map leave-one-out):
  - physics inversion  (calibration competitive-Langmuir, calibration.quantify)
  - DL regression      (spectrum -> log10 µM, MLP, trained on the real mixtures)
Magnitude matters for absolute µM, so spectra are NOT L2-normalised (per-feature
standardised instead). Metrics: median recovery% ± SE, R² on log10 µM, within-2x and
within-order-of-magnitude hit rate. Honest about the gain-ambiguity ceiling.
"""
import os, sys, glob
sys.path.insert(0, os.getcwd())
import numpy as np
root = "S:/Google Drive"
acf = next(os.path.join(e.path, "ACF_PEST_DB") for e in os.scandir(root)
           if e.is_dir() and os.path.isdir(os.path.join(e.path, "ACF_PEST_DB")))
pure = os.path.join(acf, "Pure")
folders = [os.path.join(acf, "Ratio", "Binary"), os.path.join(acf, "Ratio", "Tertiary")]
from unmix import _templates, _baseline_removed, _l2
from real_data import load_map
from dataset import is_blank
from sers_mixture import als_baseline
from validate import parse_mixture_label
from calibration import calibrate, quantify
from io_utils import load_calibration_csv
import torch, torch.nn as nn

LO, HI = 300.0, 1800.0
names, wn, means = _templates(pure, True, None); m = (wn >= LO) & (wn <= HI)
nb = [names[i] for i in range(len(names)) if not is_blank(names[i])]
nonbg = [i for i in range(len(names)) if not is_blank(names[i])]
P = _l2(_baseline_removed(means[nonbg][:, m], True))
ax_c, names_c, dils = load_calibration_csv(os.path.join(acf, "Ratio", "results", "calibration_spectra.csv"))
mc = (ax_c >= LO) & (ax_c <= HI)
calib = calibrate([(dils[names_c.index(n)][0], np.asarray(dils[names_c.index(n)][1])[:, mc]) for n in nb], P, nb)

def mix_abs(p):                     # baseline-removed intensity-weighted mean, NOT normalised
    _w, c, _mn, _c = load_map(p); c = np.asarray(c, float)[:, m]
    w = c.sum(1); w = w/(w.sum()+1e-12); mean = w@c
    return np.clip(mean - als_baseline(mean), 0, None)

Xabs, C = [], []
for folder in folders:
    for p in sorted(glob.glob(os.path.join(folder, "*_corrected.csv"))):
        t = parse_mixture_label(os.path.basename(p), nb)
        if t and len(t) >= 2:
            Xabs.append(mix_abs(p)); C.append([t.get(n, 0.0)*1e-5 for n in nb])   # M
Xabs = np.array(Xabs, np.float64); C = np.array(C, np.float64); N = len(Xabs)
FLOOR = 1e-8
logC = np.log10(np.clip(C, FLOOR, None)).astype(np.float32)
print(f"{N} mixtures · compounds {nb}")

# ---- physics inversion µM (leave-one-out is irrelevant; it uses only calibration) ----
def phys_uM(i):
    present = [k for k in range(len(nb)) if C[i, k] > 0]
    q = quantify(Xabs[i], P, calib, present=present)
    return q["C"] * 1e6
Pphys = np.array([phys_uM(i) for i in range(N)])

# ---- DL regression spectrum -> log10 µM (per-feature standardised) ----
def net_(nf): return nn.Sequential(nn.Linear(nf,256), nn.BatchNorm1d(256), nn.ReLU(),
                                   nn.Dropout(0.15), nn.Linear(256,64), nn.ReLU(), nn.Linear(64,3))
def dl_loo(seed=0):
    Pm = np.zeros((N, 3))
    for i in range(N):
        tr = [j for j in range(N) if j != i]
        mu = Xabs[tr].mean(0); sd = Xabs[tr].std(0) + 1e-8
        Xtr = ((Xabs[tr]-mu)/sd).astype(np.float32); Xte = ((Xabs[i]-mu)/sd).astype(np.float32)
        Ytr = np.log10(np.clip(C[tr], FLOOR, None)).astype(np.float32) + 6.0   # log10 µM
        torch.manual_seed(seed); net = net_(Xtr.shape[1])
        op = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-3)
        Xt = torch.tensor(Xtr); Yt = torch.tensor(Ytr)
        for _ in range(400):
            net.train(); op.zero_grad(); ((net(Xt)-Yt)**2).mean().backward(); op.step()
        net.eval()
        with torch.no_grad(): Pm[i] = 10.0**net(torch.tensor(Xte[None,:])).numpy()[0]   # µM
    return Pm
Pdl = dl_loo()

def report(tag, PuM):
    print(f"\n{tag}:")
    for k, n in enumerate(nb):
        pres = np.where(C[:, k] > 0)[0]
        true = C[pres, k]*1e6; pred = np.clip(PuM[pres, k], 1e-6, None)
        rec = pred/true*100; lr = np.abs(np.log10(pred/true))
        med = np.median(rec); se = (np.percentile(rec,84)-np.percentile(rec,16))/2/np.sqrt(len(rec))
        yt = np.log10(true); yp = np.log10(pred)
        r2 = 1 - np.sum((yt-yp)**2)/max(np.sum((yt-yt.mean())**2), 1e-9)
        print(f"  {n}: recovery {med:.0f}% (±{se:.0f})  R²(log) {r2:+.2f}  "
              f"within-2x {np.mean(lr<np.log10(2)):.0%}  within-order {np.mean(lr<1):.0%}  (n={len(pres)})")

report("PHYSICS inversion (competitive Langmuir)", Pphys)
report("DL regression (spectrum -> log10 µM)", Pdl)
