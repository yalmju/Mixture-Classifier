# -*- coding: utf-8 -*-
"""Practical evaluation: leave-one-map-out CV over ALL 35 mixtures (28 binary + 7 ternary),
so the model also learns ternary competition. Full-spectrum methods (PLS, full-spectrum DL)
vs NNLS / linear RF. Metrics: overall composition + buried non-THI recovery on THI folds.
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
from dl_quantify import (simulate_mixtures, surface_composition, fit_response_factors,
                         apply_response_factors, _ratio)
from calibration import calibrate
from io_utils import load_calibration_csv
from sklearn.cross_decomposition import PLSRegression
import torch, torch.nn as nn

LO, HI = 300.0, 1800.0
names, wn, means = _templates(pure, True, None); m = (wn >= LO) & (wn <= HI)
nb = [names[i] for i in range(len(names)) if not is_blank(names[i])]
THI = nb.index("THI"); nonTHI = [i for i in range(3) if i != THI]
nonbg = [i for i in range(len(names)) if not is_blank(names[i])]
P = _l2(_baseline_removed(means[nonbg][:, m], True))
cal_csv = os.path.join(acf, "Ratio", "results", "calibration_spectra.csv")
ax_c, names_c, dils = load_calibration_csv(cal_csv); mc = (ax_c >= LO) & (ax_c <= HI)
calib = calibrate([(dils[names_c.index(n)][0], np.asarray(dils[names_c.index(n)][1])[:, mc]) for n in nb], P, nb)

def mix_spec(p):
    _w, c, _mn, _c = load_map(p); c = np.asarray(c, float)[:, m]
    w = c.sum(1); w = w / (w.sum() + 1e-12); mean = w @ c
    y = np.clip(mean - als_baseline(mean), 0, None); return y / (np.linalg.norm(y) + 1e-12)

X, S, Y, hasTHI = [], [], [], []
for folder in folders:
    for p in sorted(glob.glob(os.path.join(folder, "*_corrected.csv"))):
        t = parse_mixture_label(os.path.basename(p), nb)
        if t and len(t) >= 2:
            y = mix_spec(p); X.append(y); S.append(surface_composition(y[None, :], P)[0])
            Y.append(_ratio([t.get(n, 0) for n in nb])); hasTHI.append(t.get("THI", 0) > 0)
X = np.array(X); S = np.array(S); Y = np.array(Y); hasTHI = np.array(hasTHI); N = len(X); nfeat = X.shape[1]
print(f"{N} mixtures ({hasTHI.sum()} contain THI) · n_feat {nfeat}\n")

# pretrain a full-spectrum net ONCE on physics-simulated (spectrum -> composition)
rng = np.random.default_rng(0)
Xs, Cs = simulate_mixtures(P, calib.K, calib.gA, 8000, rng, noise=0.015, baseline=0.03, gain_lo=0.8, gain_hi=1.25)
Xs = np.array([r / (np.linalg.norm(r) + 1e-12) for r in Xs]).astype(np.float32)
Ys = np.array([_ratio(c) for c in Cs]).astype(np.float32)
def make_net():
    return nn.Sequential(nn.Linear(nfeat,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.15),
                         nn.Linear(256,64), nn.ReLU(), nn.Linear(64,3))
sm = nn.LogSoftmax(dim=1)
torch.manual_seed(0); base = make_net()
opt = torch.optim.Adam(base.parameters(), lr=1e-3, weight_decay=1e-4)
Xs_t = torch.tensor(Xs); Ys_t = torch.tensor(Ys)
for ep in range(30):
    for i in range(0, len(Xs), 256):
        xb, yb = Xs_t[i:i+256], Ys_t[i:i+256]
        opt.zero_grad(); (sm(base(xb)).exp()-yb).abs().sum(1).mean().backward(); opt.step()
base_state = {k: v.clone() for k, v in base.state_dict().items()}

def dl_predict(tr, xte):
    net = make_net(); net.load_state_dict(base_state)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-3)
    Xt_ = torch.tensor(X[tr].astype(np.float32)); Yt_ = torch.tensor(Y[tr].astype(np.float32))
    for _ in range(250):
        net.train(); opt.zero_grad(); (sm(net(Xt_)).exp()-Yt_).abs().sum(1).mean().backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return torch.softmax(net(torch.tensor(X[xte][None,:].astype(np.float32))), 1).numpy()[0]

def norm_rows(A): return np.array([_ratio(np.clip(a,0,None)) for a in A])
methods = ["NNLS","linear RF","PLS","DL full-spec"]
P_all = {mth: np.zeros((N,3)) for mth in methods}
for i in range(N):
    tr = [j for j in range(N) if j != i]
    P_all["NNLS"][i] = S[i]
    r = fit_response_factors(S[tr], Y[tr]); P_all["linear RF"][i] = apply_response_factors(S[i][None,:], r)[0]
    pls = PLSRegression(n_components=3).fit(X[tr], Y[tr]); P_all["PLS"][i] = norm_rows(pls.predict(X[i][None,:]))[0]
    P_all["DL full-spec"][i] = dl_predict(tr, i)

print("LOO-CV over all 35 (overall + buried recovery on THI-containing folds):")
print(f"  {'method':<15}{'comp err':<12}{'buried nonTHI err':<20}{'buried detect'}")
for mth in methods:
    Pm = P_all[mth]
    ce = np.mean([0.5*np.abs(Pm[i]-Y[i]).sum() for i in range(N)])
    ti = np.where(hasTHI)[0]
    be = np.mean([np.abs(Pm[i][nonTHI]-Y[i][nonTHI]).sum() for i in ti])
    det = np.mean([np.mean([(Pm[i][j]>0.05) for j in nonTHI if Y[i][j]>0.05] or [1.0]) for i in ti])
    print(f"  {mth:<15}{ce:<12.1%}{be:<20.2f}{det:.0%}")
