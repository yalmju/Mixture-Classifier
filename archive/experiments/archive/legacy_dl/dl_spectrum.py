# -*- coding: utf-8 -*-
"""Strengthen the DL: give it the FULL SPECTRUM (like PLS) instead of the 3-dim surface
composition. Physics-pretrained on simulated (spectrum, composition), fine-tuned on real
binary, tested on unseen ternary. Compared head-to-head with PLS / RF / linear RF.
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
from dl_quantify import (simulate_mixtures, surface_composition,
                         fit_response_factors, apply_response_factors, _ratio)
from sklearn.cross_decomposition import PLSRegression

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
    X, S, Y = [], [], []
    for p in sorted(glob.glob(os.path.join(folder, "*_corrected.csv"))):
        t = parse_mixture_label(os.path.basename(p), nb)
        if t and len(t) >= 2:
            y = mix_spec(p); X.append(y); S.append(surface_composition(y[None, :], P)[0])
            Y.append(_ratio([t.get(n, 0) for n in nb]))
    return np.array(X), np.array(S), np.array(Y)

Xb, Sb, Yb = dataset(d_bin); Xt, St, Yt = dataset(d_tern)
nfeat = Xb.shape[1]

# physics-simulated (spectrum, composition) pretrain — L2-normalised spectra
rng = np.random.default_rng(0)
Xs, Cs = simulate_mixtures(P, K, gA, 8000, rng, noise=0.015, baseline=0.03, gain_lo=0.8, gain_hi=1.25)
Xs = np.array([r / (np.linalg.norm(r) + 1e-12) for r in Xs]).astype(np.float32)
Ys = np.array([_ratio(c) for c in Cs]).astype(np.float32)

def train_spec_dl(Xtr, Ytr, epochs_pre=25, epochs_ft=400, seed=0):
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(nfeat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.15),
                        nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 3))
    sm = nn.LogSoftmax(dim=1)
    def loss_fn(xb, yb): return (sm(net(xb)).exp() - yb).abs().sum(1).mean()
    # pretrain on synthetic (minibatch)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    dl = DataLoader(TensorDataset(torch.tensor(Xs), torch.tensor(Ys)), batch_size=256, shuffle=True)
    for _ in range(epochs_pre):
        net.train()
        for xb, yb in dl:
            opt.zero_grad(); loss_fn(xb, yb).backward(); opt.step()
    # fine-tune on real (full-batch)
    if len(Xtr):
        opt = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-3)
        Xt_ = torch.tensor(Xtr.astype(np.float32)); Yt_ = torch.tensor(Ytr.astype(np.float32))
        for _ in range(epochs_ft):
            net.train(); opt.zero_grad(); loss_fn(Xt_, Yt_).backward(); opt.step()
    net.eval()
    return net

def predict_spec(net, X):
    import torch
    with torch.no_grad():
        return torch.softmax(net(torch.tensor(np.atleast_2d(X).astype(np.float32))), dim=1).numpy()

def err(a, b): return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum()
def norm_rows(A): return np.array([_ratio(np.clip(a, 0, None)) for a in A])

# ---- ternary generalization ----
r = fit_response_factors(Sb, Yb)
pls = PLSRegression(n_components=3).fit(Xb, Yb)
net = train_spec_dl(Xb, Yb, seed=0)
res = {
    "NNLS": [err(St[i], Yt[i]) for i in range(len(Yt))],
    "linear RF": [err(apply_response_factors(St[i][None, :], r)[0], Yt[i]) for i in range(len(Yt))],
    "PLS (spectrum)": [err(p, Yt[i]) for i, p in enumerate(norm_rows(pls.predict(Xt)))],
    "DL surface (old)": None,
    "DL spectrum (new)": [err(p, Yt[i]) for i, p in enumerate(predict_spec(net, Xt))],
}
print("Binary -> unseen TERNARY (mean composition error):")
for k, v in res.items():
    if v is not None: print(f"  {k:<20} {np.mean(v):.1%}")

# per-mixture detail for the new DL vs PLS
print("\nper-ternary (true | PLS | DL-spectrum):")
Ppls = norm_rows(pls.predict(Xt)); Pdl = predict_spec(net, Xt)
f = lambda v: "/".join(f"{x:.2f}" for x in v)
labs = [os.path.basename(p).replace("_corrected.csv","") for p in sorted(glob.glob(os.path.join(d_tern,"*_corrected.csv"))) if len(parse_mixture_label(os.path.basename(p),nb) or {})>=2]
for i in range(len(Yt)):
    print(f"  {labs[i]:<16} {f(Yt[i])}   PLS {f(Ppls[i])}   DL {f(Pdl[i])}")
