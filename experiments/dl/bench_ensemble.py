# -*- coding: utf-8 -*-
"""Benchmark red-point fixes on 35-map LOO:
 baseline  = single full-spectrum DL
 (1) ensemble = mean of 5 bootstrap-bagged DL models
 (2) hybrid   = mean(PLS, ensemble-DL)
Metrics: overall composition err, buried non-THI err/detect, red-point count (err>0.35).
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
from dl_quantify import simulate_mixtures, surface_composition, _ratio
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
    w = c.sum(1); w = w/(w.sum()+1e-12); mean = w@c
    y = np.clip(mean-als_baseline(mean), 0, None); return y/(np.linalg.norm(y)+1e-12)

X, Y = [], []
for folder in folders:
    for p in sorted(glob.glob(os.path.join(folder, "*_corrected.csv"))):
        t = parse_mixture_label(os.path.basename(p), nb)
        if t and len(t) >= 2:
            X.append(mix_spec(p)); Y.append(_ratio([t.get(n,0) for n in nb]))
X=np.array(X); Y=np.array(Y); N=len(X); nfeat=X.shape[1]
print(f"{N} mixtures\n")

rng=np.random.default_rng(0)
Xs,Cs=simulate_mixtures(P,calib.K,calib.gA,8000,rng,noise=0.015,baseline=0.03,gain_lo=0.8,gain_hi=1.25)
Xs=np.array([r/(np.linalg.norm(r)+1e-12) for r in Xs]).astype(np.float32); Ys=np.array([_ratio(c) for c in Cs]).astype(np.float32)
def net_(): return nn.Sequential(nn.Linear(nfeat,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.15),nn.Linear(256,64),nn.ReLU(),nn.Linear(64,3))
sm=nn.LogSoftmax(dim=1); torch.manual_seed(0); base=net_()
o=torch.optim.Adam(base.parameters(),lr=1e-3,weight_decay=1e-4); Xs_t=torch.tensor(Xs); Ys_t=torch.tensor(Ys)
for ep in range(30):
    for i in range(0,len(Xs),256):
        o.zero_grad(); (sm(base(Xs_t[i:i+256])).exp()-Ys_t[i:i+256]).abs().sum(1).mean().backward(); o.step()
bs={k:v.clone() for k,v in base.state_dict().items()}

def fit_member(Xtr, Ytr, seed):
    torch.manual_seed(seed)
    idx = np.random.default_rng(seed).integers(0, len(Xtr), len(Xtr))   # bootstrap
    net=net_(); net.load_state_dict(bs); op=torch.optim.Adam(net.parameters(),lr=3e-4,weight_decay=1e-3)
    Xt_=torch.tensor(Xtr[idx].astype(np.float32)); Yt_=torch.tensor(Ytr[idx].astype(np.float32))
    for _ in range(220):
        net.train(); op.zero_grad(); (sm(net(Xt_)).exp()-Yt_).abs().sum(1).mean().backward(); op.step()
    net.eval(); return net
def pred(net, x):
    with torch.no_grad(): return torch.softmax(net(torch.tensor(x[None,:].astype(np.float32))),1).numpy()[0]
def norm_rows(A): return np.array([_ratio(np.clip(a,0,None)) for a in A])

P_single=np.zeros((N,3)); P_ens=np.zeros((N,3)); P_hyb=np.zeros((N,3))
for i in range(N):
    tr=[j for j in range(N) if j!=i]
    members=[fit_member(X[tr],Y[tr],s) for s in range(5)]
    P_single[i]=pred(members[0], X[i])
    P_ens[i]=np.mean([pred(mm, X[i]) for mm in members], axis=0)
    pls=norm_rows(PLSRegression(n_components=3).fit(X[tr],Y[tr]).predict(X[i][None,:]))[0]
    P_hyb[i]=_ratio(0.5*P_ens[i]+0.5*pls)

def report(name, Pm):
    ce=np.mean([0.5*np.abs(Pm[i]-Y[i]).sum() for i in range(N)])
    red=sum(0.5*np.abs(Pm[i]-Y[i]).sum()>0.35 for i in range(N))
    ti=np.where(Y[:,THI]>0)[0]
    be=np.mean([np.abs(Pm[i][nonTHI]-Y[i][nonTHI]).sum() for i in ti])
    det=np.mean([np.mean([(Pm[i][j]>0.05) for j in nonTHI if Y[i][j]>0.05] or [1.0]) for i in ti])
    print(f"  {name:<24} comp {ce:.1%}  red>0.35 {red}/{N}  buried_err {be:.2f}  detect {det:.0%}")

print("35-map LOO benchmark:")
report("baseline single DL", P_single)
report("(1) ensemble x5 (bagged)", P_ens)
report("(2) PLS + ensemble hybrid", P_hyb)
