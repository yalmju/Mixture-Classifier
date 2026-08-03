# -*- coding: utf-8 -*-
"""Binary 'hard' cases: recover the WEAKER adsorber (DQ<TBZ<THI) buried by the stronger one.
Grouped by competition pair. NNLS vs PLS vs DL, binary leave-one-map-out.
"""
import os, sys, glob
sys.path.insert(0, os.getcwd())
import numpy as np

root = "S:/Google Drive"
acf = next(os.path.join(e.path, "ACF_PEST_DB") for e in os.scandir(root)
           if e.is_dir() and os.path.isdir(os.path.join(e.path, "ACF_PEST_DB")))
pure = os.path.join(acf, "Pure"); d_bin = os.path.join(acf, "Ratio", "Binary")
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
RANK = {"DQ": 0, "TBZ": 1, "THI": 2}                      # adsorption affinity order (weak→strong)
names, wn, means = _templates(pure, True, None); m = (wn >= LO) & (wn <= HI)
nb = [names[i] for i in range(len(names)) if not is_blank(names[i])]
nonbg = [i for i in range(len(names)) if not is_blank(names[i])]
P = _l2(_baseline_removed(means[nonbg][:, m], True))
cal_csv = os.path.join(acf, "Ratio", "results", "calibration_spectra.csv")
ax_c, names_c, dils = load_calibration_csv(cal_csv); mc = (ax_c >= LO) & (ax_c <= HI)
calib = calibrate([(dils[names_c.index(n)][0], np.asarray(dils[names_c.index(n)][1])[:, mc]) for n in nb], P, nb)

def mix_spec(p):
    _w, c, _mn, _c = load_map(p); c = np.asarray(c, float)[:, m]
    w = c.sum(1); w = w/(w.sum()+1e-12); mean = w@c
    y = np.clip(mean-als_baseline(mean), 0, None); return y/(np.linalg.norm(y)+1e-12)

X, S, Y, meta = [], [], [], []
for p in sorted(glob.glob(os.path.join(d_bin, "*_corrected.csv"))):
    t = parse_mixture_label(os.path.basename(p), nb)
    if t and len(t) >= 2:
        y = mix_spec(p); X.append(y); S.append(surface_composition(y[None,:],P)[0])
        Y.append(_ratio([t.get(n,0) for n in nb]))
        present = [n for n in nb if t.get(n,0)>0]
        weak = min(present, key=lambda n: RANK[n]); strong = max(present, key=lambda n: RANK[n])
        meta.append((weak, strong, nb.index(weak)))
X=np.array(X); S=np.array(S); Y=np.array(Y); N=len(X); nfeat=X.shape[1]

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
def dl_pred(tr,i):
    nn_=net_(); nn_.load_state_dict(bs); op=torch.optim.Adam(nn_.parameters(),lr=3e-4,weight_decay=1e-3)
    Xt_=torch.tensor(X[tr].astype(np.float32)); Yt_=torch.tensor(Y[tr].astype(np.float32))
    for _ in range(250):
        nn_.train(); op.zero_grad(); (sm(nn_(Xt_)).exp()-Yt_).abs().sum(1).mean().backward(); op.step()
    nn_.eval()
    with torch.no_grad(): return torch.softmax(nn_(torch.tensor(X[i][None,:].astype(np.float32))),1).numpy()[0]

def norm_rows(A): return np.array([_ratio(np.clip(a,0,None)) for a in A])
Pn=S.copy(); Pp=np.zeros((N,3)); Pd=np.zeros((N,3))
for i in range(N):
    tr=[j for j in range(N) if j!=i]
    Pp[i]=norm_rows(PLSRegression(n_components=3).fit(X[tr],Y[tr]).predict(X[i][None,:]))[0]
    Pd[i]=dl_pred(tr,i)

print("Recovery of the BURIED (weaker) adsorber in binary mixtures, by competition pair")
print("(error = |pred − true| on the weak component; lower better)\n")
pairs = {}
for i,(w,s,wi) in enumerate(meta):
    key=f"{w} under {s}"; pairs.setdefault(key,[]).append(i)
print(f"  {'pair':<16}{'n':<4}{'true_weak':<11}{'NNLS':<9}{'PLS':<9}{'DL'}")
for key,idx in sorted(pairs.items(), key=lambda kv: -len(kv[1])):
    tw=np.mean([Y[i][meta[i][2]] for i in idx])
    en=np.mean([abs(Pn[i][meta[i][2]]-Y[i][meta[i][2]]) for i in idx])
    ep=np.mean([abs(Pp[i][meta[i][2]]-Y[i][meta[i][2]]) for i in idx])
    ed=np.mean([abs(Pd[i][meta[i][2]]-Y[i][meta[i][2]]) for i in idx])
    print(f"  {key:<16}{len(idx):<4}{tw:<11.2f}{en:<9.2f}{ep:<9.2f}{ed:.2f}")
# overall weak-component recovery
allw=lambda Pm: np.mean([abs(Pm[i][meta[i][2]]-Y[i][meta[i][2]]) for i in range(N)])
print(f"\n  {'ALL binary':<16}{N:<4}{'':<11}{allw(Pn):<9.2f}{allw(Pp):<9.2f}{allw(Pd):.2f}")
