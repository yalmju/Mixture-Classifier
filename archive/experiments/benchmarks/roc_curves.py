# -*- coding: utf-8 -*-
"""Detection ROC curves — presence/absence of each component (score = predicted fraction),
micro-averaged over the 3 components, 35-map leave-one-out (single seed). One curve per
method with its AUC; the paper-standard figure behind the AUC numbers. Saves docs/roc_curves.png."""
import os, sys, glob
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
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
                         apply_response_factors, train_composition, predict_composition, _ratio)
from calibration import calibrate
from io_utils import load_calibration_csv
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import torch, torch.nn as nn

LO, HI = 300.0, 1800.0
names, wn, means = _templates(pure, True, None); m = (wn >= LO) & (wn <= HI)
nb = [names[i] for i in range(len(names)) if not is_blank(names[i])]
nonbg = [i for i in range(len(names)) if not is_blank(names[i])]
P = _l2(_baseline_removed(means[nonbg][:, m], True))
ax_c, names_c, dils = load_calibration_csv(os.path.join(acf, "Ratio", "results", "calibration_spectra.csv"))
mc = (ax_c >= LO) & (ax_c <= HI)
calib = calibrate([(dils[names_c.index(n)][0], np.asarray(dils[names_c.index(n)][1])[:, mc]) for n in nb], P, nb)
def mix_spec(p):
    _w, c, _mn, _c = load_map(p); c = np.asarray(c, float)[:, m]
    w = c.sum(1); w = w/(w.sum()+1e-12); mean = w@c
    y = np.clip(mean-als_baseline(mean), 0, None); return y/(np.linalg.norm(y)+1e-12)
X, Y = [], []
for folder in folders:
    for p in sorted(glob.glob(os.path.join(folder, "*_corrected.csv"))):
        t = parse_mixture_label(os.path.basename(p), nb)
        if t and len(t) >= 2: X.append(mix_spec(p)); Y.append(_ratio([t.get(n,0) for n in nb]))
X = np.array(X, np.float32); Y = np.array(Y, np.float32); N, nfeat = X.shape
S = surface_composition(X, P)
rng = np.random.default_rng(0)
Xs, Cs = simulate_mixtures(P, calib.K, calib.gA, 8000, rng, noise=0.015, baseline=0.03, gain_lo=0.8, gain_hi=1.25)
PRE = (np.array([r/(np.linalg.norm(r)+1e-12) for r in Xs], np.float32), np.array([_ratio(c) for c in Cs], np.float32))
def norm_rows(A): return np.array([_ratio(np.clip(a,0,None)) for a in A])
class CNN(nn.Module):
    def __init__(s):
        super().__init__(); s.c=nn.Sequential(nn.Conv1d(1,16,7,padding=3),nn.BatchNorm1d(16),nn.ReLU(),nn.MaxPool1d(4),nn.Conv1d(16,32,5,padding=2),nn.BatchNorm1d(32),nn.ReLU(),nn.AdaptiveAvgPool1d(8)); s.f=nn.Sequential(nn.Flatten(),nn.Linear(256,64),nn.ReLU(),nn.Dropout(0.15),nn.Linear(64,3))
    def forward(s,x): return s.f(s.c(x.unsqueeze(1)))
_sm=nn.LogSoftmax(dim=1)
def _wl(net,xb,yb): p=_sm(net(xb)).exp(); w=1.0+2.0*(1.0-yb); return (w*(p-yb).abs()).sum(1).mean()
def torch_loo(build):
    Pm=np.zeros((N,3))
    for i in range(N):
        tr=[j for j in range(N) if j!=i]; torch.manual_seed(0); net=build()
        op=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4); Xp=torch.tensor(PRE[0]); Yp=torch.tensor(PRE[1])
        for _ in range(20):
            for b in range(0,len(Xp),256): op.zero_grad(); _wl(net,Xp[b:b+256],Yp[b:b+256]).backward(); op.step()
        op=torch.optim.Adam(net.parameters(),lr=3e-4,weight_decay=1e-3); Xt=torch.tensor(X[tr]); Yt=torch.tensor(Y[tr])
        for _ in range(220): net.train(); op.zero_grad(); _wl(net,Xt,Yt).backward(); op.step()
        net.eval()
        with torch.no_grad(): Pm[i]=torch.softmax(net(torch.tensor(X[i][None,:])),1).numpy()[0]
    return Pm
def sk_loo(mk):
    return np.array([norm_rows(mk().fit(X[[j for j in range(N) if j!=i]],Y[[j for j in range(N) if j!=i]]).predict(X[i][None,:]))[0] for i in range(N)])
def mlp_loo():
    return np.array([predict_composition(train_composition(X[[j for j in range(N) if j!=i]],Y[[j for j in range(N) if j!=i]],3,pretrain=PRE,seed=0),X[i])[0] for i in range(N)])
def rflin_loo():
    Pm=np.zeros((N,3))
    for i in range(N):
        tr=[j for j in range(N) if j!=i]; r=fit_response_factors(S[tr],Y[tr]); Pm[i]=apply_response_factors(S[i][None,:],r)[0]
    return Pm
PREDS = {"NNLS": S, "response factor": rflin_loo(), "PLS": sk_loo(lambda: PLSRegression(n_components=3)),
         "SVR": sk_loo(lambda: MultiOutputRegressor(SVR(C=10,gamma="scale"))),
         "RandomForest": sk_loo(lambda: RandomForestRegressor(n_estimators=300,random_state=0)),
         "1D-CNN": torch_loo(CNN), "MLP-DL (ours)": mlp_loo()}
lab = np.concatenate([(Y[:,c]>0.05).astype(int) for c in range(3)])
COL = {"NNLS":"#9aa3ad","response factor":"#c98a15","PLS":"#1a73e8","SVR":"#6b5fd6","RandomForest":"#0f9d6b","1D-CNN":"#d64545","MLP-DL (ours)":"#4a9e2a"}
plt.figure(figsize=(6.4,6.0))
import csv as _csv
_rows=[]
for name, Pm in PREDS.items():
    sc = np.concatenate([Pm[:,c] for c in range(3)])
    fpr,tpr,_ = roc_curve(lab, sc); a=auc(fpr,tpr)
    lw = 2.6 if "ours" in name else 1.4
    plt.plot(fpr,tpr,color=COL[name],lw=lw,label=f"{name}  (AUC {a:.3f})")
    for f,t in zip(fpr,tpr): _rows.append([name,f"{a:.4f}",f"{f:.4f}",f"{t:.4f}"])
with open(os.path.join(os.getcwd(),"docs","roc_curves.csv"),"w",newline="",encoding="utf-8") as _f:
    _w=_csv.writer(_f); _w.writerow(["method","AUC","fpr","tpr"]); _w.writerows(_rows)
plt.plot([0,1],[0,1],ls="--",color="#c7ccd2",lw=1)
plt.xlabel("false positive rate"); plt.ylabel("true positive rate")
plt.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
for s in ("top","right"): plt.gca().spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(os.getcwd(),"docs","roc_curves.png"), dpi=120, facecolor="white")
print("saved docs/roc_curves.png + docs/roc_curves.csv")
