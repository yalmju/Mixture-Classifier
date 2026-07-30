# -*- coding: utf-8 -*-
"""Evidence benchmark: all methods on 35-map leave-one-out, with proper metrics and
multi-seed error bars — the justification for choosing the MLP DL over alternatives.

Methods: NNLS (no correction), linear response factor, PLS, Random Forest, SVR,
MLP-DL (ours, minor-weighted loss + physics pretrain), 1D-CNN.
Metrics: composition error (mean±SD over seeds), MAE, pooled R², buried non-THI error,
buried detection. Saves docs/model_benchmark.png (composition error ± SD).
"""
import os, sys, glob
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
SEEDS = [0, 1, 2]
names, wn, means = _templates(pure, True, None); m = (wn >= LO) & (wn <= HI)
nb = [names[i] for i in range(len(names)) if not is_blank(names[i])]
THI = nb.index("THI"); nonTHI = [i for i in range(3) if i != THI]
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
        if t and len(t) >= 2:
            X.append(mix_spec(p)); Y.append(_ratio([t.get(n,0) for n in nb]))
X = np.array(X, np.float32); Y = np.array(Y, np.float32); N = len(X); nfeat = X.shape[1]
S = surface_composition(X, P)                         # NNLS surface composition
rng = np.random.default_rng(0)
Xs, Cs = simulate_mixtures(P, calib.K, calib.gA, 8000, rng, noise=0.015, baseline=0.03, gain_lo=0.8, gain_hi=1.25)
PRE = (np.array([r/(np.linalg.norm(r)+1e-12) for r in Xs], np.float32), np.array([_ratio(c) for c in Cs], np.float32))
def norm_rows(A): return np.array([_ratio(np.clip(a,0,None)) for a in A])
print(f"{N} mixtures · n_feat {nfeat} · seeds {SEEDS}")

# --- CNN ---
class CNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.c = nn.Sequential(nn.Conv1d(1,16,7,padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
                            nn.Conv1d(16,32,5,padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.AdaptiveAvgPool1d(8))
        s.f = nn.Sequential(nn.Flatten(), nn.Linear(32*8,64), nn.ReLU(), nn.Dropout(0.15), nn.Linear(64,3))
    def forward(s, x): return s.f(s.c(x.unsqueeze(1)))
_sm = nn.LogSoftmax(dim=1)
def _wl(net, xb, yb): p=_sm(net(xb)).exp(); w=1.0+2.0*(1.0-yb); return (w*(p-yb).abs()).sum(1).mean()
def cnn_loo(seed):
    Pm=np.zeros((N,3))
    for i in range(N):
        tr=[j for j in range(N) if j!=i]; torch.manual_seed(seed); net=CNN()
        op=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4)
        Xp=torch.tensor(PRE[0]); Yp=torch.tensor(PRE[1])
        for _ in range(20):
            for b in range(0,len(Xp),256): op.zero_grad(); _wl(net,Xp[b:b+256],Yp[b:b+256]).backward(); op.step()
        op=torch.optim.Adam(net.parameters(),lr=3e-4,weight_decay=1e-3)
        Xt=torch.tensor(X[tr]); Yt=torch.tensor(Y[tr])
        for _ in range(220): net.train(); op.zero_grad(); _wl(net,Xt,Yt).backward(); op.step()
        net.eval()
        with torch.no_grad(): Pm[i]=torch.softmax(net(torch.tensor(X[i][None,:])),1).numpy()[0]
    return Pm
def mlp_loo(seed):
    Pm=np.zeros((N,3))
    for i in range(N):
        tr=[j for j in range(N) if j!=i]
        mdl=train_composition(X[tr],Y[tr],3,pretrain=PRE,seed=seed)
        Pm[i]=predict_composition(mdl,X[i])[0]
    return Pm
def rf_loo(seed):
    Pm=np.zeros((N,3))
    for i in range(N):
        tr=[j for j in range(N) if j!=i]
        Pm[i]=norm_rows(RandomForestRegressor(n_estimators=300,random_state=seed).fit(X[tr],Y[tr]).predict(X[i][None,:]))[0]
    return Pm
def pls_loo(_seed=0):
    return np.array([norm_rows(PLSRegression(n_components=3).fit(X[[j for j in range(N) if j!=i]],Y[[j for j in range(N) if j!=i]]).predict(X[i][None,:]))[0] for i in range(N)])
def svr_loo(_seed=0):
    Pm=np.zeros((N,3))
    for i in range(N):
        tr=[j for j in range(N) if j!=i]
        Pm[i]=norm_rows(MultiOutputRegressor(SVR(C=10,gamma="scale")).fit(X[tr],Y[tr]).predict(X[i][None,:]))[0]
    return Pm
def rf_lin_loo(_seed=0):
    Pm=np.zeros((N,3))
    for i in range(N):
        tr=[j for j in range(N) if j!=i]; r=fit_response_factors(S[tr],Y[tr])
        Pm[i]=apply_response_factors(S[i][None,:],r)[0]
    return Pm

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
def metrics(Pm):
    ce=np.mean([0.5*np.abs(Pm[i]-Y[i]).sum() for i in range(N)])
    mae=float(np.abs(Pm-Y).mean())
    rmse=float(np.sqrt(((Pm-Y)**2).mean()))
    ss=float(((Y-Y.mean(0))**2).sum()); r2=1-float(((Y-Pm)**2).sum())/ss
    # detection = binary "component present (>5%)" per component → ROC-AUC / PR-AUC / F1
    aucs, aps, f1s = [], [], []
    for c in range(3):
        lab=(Y[:,c]>0.05).astype(int)
        if 0 < lab.sum() < len(lab):
            aucs.append(roc_auc_score(lab, Pm[:,c]))
            aps.append(average_precision_score(lab, Pm[:,c]))
            f1s.append(f1_score(lab, (Pm[:,c]>0.05).astype(int), zero_division=0))
    auc=float(np.mean(aucs)); ap=float(np.mean(aps)); f1=float(np.mean(f1s))
    ti=np.where(Y[:,THI]>0)[0]
    be=np.mean([np.abs(Pm[i][nonTHI]-Y[i][nonTHI]).sum() for i in ti])
    return dict(ce=ce, mae=mae, rmse=rmse, r2=r2, auc=auc, ap=ap, f1=f1, be=be)

METHODS = [("NNLS", lambda s: S, False), ("response factor", rf_lin_loo, False),
           ("PLS", pls_loo, False), ("SVR", svr_loo, False),
           ("RandomForest", rf_loo, True), ("1D-CNN", cnn_loo, True),
           ("MLP-DL (ours)", mlp_loo, True)]
rows = {}
for name, fn, stoch in METHODS:
    seeds = SEEDS if stoch else [0]
    ms = [metrics(fn(s)) for s in seeds]
    agg = {k: (np.mean([mm[k] for mm in ms]), np.std([mm[k] for mm in ms])) for k in ms[0]}
    rows[name] = agg
    print(f"  {name:<16} comp {agg['ce'][0]:.1%}±{agg['ce'][1]:.1%}  RMSE {agg['rmse'][0]:.3f}"
          f"  R2 {agg['r2'][0]:.2f}  AUC {agg['auc'][0]:.3f}  PR-AUC {agg['ap'][0]:.2f}"
          f"  F1 {agg['f1'][0]:.2f}  buried {agg['be'][0]:.2f}")

# CSV table (all metrics, mean±SD)
import csv
order=[k for k,_,_ in METHODS]
with open(os.path.join(os.getcwd(),"docs","model_benchmark.csv"),"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["method","comp_err","comp_err_SD","RMSE","R2","ROC_AUC","PR_AUC","F1","buried_err"])
    for k in order:
        a=rows[k]; w.writerow([k, f"{a['ce'][0]:.4f}", f"{a['ce'][1]:.4f}", f"{a['rmse'][0]:.4f}",
                               f"{a['r2'][0]:.4f}", f"{a['auc'][0]:.4f}", f"{a['ap'][0]:.4f}",
                               f"{a['f1'][0]:.4f}", f"{a['be'][0]:.4f}"])

# figure: composition error (lower better) + detection ROC-AUC (higher better), ±SD
cols=["#9aa3ad","#c98a15","#1a73e8","#6b5fd6","#0f9d6b","#d64545","#4a9e2a"]
fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4.8))
ce=[rows[k]["ce"][0]*100 for k in order]; ces=[rows[k]["ce"][1]*100 for k in order]
b=a1.bar(range(len(order)), ce, yerr=ces, capsize=4, color=cols, edgecolor="white")
a1.set_ylabel("composition error (%)")
au=[rows[k]["auc"][0] for k in order]; aus=[rows[k]["auc"][1] for k in order]
b=a2.bar(range(len(order)), au, yerr=aus, capsize=4, color=cols, edgecolor="white")
a2.set_ylim(0.5,1.02); a2.set_ylabel("detection ROC-AUC")
for ax in (a1,a2):
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=20, ha="right", fontsize=8.5)
    for s in ("top","right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(os.getcwd(),"docs","model_benchmark.png"), dpi=120, facecolor="white")
print("saved docs/model_benchmark.png + docs/model_benchmark.csv")
