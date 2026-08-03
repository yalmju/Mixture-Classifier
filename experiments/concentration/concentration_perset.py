# -*- coding: utf-8 -*-
"""Is absolute µM a gain-consistency problem? Run the µM benchmark PER dataset (each is a
more consistent measurement set) vs combined. If within-set is much better than combined,
absolute µM is recoverable once gain is consistent (which the reproducible ink provides)."""
import os, sys, glob
sys.path.insert(0, os.getcwd())
import numpy as np
root = "S:/Google Drive"
acf = next(os.path.join(e.path, "ACF_PEST_DB") for e in os.scandir(root)
           if e.is_dir() and os.path.isdir(os.path.join(e.path, "ACF_PEST_DB")))
pure = os.path.join(acf, "Pure")
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
def mix_abs(p):
    _w, c, _mn, _c = load_map(p); c = np.asarray(c, float)[:, m]
    w = c.sum(1); w = w/(w.sum()+1e-12); mean = w@c
    return np.clip(mean - als_baseline(mean), 0, None)
def load_set(folder, fac):
    Xa, C = [], []
    for p in sorted(glob.glob(os.path.join(folder, "*orrected.csv"))):
        t = parse_mixture_label(os.path.basename(p), nb)
        if t and len(t) >= 2:
            Xa.append(mix_abs(p)); C.append([t.get(n, 0.0)*fac for n in nb])
    return np.array(Xa, np.float64), np.array(C, np.float64)
FLOOR=1e-8
def net_(nf): return nn.Sequential(nn.Linear(nf,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.15),nn.Linear(256,64),nn.ReLU(),nn.Linear(64,3))
def dl_loo(Xabs,C):
    N=len(Xabs); Pm=np.zeros((N,3))
    for i in range(N):
        tr=[j for j in range(N) if j!=i]; mu=Xabs[tr].mean(0); sd=Xabs[tr].std(0)+1e-8
        Xt=((Xabs[tr]-mu)/sd).astype(np.float32); Xe=((Xabs[i]-mu)/sd).astype(np.float32)
        Yt=(np.log10(np.clip(C[tr],FLOOR,None))+6).astype(np.float32)
        torch.manual_seed(0); net=net_(Xt.shape[1]); op=torch.optim.Adam(net.parameters(),lr=3e-4,weight_decay=1e-3)
        Xtt=torch.tensor(Xt); Ytt=torch.tensor(Yt)
        for _ in range(400): net.train(); op.zero_grad(); ((net(Xtt)-Ytt)**2).mean().backward(); op.step()
        net.eval()
        with torch.no_grad(): Pm[i]=10.0**net(torch.tensor(Xe[None,:])).numpy()[0]
    return Pm
def phys(Xabs,C):
    N=len(Xabs); Pm=np.zeros((N,3))
    for i in range(N):
        pres=[k for k in range(3) if C[i,k]>0]; Pm[i]=quantify(Xabs[i],P,calib,present=pres)["C"]*1e6
    return Pm
def summ(tag,Xabs,C,PuM):
    lrs=[]; wo=[]; w2=[]
    for k in range(3):
        pr=np.where(C[:,k]>0)[0]; true=C[pr,k]*1e6; pred=np.clip(PuM[pr,k],1e-4,None)
        lr=np.abs(np.log10(pred/true)); lrs+=list(lr); wo+=list(lr<1); w2+=list(lr<np.log10(2))
    print(f"    {tag:<10} within-2x {np.mean(w2):.0%}  within-order {np.mean(wo):.0%}  median |log ratio| {np.median(lrs):.2f}")
SETS = {"Ratio_mix (1000µM range)": (os.path.join(acf,"Ratio","Ratio_mix"),1e-6),
        "Binary+Tertiary": None}
# Binary+Tertiary combined
Xb1,Cb1=load_set(os.path.join(acf,"Ratio","Binary"),1e-5); Xb2,Cb2=load_set(os.path.join(acf,"Ratio","Tertiary"),1e-5)
Xbt=np.vstack([Xb1,Xb2]); Cbt=np.vstack([Cb1,Cb2])
Xrm,Crm=load_set(os.path.join(acf,"Ratio","Ratio_mix"),1e-6)
for tag,(Xa,C) in [("Ratio_mix ONLY",(Xrm,Crm)),("Binary+Tertiary ONLY",(Xbt,Cbt)),
                   ("COMBINED",(np.vstack([Xrm,Xbt]),np.vstack([Crm,Cbt])))]:
    print(f"\n{tag}  (n={len(Xa)}):")
    summ("physics",Xa,C,phys(Xa,C)); summ("DL",Xa,C,dl_loo(Xa,C))
