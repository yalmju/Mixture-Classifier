# -*- coding: utf-8 -*-
"""Substrate-band gain normalisation: the ink has a substrate peak ~2100 cm-1 (Raman-silent
region → concentration-independent, gain-proportional). Dividing each spectrum by it should
cancel the per-session gain and let absolute µM transfer across sessions.

Compares COMBINED-set µM (Binary+Tertiary + Ratio_mix, 69 mixtures) with and without
substrate normalisation. If normalised ≈ within-session (~70% within-order), it works."""
import os, sys, glob
sys.path.insert(0, os.getcwd())
import numpy as np
root = "S:/Google Drive"
acf = next(os.path.join(e.path, "ACF_PEST_DB") for e in os.scandir(root)
           if e.is_dir() and os.path.isdir(os.path.join(e.path, "ACF_PEST_DB")))
pure = os.path.join(acf, "Pure")
from unmix import _templates
from real_data import load_map
from dataset import is_blank
from sers_mixture import als_baseline
from validate import parse_mixture_label
import torch, torch.nn as nn

names, wn, means = _templates(pure, True, None)
nb = [names[i] for i in range(len(names)) if not is_blank(names[i])]
ANA = (wn >= 300) & (wn <= 1800)                     # analytical window
SUB = (wn >= 2050) & (wn <= 2150)                    # substrate peak window (~2100)
print("substrate window points:", int(SUB.sum()), " ~", f"{wn[SUB].min():.0f}-{wn[SUB].max():.0f}")

def spectra(path):
    _w, c, _mn, _c = load_map(path); c = np.asarray(c, float)
    w = c.sum(1); w = w/(w.sum()+1e-12); mean = w @ c
    bl = mean - als_baseline(mean)
    ana = np.clip(bl[ANA], 0, None)
    sub_ref = float(np.clip(bl[SUB], 0, None).max())   # substrate peak height
    return ana, sub_ref

SETS = [(os.path.join(acf,"Ratio","Binary"),1e-5,"B"), (os.path.join(acf,"Ratio","Tertiary"),1e-5,"B"),
        (os.path.join(acf,"Ratio","Ratio_mix"),1e-6,"R")]
Xa, C, sub, sess = [], [], [], []
for folder, fac, tag in SETS:
    for p in sorted(glob.glob(os.path.join(folder,"*orrected.csv"))):
        t = parse_mixture_label(os.path.basename(p), nb)
        if t and len(t) >= 2:
            a, s = spectra(p); Xa.append(a); sub.append(s); sess.append(tag)
            C.append([t.get(n,0.0)*fac for n in nb])
Xa = np.array(Xa, np.float64); C = np.array(C, np.float64); sub = np.array(sub); sess = np.array(sess)
N = len(Xa)
print(f"\n{N} mixtures. substrate peak height by session (gain proxy):")
for tg in ("B","R"):
    s = sub[sess==tg]; print(f"  {tg}: n={len(s)}  median {np.median(s):.0f}  (min {s.min():.0f} max {s.max():.0f})")
print(f"  → session gain ratio (R/B, by substrate peak): {np.median(sub[sess=='R'])/np.median(sub[sess=='B']):.2f}x")

FLOOR=1e-8
def net_(nf): return nn.Sequential(nn.Linear(nf,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.15),nn.Linear(256,64),nn.ReLU(),nn.Linear(64,3))
def dl_loo(X):
    Pm=np.zeros((N,3))
    for i in range(N):
        tr=[j for j in range(N) if j!=i]; mu=X[tr].mean(0); sd=X[tr].std(0)+1e-8
        Xt=((X[tr]-mu)/sd).astype(np.float32); Xe=((X[i]-mu)/sd).astype(np.float32)
        Yt=(np.log10(np.clip(C[tr],FLOOR,None))+6).astype(np.float32)
        torch.manual_seed(0); net=net_(Xt.shape[1]); op=torch.optim.Adam(net.parameters(),lr=3e-4,weight_decay=1e-3)
        Xtt=torch.tensor(Xt); Ytt=torch.tensor(Yt)
        for _ in range(400): net.train(); op.zero_grad(); ((net(Xtt)-Ytt)**2).mean().backward(); op.step()
        net.eval()
        with torch.no_grad(): Pm[i]=10.0**net(torch.tensor(Xe[None,:])).numpy()[0]
    return Pm
def summ(tag,PuM):
    lr=[]
    for k in range(3):
        pr=np.where(C[:,k]>0)[0]; true=C[pr,k]*1e6; pred=np.clip(PuM[pr,k],1e-4,None)
        lr+=list(np.abs(np.log10(pred/true)))
    lr=np.array(lr); print(f"  {tag:<26} within-2x {np.mean(lr<np.log10(2)):.0%}  within-order {np.mean(lr<1):.0%}  median {np.median(lr):.2f}")

print("\nCOMBINED µM (DL LOO):")
summ("raw (no normalisation)", dl_loo(Xa))
summ("substrate-normalised", dl_loo(Xa / sub[:,None]))
