# -*- coding: utf-8 -*-
"""Absolute concentration (µM) — combined dataset with a real concentration axis.
  Binary + Tertiary : µM = filename code x 10   (code 1->10µM ... 100->1mM)
  Ratio_mix         : µM = filename code directly (DQ1000 = 1000 µM)
DL (spectrum->log10 µM, leave-one-out) vs physics inversion. Saves a predicted-vs-true
µM log-log scatter (docs/concentration_pred.png) and prints metrics."""
import os, sys, glob
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
root = "S:/Google Drive"
acf = next(os.path.join(e.path, "ACF_PEST_DB") for e in os.scandir(root)
           if e.is_dir() and os.path.isdir(os.path.join(e.path, "ACF_PEST_DB")))
pure = os.path.join(acf, "Pure")
SETS = [(os.path.join(acf, "Ratio", "Binary"), 1e-5), (os.path.join(acf, "Ratio", "Tertiary"), 1e-5),
        (os.path.join(acf, "Ratio", "Ratio_mix"), 1e-6)]   # (folder, code->M factor)
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
Xabs, C = [], []
for folder, fac in SETS:
    for p in sorted(glob.glob(os.path.join(folder, "*orrected.csv"))):
        t = parse_mixture_label(os.path.basename(p), nb)
        if t and len(t) >= 2:
            Xabs.append(mix_abs(p)); C.append([t.get(n, 0.0)*fac for n in nb])
Xabs = np.array(Xabs, np.float64); C = np.array(C, np.float64); N = len(Xabs)
print(f"{N} mixtures · µM range {C[C>0].min()*1e6:.2g}–{C.max()*1e6:.0f}")
FLOOR = 1e-8
def net_(nf): return nn.Sequential(nn.Linear(nf,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.15),nn.Linear(256,64),nn.ReLU(),nn.Linear(64,3))
def dl_loo():
    Pm = np.zeros((N,3))
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
def phys():
    Pm=np.zeros((N,3))
    for i in range(N):
        pres=[k for k in range(3) if C[i,k]>0]; q=quantify(Xabs[i],P,calib,present=pres); Pm[i]=q["C"]*1e6
    return Pm
Pdl=dl_loo(); Pph=phys()
def stats(tag, PuM):
    print(f"\n{tag}:")
    for k,n in enumerate(nb):
        pr=np.where(C[:,k]>0)[0]; true=C[pr,k]*1e6; pred=np.clip(PuM[pr,k],1e-4,None)
        lr=np.abs(np.log10(pred/true)); yt=np.log10(true); yp=np.log10(pred)
        r2=1-np.sum((yt-yp)**2)/max(np.sum((yt-yt.mean())**2),1e-9)
        print(f"  {n}: R²(log) {r2:+.2f}  within-2x {np.mean(lr<np.log10(2)):.0%}  within-order {np.mean(lr<1):.0%}  (n={len(pr)})")
stats("PHYSICS inversion", Pph); stats("DL regression", Pdl)
# figure: predicted vs true µM (log-log), per compound colour
COL={"DQ":"#1a73e8","TBZ":"#4a9e2a","THI":"#d6336c"}
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,5.4),sharex=True,sharey=True)
for ax,PuM,tag in [(a1,Pph,"Physics inversion"),(a2,Pdl,"DL regression (ours)")]:
    for k,n in enumerate(nb):
        pr=np.where(C[:,k]>0)[0]; ax.scatter(C[pr,k]*1e6, np.clip(PuM[pr,k],1e-4,None), s=26, color=COL[n], alpha=0.8, edgecolors="white", linewidths=0.4, label=n)
    lims=[0.05,3000]; ax.plot(lims,lims,ls="--",color="#8b95a1",lw=1)
    ax.plot(lims,[2*x for x in lims],ls=":",color="#c7ccd2",lw=1); ax.plot(lims,[0.5*x for x in lims],ls=":",color="#c7ccd2",lw=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("true concentration (µM)"); ax.set_title(tag, fontweight="bold", fontsize=11)
    for s in ("top","right"): ax.spines[s].set_visible(False)
a1.set_ylabel("predicted (µM)"); a1.legend(fontsize=9, framealpha=0)
fig.suptitle("Absolute concentration — predicted vs true (log-log; dashed=ideal, dotted=±2×)", fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0,0,1,0.96)); fig.savefig(os.path.join(os.getcwd(),"docs","concentration_pred.png"), dpi=120, facecolor="white")
print("\nsaved docs/concentration_pred.png")
