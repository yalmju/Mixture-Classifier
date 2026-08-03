# -*- coding: utf-8 -*-
"""Absolute concentration (µM) on the consolidated Ratio_mix set ONLY (34 mixtures,
direct µM in the filename, one consistent measurement session — do NOT mix with the
Binary/Tertiary set). DL (spectrum→log10 µM, leave-one-out) vs physics inversion.
Saves docs/concentration_pred.png and prints metrics."""
import os, sys, glob
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
root = "S:/Google Drive"
acf = next(os.path.join(e.path, "ACF_PEST_DB") for e in os.scandir(root)
           if e.is_dir() and os.path.isdir(os.path.join(e.path, "ACF_PEST_DB")))
pure = os.path.join(acf, "Pure"); folder = os.path.join(acf, "Ratio", "Ratio_mix")
from unmix import _templates, _baseline_removed, _l2
from real_data import load_map
from dataset import is_blank
from sers_mixture import als_baseline
from validate import parse_mixture_label
from calibration import calibrate, quantify
from io_utils import load_calibration_csv
import torch, torch.nn as nn
from sklearn.cross_decomposition import PLSRegression
LO, HI = 300.0, 1800.0
names, wn, means = _templates(pure, True, None); m = (wn >= LO) & (wn <= HI)
nb = [names[i] for i in range(len(names)) if not is_blank(names[i])]
nonbg = [i for i in range(len(names)) if not is_blank(names[i])]
P = _l2(_baseline_removed(means[nonbg][:, m], True))
ax_c, nc, dils = load_calibration_csv(os.path.join(acf, "Pest", "Standard", "260729", "calibration_spectra.csv"))
mc = (ax_c >= LO) & (ax_c <= HI)
calib = calibrate([(dils[nc.index(n)][0], np.asarray(dils[nc.index(n)][1])[:, mc]) for n in nb], P, nb)
def mix_abs(p):
    _w, c, _mn, _c = load_map(p); c = np.asarray(c, float)[:, m]
    w = c.sum(1); w = w/(w.sum()+1e-12); mean = w@c
    return np.clip(mean - als_baseline(mean), 0, None)
Xabs, C = [], []
for p in sorted(glob.glob(os.path.join(folder, "*orrected.csv"))):
    t = parse_mixture_label(os.path.basename(p), nb)
    if t and len(t) >= 2:
        Xabs.append(mix_abs(p)); C.append([t.get(n, 0.0)*1e-6 for n in nb])   # µM → M
Xabs = np.array(Xabs, np.float64); C = np.array(C, np.float64); N = len(Xabs)
print(f"Ratio_mix: {N} mixtures · µM range {C[C>0].min()*1e6:.0f}–{C.max()*1e6:.0f}")
FLOOR = 1e-8
def net_(nf): return nn.Sequential(nn.Linear(nf,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.15),nn.Linear(256,64),nn.ReLU(),nn.Linear(64,3))
class CNN1D(nn.Module):                                   # 1-D conv over the spectrum
    def __init__(self):
        super().__init__()
        self.b=nn.Sequential(nn.Conv1d(1,16,7,padding=3),nn.ReLU(),nn.MaxPool1d(2),
                             nn.Conv1d(16,32,5,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(8),
                             nn.Flatten(),nn.Linear(32*8,64),nn.ReLU(),nn.Linear(64,3))
    def forward(self,x): return self.b(x[:,None,:])
def _torch_loo(build,epochs=400):                         # shared standardised log10-µM regressor loop
    Pm=np.zeros((N,3))
    for i in range(N):
        tr=[j for j in range(N) if j!=i]; mu=Xabs[tr].mean(0); sd=Xabs[tr].std(0)+1e-8
        Xt=torch.tensor(((Xabs[tr]-mu)/sd).astype(np.float32))
        Yt=torch.tensor((np.log10(np.clip(C[tr],FLOOR,None))+6).astype(np.float32))
        torch.manual_seed(0); net=build(); op=torch.optim.Adam(net.parameters(),lr=3e-4,weight_decay=1e-3)
        for _ in range(epochs): net.train(); op.zero_grad(); ((net(Xt)-Yt)**2).mean().backward(); op.step()
        net.eval()
        with torch.no_grad():
            Pm[i]=10.0**np.clip(net(torch.tensor(((Xabs[i]-mu)/sd)[None,:].astype(np.float32))).numpy()[0],-3,6)
    return Pm
def dl_loo():  return _torch_loo(lambda: net_(Xabs.shape[1]))
def cnn_loo(): return _torch_loo(CNN1D)
def pls_loo():
    Pm=np.zeros((N,3)); Yl=np.log10(np.clip(C,FLOOR,None))+6
    for i in range(N):
        tr=[j for j in range(N) if j!=i]; mu=Xabs[tr].mean(0); sd=Xabs[tr].std(0)+1e-8
        m=PLSRegression(n_components=min(8,len(tr)-1)); m.fit((Xabs[tr]-mu)/sd,Yl[tr])
        Pm[i]=10.0**np.clip(m.predict(((Xabs[i]-mu)/sd)[None,:])[0],-3,6)
    return Pm
def phys():                                               # calibration (Langmuir) inversion
    Pm=np.zeros((N,3))
    for i in range(N):
        pres=[k for k in range(3) if C[i,k]>0]; Pm[i]=quantify(Xabs[i],P,calib,present=pres)["C"]*1e6
    return Pm
# method label → per-mixture µM prediction (leave-one-out).  "Calibration inversion" is NOT
# machine learning: it solves the competitive-Langmuir model (fit from the single-component
# standard curves) for concentration.  The other three learn spectrum→log10 µM from the mixtures.
METHODS={"Calibration inversion (Langmuir)":phys(),"PLS regression":pls_loo(),
         "1D-CNN":cnn_loo(),"MLP-DL (ours)":dl_loo()}
Pph=METHODS["Calibration inversion (Langmuir)"]; Pdl=METHODS["MLP-DL (ours)"]
def stats(tag, PuM):
    print(f"\n{tag}:")
    for k,n in enumerate(nb):
        pr=np.where(C[:,k]>0)[0]; true=C[pr,k]*1e6; pred=np.clip(PuM[pr,k],1e-4,None)
        lr=np.abs(np.log10(pred/true)); yt=np.log10(true); yp=np.log10(pred)
        r2=1-np.sum((yt-yp)**2)/max(np.sum((yt-yt.mean())**2),1e-9)
        print(f"  {n}: R²(log) {r2:+.2f}  within-2x {np.mean(lr<np.log10(2)):.0%}  within-order {np.mean(lr<1):.0%}  (n={len(pr)})")
for _tag,_P in METHODS.items(): stats(_tag,_P)
COL={"DQ":"#1a73e8","TBZ":"#4a9e2a","THI":"#d6336c"}
INK="#1c2430"; MUTE="#5b6673"; FAINT="#c7ccd2"
lims=[3,2500]
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,5.4),sharex=True,sharey=True)
for ax,PuM,tag in [(a1,Pph,"Calibration inversion (Langmuir)"),(a2,Pdl,"DL regression (ours)")]:
    g=np.geomspace(lims[0],lims[1],50)
    ax.fill_between(g,0.5*g,2*g,color="#e9edf1",zorder=0)           # ±2× band (shaded)
    ax.plot(lims,lims,ls="--",color=MUTE,lw=1.2,zorder=1)          # ideal
    for k,n in enumerate(nb):
        pr=np.where(C[:,k]>0)[0]
        ax.scatter(C[pr,k]*1e6,np.clip(PuM[pr,k],lims[0]*0.5,lims[1]*2),s=52,color=COL[n],
                   alpha=0.8,edgecolors="white",linewidths=0.7,label=n,zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect("equal"); ax.set_xlabel("true concentration (µM)",fontsize=10,color=INK)
    ax.grid(True,which="major",color=FAINT,lw=0.4,alpha=0.5)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(FAINT); ax.spines["bottom"].set_color(FAINT); ax.tick_params(colors=MUTE)
a1.set_ylabel("predicted concentration (µM)",fontsize=10,color=INK)
a1.legend(fontsize=10,framealpha=0,loc="upper left",title="component",title_fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(os.getcwd(),"docs","concentration_pred.png"),dpi=130,facecolor="white",bbox_inches="tight")
# also export the data so it can be re-plotted elsewhere
import csv as _csv
_mk=list(METHODS.keys())
with open(os.path.join(os.getcwd(),"docs","concentration_pred.csv"),"w",newline="",encoding="utf-8") as f:
    w=_csv.writer(f); w.writerow(["component","true_uM"]+[f"{m}_uM" for m in _mk])
    for k,n in enumerate(nb):
        for i in np.where(C[:,k]>0)[0]:
            w.writerow([n,f"{C[i,k]*1e6:.4g}"]+[f"{METHODS[m][i,k]:.4g}" for m in _mk])
print("\nsaved docs/concentration_pred.png + docs/concentration_pred.csv")

# ---- strength-first figure: cumulative accuracy curve + fold-error spread (all methods) ----
def foldall(PuM):                                            # |log10 fold| over all present components
    return np.concatenate([np.abs(np.log10(np.clip(PuM[np.where(C[:,k]>0)[0],k],1e-4,None)/(C[np.where(C[:,k]>0)[0],k]*1e6)))
                           for k in range(3)])
MC={"Calibration inversion (Langmuir)":"#8b95a1","PLS regression":"#1a73e8","1D-CNN":"#d64545","MLP-DL (ours)":"#2f9e44"}
FOLD={m:foldall(P) for m,P in METHODS.items()}
tol=np.linspace(0,1.3,140)                                    # log10 tolerance (0=exact … 1.3≈20x)
fig2,(b1,b2)=plt.subplots(1,2,figsize=(11.6,4.9),gridspec_kw={"width_ratios":[1.28,1]})
# panel A — cumulative accuracy, one line per method
for m in METHODS:
    f=FOLD[m]; ours="ours" in m
    b1.plot(10**tol,[np.mean(f<=t)*100 for t in tol],color=MC[m],lw=3.0 if ours else 1.8,
            label=m,zorder=4 if ours else 2)
for xv in (2,10): b1.axvline(xv,color=FAINT,ls=":",lw=1.0)
b1.set_xscale("log"); b1.set_xlim(1,20); b1.set_ylim(0,100)
b1.set_xticks([1,2,3,5,10,20]); b1.set_xticklabels(["1×","2×","3×","5×","10×","20×"])
b1.set_xlabel("accuracy tolerance (predicted within N× of true)",fontsize=10,color=INK)
b1.set_ylabel("% of components within tolerance",fontsize=10,color=INK)
b1.grid(True,color=FAINT,lw=0.4,alpha=0.5); b1.legend(fontsize=8.8,framealpha=0,loc="lower right")
for s in ("top","right"): b1.spines[s].set_visible(False)
# panel B — fold-error spread per method (10^|log fold|, so 1 = perfect)
order=list(METHODS.keys()); data=[10**FOLD[m] for m in order]; pos=list(range(len(order)))
bp=b2.boxplot(data,positions=pos,widths=0.6,patch_artist=True,showfliers=False,
              medianprops=dict(color=INK,lw=1.6),whiskerprops=dict(color=MUTE),capprops=dict(color=MUTE))
for patch,m in zip(bp["boxes"],order): patch.set_facecolor(MC[m]); patch.set_alpha(0.35); patch.set_edgecolor(MC[m])
for j,m in enumerate(order):
    arr=10**FOLD[m]; jit=np.linspace(-0.16,0.16,len(arr))
    b2.scatter(np.full(len(arr),pos[j])+jit,arr,s=13,color=MC[m],alpha=0.55,edgecolors="white",linewidths=0.3,zorder=3)
b2.axhline(2,color=FAINT,ls=":",lw=1.0)
b2.set_yscale("log"); b2.set_ylim(1,60); b2.set_xticks(pos)
b2.set_xticklabels(["Calib.","PLS","1D-CNN","MLP-DL\n(ours)"],fontsize=8.5)
b2.set_ylabel("fold error  |pred / true|  (1 = perfect)",fontsize=10,color=INK)
b2.grid(True,axis="y",color=FAINT,lw=0.4,alpha=0.5)
for s in ("top","right"): b2.spines[s].set_visible(False)
fig2.tight_layout()
fig2.savefig(os.path.join(os.getcwd(),"docs","concentration_accuracy.png"),dpi=130,facecolor="white",bbox_inches="tight")
with open(os.path.join(os.getcwd(),"docs","concentration_accuracy.csv"),"w",newline="",encoding="utf-8") as f:
    w=_csv.writer(f); w.writerow(["fold_tolerance_x"]+[f"{m}_pct_within" for m in order])
    for t in tol: w.writerow([f"{10**t:.3f}"]+[f"{np.mean(FOLD[m]<=t)*100:.2f}" for m in order])
print("saved docs/concentration_accuracy.png + docs/concentration_accuracy.csv")
print("\nwithin-2x / within-order (all components):")
for m in order:
    f=FOLD[m]; print(f"  {m:32s} 2x {np.mean(f<np.log10(2)):.0%}   order {np.mean(f<1):.0%}")
