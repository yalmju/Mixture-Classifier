# -*- coding: utf-8 -*-
"""Ternary classification triangle v2: predicted points coloured by ACCURACY (1−error,
green→red colormap); each corner shows that substance's recovery %.  NNLS vs DL, 35-map LOO.
"""
import os, sys, glob
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

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
from composition import bary
import torch, torch.nn as nn

npz = os.path.join(os.path.dirname(__file__), "tri_preds.npz")
LO, HI = 300.0, 1800.0
names, wn, means = _templates(pure, True, None); m = (wn >= LO) & (wn <= HI)
nb = [names[i] for i in range(len(names)) if not is_blank(names[i])]
nonbg = [i for i in range(len(names)) if not is_blank(names[i])]
P = _l2(_baseline_removed(means[nonbg][:, m], True))

if os.path.exists(npz):
    d = np.load(npz); Y, Pn, Pd = d["Y"], d["Pn"], d["Pd"]
else:
    cal_csv = os.path.join(acf, "Ratio", "results", "calibration_spectra.csv")
    ax_c, names_c, dils = load_calibration_csv(cal_csv); mc = (ax_c >= LO) & (ax_c <= HI)
    calib = calibrate([(dils[names_c.index(n)][0], np.asarray(dils[names_c.index(n)][1])[:, mc]) for n in nb], P, nb)
    def mix_spec(p):
        _w, c, _mn, _c = load_map(p); c = np.asarray(c, float)[:, m]
        w = c.sum(1); w = w/(w.sum()+1e-12); mean = w@c
        y = np.clip(mean-als_baseline(mean), 0, None); return y/(np.linalg.norm(y)+1e-12)
    X, S, Y = [], [], []
    for folder in folders:
        for p in sorted(glob.glob(os.path.join(folder, "*_corrected.csv"))):
            t = parse_mixture_label(os.path.basename(p), nb)
            if t and len(t) >= 2:
                y = mix_spec(p); X.append(y); S.append(surface_composition(y[None,:],P)[0]); Y.append(_ratio([t.get(n,0) for n in nb]))
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
        n2=net_(); n2.load_state_dict(bs); op=torch.optim.Adam(n2.parameters(),lr=3e-4,weight_decay=1e-3)
        Xt_=torch.tensor(X[tr].astype(np.float32)); Yt_=torch.tensor(Y[tr].astype(np.float32))
        for _ in range(250):
            n2.train(); op.zero_grad(); (sm(n2(Xt_)).exp()-Yt_).abs().sum(1).mean().backward(); op.step()
        n2.eval()
        with torch.no_grad(): return torch.softmax(n2(torch.tensor(X[i][None,:].astype(np.float32))),1).numpy()[0]
    Pn=S.copy(); Pd=np.array([dl_pred([j for j in range(N) if j!=i],i) for i in range(N)])
    np.savez(npz, Y=Y, Pn=Pn, Pd=Pd)

N=len(Y)
INK="#1c2430"; MUTE="#5b6673"; FAINT="#c7ccd2"
cmap = plt.get_cmap("RdYlGn"); EMAX = 0.6                 # error→colour (0=green,0.6+=red)
def recovery(Pm, s):
    v=[Pm[i][s]/Y[i][s]*100 for i in range(N) if Y[i][s]>0.02]
    if not v: return float("nan"), 0.0
    v=np.array(v); se=v.std(ddof=1)/np.sqrt(len(v)) if len(v)>1 else 0.0
    return float(v.mean()), float(se)

def panel(ax, Pm, title):
    A,B,C = bary([1,0,0]), bary([0,0,1]), bary([0,1,0])
    for t in (0.25,0.5,0.75):
        for Pp,Q,R in [(A,B,C),(B,A,C),(C,A,B)]:
            ax.plot([Pp[0]+t*(Q[0]-Pp[0]),Pp[0]+t*(R[0]-Pp[0])],[Pp[1]+t*(Q[1]-Pp[1]),Pp[1]+t*(R[1]-Pp[1])],color=FAINT,lw=0.5,alpha=0.4,zorder=0)
    ax.plot([A[0],B[0],C[0],A[0]],[A[1],B[1],C[1],A[1]],color=INK,lw=1.0,zorder=1)
    for f,s,ha,va,dx,dy,col,si in [([1,0,0],"DQ","center","bottom",0,0.05,"#1a73e8",0),([0,0,1],"THI","right","top",-0.03,-0.02,"#d6336c",2),([0,1,0],"TBZ","left","top",0.03,-0.02,"#4a9e2a",1)]:
        p=bary(f); rec,se=recovery(Pm,si)
        ax.text(p[0]+dx,p[1]+dy,f"{s}\nrecovery {rec:.0f}±{se:.0f}%",ha=ha,va=va,fontsize=10,fontweight="bold",color=col,linespacing=1.25)
    for i in range(N):
        p0=bary(Y[i]); p1=bary(Pm[i]); e=0.5*np.abs(Pm[i]-Y[i]).sum()
        ax.scatter(*p0,s=34,facecolors="none",edgecolors=MUTE,linewidths=1.0,zorder=3)
        if np.linalg.norm(p1-p0)>1e-3:
            ax.add_patch(FancyArrowPatch(p0,p1,arrowstyle="-|>",mutation_scale=7,color="#b6bcc4",lw=0.8,zorder=2,shrinkA=2,shrinkB=2))
        ax.scatter(*p1,s=46,color=cmap(1-min(e/EMAX,1)),edgecolors="white",linewidths=0.5,zorder=4)
    me=np.mean([0.5*np.abs(Pm[i]-Y[i]).sum() for i in range(N)])
    ax.set_title(f"{title}\nmean composition error {me:.0%}",fontsize=11,fontweight="bold",color=INK,pad=8)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.16,1.16); ax.set_ylim(-0.13,1.10)

fig,axes=plt.subplots(1,2,figsize=(12,6))
panel(axes[0],Pn,"NNLS (classical unmixing)")
panel(axes[1],Pd,"full-spectrum DL (ours)")
sm2=ScalarMappable(norm=Normalize(0,1),cmap=plt.get_cmap("RdYlGn"))
cb=fig.colorbar(sm2,ax=axes,fraction=0.025,pad=0.02)
cb.set_label("prediction accuracy   (1 = exact · green good, red poor)",fontsize=9)
cb.set_ticks([0,0.5,1.0])
h=[Line2D([],[],marker="o",mfc="none",mec=MUTE,mew=1.0,ls="",ms=8,label="true composition"),
   Line2D([],[],marker="o",mfc="#4a9e2a",mec="white",ls="",ms=8,label="prediction (colour = accuracy)")]
fig.legend(handles=h,loc="lower center",ncol=2,fontsize=9,framealpha=0,bbox_to_anchor=(0.45,-0.02))
fig.suptitle("Composition classification on the ternary simplex  (leave-one-out, 35 real mixtures)",
             fontsize=12.5,fontweight="bold",color=INK,y=1.0)
fig.tight_layout(rect=(0,0.03,1,0.97))
fig.savefig(os.path.join(os.getcwd(),"docs","classify_triangle.png"),dpi=120,facecolor="white",bbox_inches="tight")
print("saved docs/classify_triangle.png")
