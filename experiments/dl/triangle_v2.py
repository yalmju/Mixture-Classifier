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
folders = [os.path.join(acf, "Ratio", "Ratio_mix")]     # consolidated 34-mixture set (same as concentration)
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
        for p in sorted(glob.glob(os.path.join(folder, "*orrected.csv"))):   # '_corrected' & '-corrected'
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

_H=3**0.5/2; _TOP=np.array([0.5,_H]); _BL=np.array([0.0,0.0]); _BR=np.array([1.0,0.0])
def bary(frac):                                              # THI top · TBZ bottom-left · DQ bottom-right
    f=np.asarray(frac,float)
    return f[2]*_TOP + f[1]*_BL + f[0]*_BR

def panel(ax, Pm):
    A,B,C = bary([1,0,0]), bary([0,0,1]), bary([0,1,0])
    # accuracy heatmap over the interior — IDW from each sample's accuracy at its true composition
    gx=np.linspace(min(A[0],B[0],C[0]),max(A[0],B[0],C[0]),260)
    gy=np.linspace(min(A[1],B[1],C[1]),max(A[1],B[1],C[1]),260)
    GX,GY=np.meshgrid(gx,gy)
    det=(B[1]-C[1])*(A[0]-C[0])+(C[0]-B[0])*(A[1]-C[1])
    l1=((B[1]-C[1])*(GX-C[0])+(C[0]-B[0])*(GY-C[1]))/det
    l2=((C[1]-A[1])*(GX-C[0])+(A[0]-C[0])*(GY-C[1]))/det
    inside=(l1>=-1e-9)&(l2>=-1e-9)&((1-l1-l2)>=-1e-9)
    pts=np.array([bary(Y[i]) for i in range(N)])
    acc=np.array([1-min((0.5*np.abs(Pm[i]-Y[i]).sum())/EMAX,1) for i in range(N)])
    gi=np.stack([GX.ravel(),GY.ravel()],1)                # smooth IDW over the whole grid …
    d2=((gi[:,None,0]-pts[None,:,0])**2+(gi[:,None,1]-pts[None,:,1])**2)+3e-3
    wv=1.0/d2**0.9
    Z=((wv*acc[None,:]).sum(1)/wv.sum(1)).reshape(GX.shape)
    try:
        from scipy.ndimage import gaussian_filter
        Z=gaussian_filter(Z,sigma=7)                      # … then blur out the blotches
    except Exception:
        pass
    ax.pcolormesh(GX,GY,np.ma.masked_where(~inside,Z),cmap=cmap,vmin=0,vmax=1,
                  alpha=0.42,shading="gouraud",zorder=0)
    for t in (0.25,0.5,0.75):
        for Pp,Q,R in [(A,B,C),(B,A,C),(C,A,B)]:
            ax.plot([Pp[0]+t*(Q[0]-Pp[0]),Pp[0]+t*(R[0]-Pp[0])],[Pp[1]+t*(Q[1]-Pp[1]),Pp[1]+t*(R[1]-Pp[1])],color=FAINT,lw=0.5,alpha=0.4,zorder=0)
    ax.plot([A[0],B[0],C[0],A[0]],[A[1],B[1],C[1],A[1]],color=INK,lw=1.0,zorder=1)
    for f,s,ha,va,dx,dy,col,si in [([0,0,1],"THI","center","bottom",0,0.06,"#d6336c",2),([0,1,0],"TBZ","center","top",0,-0.06,"#4a9e2a",1),([1,0,0],"DQ","center","top",0,-0.06,"#1a73e8",0)]:
        p=bary(f); rec,se=recovery(Pm,si)
        ax.text(p[0]+dx,p[1]+dy,f"{s}\n{rec:.0f}±{se:.0f}%",ha=ha,va=va,fontsize=15,fontweight="bold",color=col,linespacing=1.3)
    for i in range(N):
        p0=bary(Y[i]); p1=bary(Pm[i]); e=0.5*np.abs(Pm[i]-Y[i]).sum()
        if np.linalg.norm(p1-p0)>1e-3:                       # drift arrow (soft grey)
            ax.add_patch(FancyArrowPatch(p0,p1,arrowstyle="-|>",mutation_scale=14,
                         color="#6b7280",lw=1.5,alpha=0.8,zorder=2,shrinkA=4,shrinkB=6))
        ax.scatter(*p0,s=60,facecolors="white",edgecolors="#333333",linewidths=1.5,zorder=3)
        ax.scatter(*p1,s=115,color=cmap(1-min(e/EMAX,1)),edgecolors="white",linewidths=0.9,zorder=4)
    print(f"  {'panel'}: {N} true points plotted")
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.16,1.16); ax.set_ylim(-0.13,1.10)

def save_one(Pm, fname):                                     # one full-size triangle per PNG (no overlap)
    fig,ax=plt.subplots(figsize=(7.6,7.2))
    panel(ax,Pm)
    sm2=ScalarMappable(norm=Normalize(0,1),cmap=plt.get_cmap("RdYlGn"))
    cb=fig.colorbar(sm2,ax=ax,fraction=0.03,pad=0.02,shrink=0.42,aspect=20)   # small scale bar
    cb.set_label("accuracy (green = exact, red = poor)",fontsize=11); cb.set_ticks([0,0.5,1.0]); cb.ax.tick_params(labelsize=10)
    h=[Line2D([],[],marker="o",mfc="none",mec=MUTE,mew=1.4,ls="",ms=11,label="true composition"),
       Line2D([],[],marker="o",mfc="#4a9e2a",mec="white",ls="",ms=12,label="prediction (colour = accuracy)")]
    fig.legend(handles=h,loc="lower center",ncol=2,fontsize=11,framealpha=0,bbox_to_anchor=(0.5,0.0))
    fig.savefig(os.path.join(os.getcwd(),"docs",fname),dpi=130,facecolor="white",bbox_inches="tight")
save_one(Pn,"classify_triangle_nnls.png")
save_one(Pd,"classify_triangle_dl.png")
print("saved docs/classify_triangle_nnls.png + docs/classify_triangle_dl.png")
