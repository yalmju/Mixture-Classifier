# -*- coding: utf-8 -*-
"""Grouped 'collapse vs spread' ternary (figure B).  Each mixture's PREDICTED composition
is plotted, coloured by its TRUE dominant component (DQ blue · TBZ green · THI red).
DL keeps the three groups near their own corners; NNLS collapses most groups toward THI.
Uses cached tri_preds.npz (Y=solution, Pn=NNLS, Pd=DL). Two PNGs, no titles (figure-set ready)."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

D = os.path.join(os.getcwd(), "docs")
d = np.load(os.path.join(os.path.dirname(__file__), "tri_preds.npz"))
Y, Pn, Pd = d["Y"], d["Pn"], d["Pd"]                     # cols: DQ, TBZ, THI
N = len(Y)
dom = np.argmax(Y, 1)                                     # true dominant component per mixture
GC = {0: "#1a73e8", 1: "#4a9e2a", 2: "#d6336c"}          # DQ blue, TBZ green, THI red
GN = {0: "DQ-dominant", 1: "TBZ-dominant", 2: "THI-dominant"}

H = 3**0.5/2; TOP=np.array([0.5,H]); BL=np.array([0.0,0.0]); BR=np.array([1.0,0.0])
def bary(f):                                              # THI top · TBZ bottom-left · DQ bottom-right
    f=np.asarray(f,float); f=f/(f.sum()+1e-12)
    return f[2]*TOP + f[1]*BL + f[0]*BR

def draw(ax, Pm):
    A,Bv,Cc = BR, TOP, BL                                # DQ, THI, TBZ corners
    for t in (0.25,0.5,0.75):
        for P,Q,R in [(A,Bv,Cc),(Bv,A,Cc),(Cc,A,Bv)]:
            ax.plot([P[0]+t*(Q[0]-P[0]),P[0]+t*(R[0]-P[0])],[P[1]+t*(Q[1]-P[1]),P[1]+t*(R[1]-P[1])],
                    color="#d7dbe0",lw=0.7,zorder=0)
    ax.plot([TOP[0],BL[0],BR[0],TOP[0]],[TOP[1],BL[1],BR[1],TOP[1]],color="#1c2430",lw=1.3,zorder=1)
    for f,s,va,dy,col in [([0,0,1],"THI","bottom",0.05,"#d6336c"),([0,1,0],"TBZ","top",-0.06,"#4a9e2a"),([1,0,0],"DQ","top",-0.06,"#1a73e8")]:
        p=bary(f); ax.text(p[0],p[1]+dy,s,ha="center",va=va,fontsize=16,fontweight="bold",color=col)
    for g in (0,1,2):                                    # predicted points, coloured by TRUE dominant group
        idx=np.where(dom==g)[0]
        pts=np.array([bary(Pm[i]) for i in idx])
        ax.scatter(pts[:,0],pts[:,1],s=110,color=GC[g],alpha=0.6,edgecolors="white",linewidths=0.9,zorder=3)
        c=pts.mean(0)                                    # group centroid = where that group lands on average
        ax.scatter(*c,s=620,marker="*",color=GC[g],edgecolors="#1c2430",linewidths=1.6,zorder=5)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.14,1.14); ax.set_ylim(-0.12,H+0.10)

for Pm, fname in [(Pn,"triangle_group_nnls.png"), (Pd,"triangle_group_dl.png")]:
    fig,ax=plt.subplots(figsize=(7.2,6.9))
    draw(ax,Pm)
    h=[Line2D([],[],marker="o",mfc=GC[g],mec="white",ls="",ms=11,label=GN[g]) for g in (0,1,2)]
    h.append(Line2D([],[],marker="*",mfc="#888",mec="#1c2430",ls="",ms=17,label="group mean"))
    fig.legend(handles=h,loc="lower center",ncol=4,fontsize=10.5,framealpha=0,bbox_to_anchor=(0.5,0.0))
    fig.savefig(os.path.join(D,fname),dpi=130,facecolor="white",bbox_inches="tight")

import csv as _csv
with open(os.path.join(D,"triangle_group.csv"),"w",newline="",encoding="utf-8") as f:
    w=_csv.writer(f); w.writerow(["mixture_idx","true_dominant","nnls_DQ","nnls_TBZ","nnls_THI","dl_DQ","dl_TBZ","dl_THI"])
    for i in range(N):
        w.writerow([i,["DQ","TBZ","THI"][dom[i]]]+[f"{v:.4f}" for v in Pn[i]]+[f"{v:.4f}" for v in Pd[i]])
print("saved docs/triangle_group_nnls.png + docs/triangle_group_dl.png + docs/triangle_group.csv")
