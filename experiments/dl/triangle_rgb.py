# -*- coding: utf-8 -*-
"""Pseudo-colour (composition → RGB) ternary demonstrator of SURFACE DOMINANCE.
Background colour at any point = its composition blended in the substance colours
(DQ blue · TBZ green · THI red); an even 1:1:1 mix would sit at a neutral centre.
Each mixture: open circle = SOLUTION composition, filled dot = MEASURED SURFACE
composition (NNLS); the arrow is the solution→surface drift. Balanced solutions
drift toward the THI (red) corner — i.e. the surface is NOT an even 1/3 each.
Uses the cached tri_preds.npz (Y=solution, Pn=NNLS surface). No titles (figure-set ready)."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

D = os.path.join(os.getcwd(), "docs")
d = np.load(os.path.join(os.path.dirname(__file__), "tri_preds.npz"))
Y, Pn = d["Y"], d["Pn"]                                   # solution, NNLS surface  (cols: DQ,TBZ,THI)
N = len(Y)

# geometry — THI top · TBZ bottom-left · DQ bottom-right
H = 3**0.5/2; TOP=np.array([0.5,H]); BL=np.array([0.0,0.0]); BR=np.array([1.0,0.0])
def bary(f):
    f=np.asarray(f,float); f=f/(f.sum()+1e-12)
    return f[2]*TOP + f[1]*BL + f[0]*BR                   # THI→TOP, TBZ→BL, DQ→BR

# substance RGB (DQ blue, TBZ green, THI red)
C_DQ=np.array([0.10,0.45,0.91]); C_TBZ=np.array([0.29,0.62,0.16]); C_THI=np.array([0.84,0.20,0.42])

def draw(ax):
    # RGB background: every interior pixel coloured by its own composition
    res=520
    gx=np.linspace(-0.02,1.02,res); gy=np.linspace(-0.02,H+0.02,res)
    GX,GY=np.meshgrid(gx,gy)
    det=(BL[1]-BR[1])*(TOP[0]-BR[0])+(BR[0]-BL[0])*(TOP[1]-BR[1])          # bary wrt TOP,BL,BR
    a=((BL[1]-BR[1])*(GX-BR[0])+(BR[0]-BL[0])*(GY-BR[1]))/det              # weight on TOP  = THI
    b=((BR[1]-TOP[1])*(GX-BR[0])+(TOP[0]-BR[0])*(GY-BR[1]))/det            # weight on BL   = TBZ
    c=1-a-b                                                                # weight on BR   = DQ
    inside=(a>=-1e-9)&(b>=-1e-9)&(c>=-1e-9)
    img=np.zeros((res,res,4))
    for w,col in ((a,C_THI),(b,C_TBZ),(c,C_DQ)):
        for k in range(3): img[:,:,k]+=np.clip(w,0,None)*col[k]
    img[:,:,3]=np.where(inside,0.55,0.0)                                   # alpha inside only
    ax.imshow(img,extent=[gx[0],gx[-1],gy[0],gy[-1]],origin="lower",zorder=0,interpolation="bilinear")
    # simplex edges + light gridlines
    A,Bv,Cc=BR,TOP,BL
    for t in (0.25,0.5,0.75):
        for P,Q,R in [(A,Bv,Cc),(Bv,A,Cc),(Cc,A,Bv)]:
            ax.plot([P[0]+t*(Q[0]-P[0]),P[0]+t*(R[0]-P[0])],[P[1]+t*(Q[1]-P[1]),P[1]+t*(R[1]-P[1])],
                    color="white",lw=0.6,alpha=0.5,zorder=1)
    ax.plot([TOP[0],BL[0],BR[0],TOP[0]],[TOP[1],BL[1],BR[1],TOP[1]],color="#1c2430",lw=1.2,zorder=2)
    # corner labels
    for f,s,va,dy,col in [([0,0,1],"THI","bottom",0.05,"#d6336c"),([0,1,0],"TBZ","top",-0.06,"#4a9e2a"),([1,0,0],"DQ","top",-0.06,"#1a73e8")]:
        p=bary(f); ax.text(p[0],p[1]+dy,s,ha="center",va=va,fontsize=15,fontweight="bold",color=col)
    # solution → surface drift
    for i in range(N):
        p0=bary(Y[i]); p1=bary(Pn[i])
        if np.linalg.norm(p1-p0)>1e-3:
            ax.add_patch(FancyArrowPatch(p0,p1,arrowstyle="-|>",mutation_scale=14,color="#2b2f36",
                         lw=1.4,alpha=0.8,zorder=3,shrinkA=4,shrinkB=6))
        ax.scatter(*p0,s=62,facecolors="white",edgecolors="#1c2430",linewidths=1.6,zorder=4)
        ax.scatter(*p1,s=95,facecolors="#1c2430",edgecolors="white",linewidths=1.0,zorder=5)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.14,1.14); ax.set_ylim(-0.12,H+0.10)

fig,ax=plt.subplots(figsize=(7.4,7.0))
draw(ax)
h=[Line2D([],[],marker="o",mfc="white",mec="#1c2430",mew=1.6,ls="",ms=11,label="solution composition (true)"),
   Line2D([],[],marker="o",mfc="#1c2430",mec="white",ls="",ms=11,label="measured surface (NNLS)")]
fig.legend(handles=h,loc="lower center",ncol=2,fontsize=11,framealpha=0,bbox_to_anchor=(0.5,0.0))
fig.savefig(os.path.join(D,"triangle_rgb_surface.png"),dpi=130,facecolor="white",bbox_inches="tight")
# redraw data: solution vs measured-surface composition per mixture
import csv as _csv
with open(os.path.join(D,"triangle_rgb_surface.csv"),"w",newline="",encoding="utf-8") as f:
    w=_csv.writer(f); w.writerow(["mixture_idx","sol_DQ","sol_TBZ","sol_THI","surf_DQ","surf_TBZ","surf_THI"])
    for i in range(N):
        w.writerow([i]+[f"{v:.4f}" for v in Y[i]]+[f"{v:.4f}" for v in Pn[i]])
print("saved docs/triangle_rgb_surface.png + docs/triangle_rgb_surface.csv")
