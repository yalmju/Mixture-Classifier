# -*- coding: utf-8 -*-
"""A+B combined ternary.  RGB composition background (DQ blue · TBZ green · THI red);
each mixture's PREDICTED composition plotted, coloured by its TRUE dominant component,
with a star at each group's centroid.  Reads as colour-harmony:
  DL  → each group's star sits on its OWN colour region  (star colour ≈ background) = accurate
  NNLS→ stars pulled onto the THI (red) region, clashing with their colour           = collapsed
Uses cached tri_preds.npz (Y, Pn, Pd). Two PNGs, no titles (figure-set ready)."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

D = os.path.join(os.getcwd(), "docs")
d = np.load(os.path.join(os.path.dirname(__file__), "tri_preds.npz"))
Y, Pn, Pd = d["Y"], d["Pn"], d["Pd"]                     # cols: DQ, TBZ, THI
N = len(Y); dom = np.argmax(Y, 1)
GC = {0: "#1a73e8", 1: "#4a9e2a", 2: "#d6336c"}
GN = {0: "DQ-dominant", 1: "TBZ-dominant", 2: "THI-dominant"}
C_DQ=np.array([0.10,0.45,0.91]); C_TBZ=np.array([0.29,0.62,0.16]); C_THI=np.array([0.84,0.20,0.42])

H = 3**0.5/2; TOP=np.array([0.5,H]); BL=np.array([0.0,0.0]); BR=np.array([1.0,0.0])
def bary(f):                                              # THI top · TBZ bottom-left · DQ bottom-right
    f=np.asarray(f,float); f=f/(f.sum()+1e-12)
    return f[2]*TOP + f[1]*BL + f[0]*BR

def draw(ax, Pm):
    res=520
    gx=np.linspace(-0.02,1.02,res); gy=np.linspace(-0.02,H+0.02,res); GX,GY=np.meshgrid(gx,gy)
    det=(BL[1]-BR[1])*(TOP[0]-BR[0])+(BR[0]-BL[0])*(TOP[1]-BR[1])
    a=((BL[1]-BR[1])*(GX-BR[0])+(BR[0]-BL[0])*(GY-BR[1]))/det              # THI weight
    b=((BR[1]-TOP[1])*(GX-BR[0])+(TOP[0]-BR[0])*(GY-BR[1]))/det            # TBZ weight
    c=1-a-b                                                                # DQ weight
    inside=(a>=-1e-9)&(b>=-1e-9)&(c>=-1e-9)
    img=np.zeros((res,res,4))
    for w,col in ((a,C_THI),(b,C_TBZ),(c,C_DQ)):
        for k in range(3): img[:,:,k]+=np.clip(w,0,None)*col[k]
    img[:,:,3]=np.where(inside,0.68,0.0)                                   # stronger tint (less washed out)
    ax.imshow(img,extent=[gx[0],gx[-1],gy[0],gy[-1]],origin="lower",zorder=0,interpolation="bilinear")
    for t in (0.25,0.5,0.75):
        for P,Q,R in [(BR,TOP,BL),(TOP,BR,BL),(BL,BR,TOP)]:
            ax.plot([P[0]+t*(Q[0]-P[0]),P[0]+t*(R[0]-P[0])],[P[1]+t*(Q[1]-P[1]),P[1]+t*(R[1]-P[1])],
                    color="white",lw=0.6,alpha=0.5,zorder=1)
    ax.plot([TOP[0],BL[0],BR[0],TOP[0]],[TOP[1],BL[1],BR[1],TOP[1]],color="#1c2430",lw=1.2,zorder=2)
    # dominant-region margin = Voronoi of the three GROUP CENTROIDS (not a fixed 1/3 split)
    cen=[np.array([bary(Pm[i]) for i in np.where(dom==g)[0]]).mean(0) for g in (0,1,2)]
    def _circum(A,B,C):
        ax_,ay=A; bx,by=B; cx,cy=C
        dd=2*(ax_*(by-cy)+bx*(cy-ay)+cx*(ay-by))
        ux=((ax_*ax_+ay*ay)*(by-cy)+(bx*bx+by*by)*(cy-ay)+(cx*cx+cy*cy)*(ay-by))/dd
        uy=((ax_*ax_+ay*ay)*(cx-bx)+(bx*bx+by*by)*(ax_-cx)+(cx*cx+cy*cy)*(bx-ax_))/dd
        return np.array([ux,uy])
    cc=_circum(*cen); edges=[(TOP,BL),(BL,BR),(BR,TOP)]
    def _hit(o,dv):
        best=None
        for p,q in edges:
            e=np.asarray(q)-np.asarray(p); den=dv[0]*(-e[1])-dv[1]*(-e[0])
            if abs(den)<1e-12: continue
            df=np.asarray(p)-o
            t=(df[0]*(-e[1])-df[1]*(-e[0]))/den; s=(dv[0]*df[1]-dv[1]*df[0])/den
            if t>1e-6 and -1e-6<=s<=1+1e-6 and (best is None or t<best[0]): best=(t,o+t*dv)
        return best[1] if best else o
    for i,j,k in [(0,1,2),(1,2,0),(0,2,1)]:
        d=cen[j]-cen[i]; perp=np.array([-d[1],d[0]]); perp=perp/(np.linalg.norm(perp)+1e-12)
        if np.dot(perp,cc-cen[k])<0: perp=-perp
        end=_hit(cc,perp)
        ax.plot([cc[0],end[0]],[cc[1],end[1]],color="white",lw=4.0,alpha=0.9,solid_capstyle="round",zorder=2)
    for f,s,va,dy,col in [([0,0,1],"THI","bottom",0.05,"#d6336c"),([0,1,0],"TBZ","top",-0.06,"#4a9e2a"),([1,0,0],"DQ","top",-0.06,"#1a73e8")]:
        p=bary(f); ax.text(p[0],p[1]+dy,s,ha="center",va=va,fontsize=16,fontweight="bold",color=col)
    for g in (0,1,2):
        idx=np.where(dom==g)[0]; pts=np.array([bary(Pm[i]) for i in idx])
        ax.scatter(pts[:,0],pts[:,1],s=78,color=GC[g],alpha=0.62,edgecolors="white",linewidths=0.9,zorder=3)
        cc=pts.mean(0)                                          # group centroid marker (open donut, no number)
        ax.scatter(*cc,s=900,facecolors="white",edgecolors=GC[g],linewidths=3.4,alpha=0.95,zorder=5)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.14,1.14); ax.set_ylim(-0.12,H+0.10)

for Pm, fname in [(Pn,"triangle_combo_nnls.png"), (Pd,"triangle_combo_dl.png")]:
    fig,ax=plt.subplots(figsize=(7.2,6.9))
    draw(ax,Pm)
    h=[Line2D([],[],marker="o",mfc=GC[g],mec="white",ls="",ms=11,label=GN[g]) for g in (0,1,2)]
    h.append(Line2D([],[],marker="o",mfc="white",mec="#555",mew=2.4,ls="",ms=15,label="group mean"))
    fig.legend(handles=h,loc="lower center",ncol=4,fontsize=10.5,framealpha=0,bbox_to_anchor=(0.5,0.0))
    fig.savefig(os.path.join(D,fname),dpi=130,facecolor="white",bbox_inches="tight")
print("saved docs/triangle_combo_nnls.png + docs/triangle_combo_dl.png")
