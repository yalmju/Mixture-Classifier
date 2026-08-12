"""논문용 결론 figure — 조제간 재현성.
   · 색: 저장소 ui_common.py 팔레트
   · 그래프 내부에는 글자 없음 (축 라벨·눈금·범례만, 범례는 축 바깥)
"""
import pickle, os, itertools, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

d=os.path.dirname(__file__)
PIX=pickle.load(open(os.path.join(d,"allpix.pkl"),"rb"))
U=pickle.load(open(os.path.join(d,"unmix.pkl"),"rb"))
D=pickle.load(open(os.path.join(d,"data.pkl"),"rb"))
PAIRS=pickle.load(open(os.path.join(d,"n100.pkl"),"rb"))["PAIRS"]

# ============ ui_common.py 팔레트 ============
PAGE,INK,MUTE,FAINT,LINE = "#fafbfc","#1c2430","#5b6673","#98a1ac","#e3e8ee"
TEAL,BLUE,AMBER,CORAL   = "#0f9d6b","#1a73e8","#c98a15","#d8542a"
PURPLE,PINK,GREEN,STEEL = "#6b5fd6","#c85a8f","#4a9e2a","#546170"
SER   = {"DQ":BLUE,"TBZ":GREEN,"THI":PINK}          # 사용자 지정 — 전 검사 통과
MARK  = {"DQ":"o","TBZ":"s","THI":"^"}              # 색 외 2차 부호화
RAMP  = ["#c7ccd3",FAINT,MUTE]                      # 중립 회색 램프 (성분색과 충돌 방지)

plt.rcParams.update({
    "font.family":"Helvetica","font.size":8.5,
    "axes.linewidth":.7,"axes.edgecolor":MUTE,
    "xtick.major.width":.7,"ytick.major.width":.7,
    "xtick.color":MUTE,"ytick.color":MUTE,"xtick.labelsize":8,"ytick.labelsize":8,
    "axes.labelcolor":INK,"text.color":INK,
    "figure.facecolor":PAGE,"axes.facecolor":PAGE,"savefig.facecolor":PAGE,
    "legend.frameon":False,"pdf.fonttype":42,"ps.fonttype":42,
})
def clean(ax):
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.tick_params(length=3,pad=2)
PANELS=[]

# ================= 수치 =================
def cvp(a,b):
    m=(a+b)/2
    return np.nan if m<=0 else np.std([a,b],ddof=1)/m*100
ink=np.nanmedian([cvp(U[("DQ9",k)]["ink"],U[("DQ9-sus",c)]["ink"]) for c,k,_,_ in PAIRS])
tot=lambda key: np.median(PIX[key]["raw"][:,:3].sum(1))/max(np.median(PIX[key]["raw"][:,3]),1e-9)
rec=np.nanmedian([cvp(tot(("DQ9",k)),tot(("DQ9-sus",c))) for c,k,_,_ in PAIRS])
comp={}
for j,nm in enumerate(SER):
    v=[cvp(np.median(PIX[("DQ9",k)]["frac"][:,j]),np.median(PIX[("DQ9-sus",c)]["frac"][:,j]))
       for c,k,_,_ in PAIRS
       if (np.median(PIX[("DQ9",k)]["frac"][:,j])+np.median(PIX[("DQ9-sus",c)]["frac"][:,j]))/2>0.05]
    comp[nm]=np.nanmedian(v)
sd_pix=np.array([np.mean([PIX[k]["frac"][:,j].std(ddof=1) for k in PIX]) for j in range(3)])
sd_prep=np.array([np.std([np.median(PIX[("DQ9",k)]["frac"][:,j])-np.median(PIX[("DQ9-sus",c)]["frac"][:,j])
                          for c,k,_,_ in PAIRS],ddof=1)/np.sqrt(2) for j in range(3)])

fig=plt.figure(figsize=(7.2,6.5))
gs=fig.add_gridspec(2,2,hspace=.60,wspace=.50,left=.175,right=.965,top=.795,bottom=.095)

# ---------- a ----------
ax=fig.add_subplot(gs[0,0]); clean(ax); PANELS.append((ax,"a"))
names=["Ink reporter\n2137 cm$^{-1}$","Analyte recovery\ntotal / ink",
       "Composition  DQ","Composition  TBZ","Composition  THI"]
vals=[ink,rec,comp["DQ"],comp["TBZ"],comp["THI"]]
cols=[FAINT,MUTE,SER["DQ"],SER["TBZ"],SER["THI"]]
y=np.arange(5)[::-1]
ax.barh(y,vals,height=.6,color=cols,linewidth=0)
ax.axvline(ink,color=FAINT,lw=.7,ls=(0,(3,3)),zorder=0)
ax.set_yticks(y); ax.set_yticklabels(names,fontsize=7.5,linespacing=1.3)
ax.set_xlabel("Preparation-to-preparation CV (%)",fontsize=8.5)
ax.set_xlim(0,28); ax.set_ylim(-.55,4.55); ax.set_xticks([0,10,20])

# ---------- b ----------
ax=fig.add_subplot(gs[0,1]); clean(ax); PANELS.append((ax,"b"))
ax.plot([0,.88],[0,.88],color=FAINT,lw=.7,ls=(0,(3,3)),zorder=1)
for j,nm in enumerate(SER):
    x=np.array([np.median(PIX[("DQ9",k)]["frac"][:,j]) for c,k,_,_ in PAIRS])
    yv=np.array([np.median(PIX[("DQ9-sus",c)]["frac"][:,j]) for c,k,_,_ in PAIRS])
    ex=np.array([PIX[("DQ9",k)]["frac"][:,j].std(ddof=1) for c,k,_,_ in PAIRS])
    ey=np.array([PIX[("DQ9-sus",c)]["frac"][:,j].std(ddof=1) for c,k,_,_ in PAIRS])
    ax.errorbar(x,yv,xerr=ex,yerr=ey,fmt="none",ecolor=SER[nm],elinewidth=.8,capsize=0,
                alpha=.45,zorder=2)
    ax.scatter(x,yv,s=20,marker=MARK[nm],facecolor=SER[nm],edgecolor=PAGE,linewidth=.7,
               zorder=3,label=nm)
ax.set_xlabel("Preparation 1,  mass fraction",fontsize=8.5)
ax.set_ylabel("Preparation 2,  mass fraction",fontsize=8.5)
ax.set_xlim(-.04,.92); ax.set_ylim(-.04,.92); ax.set_aspect("equal")
ax.set_xticks([0,.2,.4,.6,.8]); ax.set_yticks([0,.2,.4,.6,.8])
ax.legend(loc="lower left",bbox_to_anchor=(0,1.015),ncol=3,fontsize=7.6,
          handletextpad=.25,columnspacing=1.1,borderpad=0,borderaxespad=0)

# ---------- c ----------
ax=fig.add_subplot(gs[1,0]); clean(ax); PANELS.append((ax,"c"))
w=.25; x=np.arange(3)
sets=[("Within-prep. SEM",sd_pix/10,RAMP[0]),("Within-prep. SD",sd_pix,RAMP[1]),
      ("Between-prep. SD",sd_prep,RAMP[2])]
for i,(nm,v,c) in enumerate(sets):
    ax.bar(x+(i-1)*w,v,width=w-.03,color=c,linewidth=0)
ax.set_xticks(x); ax.set_xticklabels(list(SER),fontsize=8.5)
ax.set_ylabel("Error-bar magnitude\n(mass-fraction units)",fontsize=8.5)
ax.set_ylim(0,.125); ax.set_yticks([0,.04,.08,.12])
ax.legend([Line2D([0],[0],color=c,lw=5) for _,_,c in sets],[s[0] for s in sets],
          loc="lower left",bbox_to_anchor=(0,1.005),ncol=2,fontsize=7.3,
          handlelength=.9,handletextpad=.35,columnspacing=.9,labelspacing=.25,
          borderpad=0,borderaxespad=0)

# ---------- d ----------
ax=fig.add_subplot(gs[1,1]); clean(ax); PANELS.append((ax,"d"))
sd1=sd_prep.mean()
LV=[9,18,36,72]; conds=[f"DQ9-TB{a}-TH{b}" for a in LV for b in LV]
lab=lambda c:(int(c.split("-TB")[1].split("-")[0]),int(c.split("-TH")[1]))
ks9=[c for c in conds if ("DQ9",c) in D["data"]]
steps=[np.abs(U[("DQ9",a)]["frac"]-U[("DQ9",b)]["frac"]) for a,b in itertools.combinations(ks9,2)
       if abs(np.log2(lab(a)[0]/lab(b)[0]))+abs(np.log2(lab(a)[1]/lab(b)[1]))==1]
sig=np.median(np.array(steps),axis=0).mean()
ks=np.arange(1,9); dp=sig/(sd1/np.sqrt(ks))
ax.axhspan(0,2,color=FAINT,alpha=.15,zorder=0)
ax.axhline(2,color=FAINT,lw=.7,ls=(0,(3,3)),zorder=1)
ax.plot(ks,dp,"-",color=MUTE,lw=1.7,zorder=3)
ax.scatter(ks,dp,s=22,facecolor=MUTE,edgecolor=PAGE,linewidth=.7,zorder=4)
ax.scatter([1],[dp[0]],s=95,facecolor="none",edgecolor=CORAL,linewidth=1.5,zorder=5)
ax.set_xlabel("Preparations averaged, $n$",fontsize=8.5)
ax.set_ylabel("Discriminability $d'$\nfor a 2-fold concentration step",fontsize=8.5)
ax.set_xlim(.3,8.7); ax.set_ylim(0,3.5); ax.set_xticks(ks); ax.set_yticks([0,1,2,3])

for _ax,_L in PANELS:
    bb=_ax.get_position()
    fig.text(max(bb.x0-.090,.006),bb.y1+.055,_L,fontsize=11,fontweight="bold",
             va="bottom",ha="left",color=INK)
fig.text(.5,.945,"Reproducibility of the SERS-ink composition readout",
         ha="center",va="bottom",fontsize=11,fontweight="bold",color=INK)
fig.text(.5,.905,"Independent preparations of identical mixtures: dried analyte spot + 2 µL ink droplet",
         ha="center",va="bottom",fontsize=8,color=MUTE)
out=os.path.join(d,"Fig_reproducibility.pdf")
fig.savefig(out); fig.savefig(out.replace(".pdf",".png"),dpi=400)
print(out)
print(f"ink {ink:.1f}  rec {rec:.1f}  comp {dict((k,round(v,1)) for k,v in comp.items())}")
print(f"sd_pix {np.round(sd_pix,3)}  sd_prep {np.round(sd_prep,3)}  d'(1) {dp[0]:.2f}")
