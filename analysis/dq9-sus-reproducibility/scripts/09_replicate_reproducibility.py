"""재라벨된 sus를 DQ9의 액적 replicate로 보고 재현성 평가."""
import pickle, os, numpy as np, itertools
from scipy import stats

d=os.path.dirname(__file__)
D=pickle.load(open(os.path.join(d,"data.pkl"),"rb")); U=pickle.load(open(os.path.join(d,"unmix.pkl"),"rb"))
LV=[9,18,36,72]
conds=[f"DQ9-TB{tb}-TH{th}" for tb in LV for th in LV]
ks9=[c for c in conds if ("DQ9",c) in D["data"]]
sus=[c for c in conds if ("DQ9-sus",c) in D["data"]]
lab=lambda c:(int(c.split("-TB")[1].split("-")[0]), int(c.split("-TH")[1]))
# sus 원본 (a,b) → 실제 조건 (TBZ=b, THI=a)
PAIRS=[]
for c in sus:
    a,b=lab(c); k=f"DQ9-TB{b}-TH{a}"
    if ("DQ9",k) in U: PAIRS.append((c,k,b,a))
PAIRS.sort(key=lambda p:(p[2],p[3]))

# 농도 역추정기 (DQ9로 적합)
tb9=np.array([lab(c)[0] for c in ks9],float); th9=np.array([lab(c)[1] for c in ks9],float)
fit={}
for j,nm,cc in [(1,"TBZ",tb9),(2,"THI",th9)]:
    y=np.log([max(U[("DQ9",c)]["frac"][j],1e-3)/max(U[("DQ9",c)]["frac"][0],1e-3) for c in ks9])
    bb,aa=np.polyfit(np.log(cc/9.0),y,1); fit[nm]=(aa,bb)
est=lambda g,c:[9.0*np.exp((np.log(max(U[(g,c)]["frac"][j],1e-3)/max(U[(g,c)]["frac"][0],1e-3))-fit[n][0])/fit[n][1])
                for j,n in [(1,"TBZ"),(2,"THI")]]
cos=lambda a,b: float(np.clip(a,0,None)@np.clip(b,0,None)/
                      (np.linalg.norm(np.clip(a,0,None))*np.linalg.norm(np.clip(b,0,None))))

print("="*104)
print("재라벨된 DQ9-sus를 DQ9의 '두 번째 액적'으로 본 재현성  (12 조건)")
print("="*104)
print(f"{'실제조건 (TBZ/THI)':20} {'DQ9  DQ/TBZ/THI':22} {'sus  DQ/TBZ/THI':22} {'조성거리':>7} {'스펙cos':>8} {'잉크비':>7}")
rows=[]
for c,k,TBZ,THI in PAIRS:
    fa=U[("DQ9",k)]["frac"]; fb=U[("DQ9-sus",c)]["frac"]
    dist=float(np.linalg.norm(fa-fb)); cs=cos(U[("DQ9",k)]["med"],U[("DQ9-sus",c)]["med"])
    ink=U[("DQ9-sus",c)]["ink"]/U[("DQ9",k)]["ink"]
    rows.append((TBZ,THI,fa,fb,dist,cs,ink))
    print(f"  TBZ{TBZ:<3} / THI{THI:<3}      {'/'.join(f'{v:.2f}' for v in fa):22} "
          f"{'/'.join(f'{v:.2f}' for v in fb):22} {dist:7.3f} {cs:8.3f} {ink:7.2f}")

A=np.array([r[2] for r in rows]); B=np.array([r[3] for r in rows])
dists=np.array([r[4] for r in rows]); coss=np.array([r[5] for r in rows]); inks=np.array([r[6] for r in rows])

print()
print("="*104)
print("1. 성분별 액적간 재현성 (조성 분율, n=12쌍)")
print("="*104)
print(f"  {'성분':6} {'평균 편의(bias)':>15} {'RMSE':>8} {'SD_차이':>9} {'95% 일치한계(LoA)':>22} {'r':>7}")
for j,nm in enumerate(["DQ","TBZ","THI"]):
    df=B[:,j]-A[:,j]
    r=stats.pearsonr(A[:,j],B[:,j])
    print(f"  {nm:6} {df.mean():+15.3f} {np.sqrt((df**2).mean()):8.3f} {df.std(ddof=1):9.3f} "
          f"{f'{df.mean()-1.96*df.std(ddof=1):+.3f} ~ {df.mean()+1.96*df.std(ddof=1):+.3f}':>22} {r.statistic:+7.3f}")
dfall=(B-A).ravel()
print(f"  {'전체':6} {dfall.mean():+15.3f} {np.sqrt((dfall**2).mean()):8.3f} {dfall.std(ddof=1):9.3f}")
print(f"\n  조성거리: 중앙값 {np.median(dists):.3f}  평균 {dists.mean():.3f}  범위 {dists.min():.3f}–{dists.max():.3f}")
print(f"  스펙트럼 cos: 중앙값 {np.median(coss):.3f}  최소 {coss.min():.3f}")
print(f"  잉크(2137) 세기비 sus/DQ9: 중앙값 {np.median(inks):.2f}  범위 {inks.min():.2f}–{inks.max():.2f}")

print()
print("="*104)
print("2. 기준선 대비 — 이 재현성이 좋은 건가?")
print("="*104)
nn=[np.linalg.norm(U[("DQ9",a)]["frac"]-U[("DQ9",b)]["frac"])
    for a,b in itertools.combinations(ks9,2)
    if abs(np.log2(lab(a)[0]/lab(b)[0]))+abs(np.log2(lab(a)[1]/lab(b)[1]))==1]
alld=[np.linalg.norm(U[("DQ9",a)]["frac"]-U[("DQ9",b)]["frac"]) for a,b in itertools.combinations(ks9,2)]
print(f"  같은 조건 다른 액적 (DQ9 vs sus)          중앙값 {np.median(dists):.3f}")
print(f"  DQ9 내부 '한 칸(2배) 이웃 조건'끼리        중앙값 {np.median(nn):.3f}   ← 이보다 작아야 조건 구분 가능")
print(f"  DQ9 내부 아무 두 조건끼리                 중앙값 {np.median(alld):.3f}")
print(f"  → 신호대잡음 (조건간/액적간) = {np.median(alld)/np.median(dists):.2f}배,  "
      f"한 칸 대비 = {np.median(nn)/np.median(dists):.2f}배")

print()
print("="*104)
print("3. 농도 역추정 재현성 — 같은 조건을 두 액적에서 추정했을 때 몇 배 어긋나나")
print("="*104)
print(f"  {'실제조건':16} {'DQ9 추정 TBZ/THI':22} {'sus 추정 TBZ/THI':22} {'배수오차 TBZ':>12} {'THI':>8}")
fe=[]
for c,k,TBZ,THI in PAIRS:
    ea=est("DQ9",k); eb=est("DQ9-sus",c)
    f1,f2=eb[0]/ea[0], eb[1]/ea[1]; fe.append((f1,f2))
    print(f"  TBZ{TBZ:<3}/THI{THI:<4}      {f'{ea[0]:.1f} / {ea[1]:.1f}':22} {f'{eb[0]:.1f} / {eb[1]:.1f}':22} "
          f"{f1:12.2f} {f2:8.2f}")
FE=np.array(fe)
for j,nm in enumerate(["TBZ","THI"]):
    lg=np.log2(FE[:,j])
    print(f"\n  {nm}: 기하평균 배수오차 {2**lg.mean():.2f}배,  기하SD {2**lg.std(ddof=1):.2f}배,  "
          f"중앙 절대배수오차 {2**np.median(np.abs(lg)):.2f}배")

print()
print("="*104)
print("4. 조건 회수 검정 — 액적을 바꿔도 맞는 조건으로 되찾아지나 (최근접 이웃)")
print("="*104)
hit=0
for c,k,TBZ,THI in PAIRS:
    fb=U[("DQ9-sus",c)]["frac"]
    dd={kk:float(np.linalg.norm(U[("DQ9",kk)]["frac"]-fb)) for kk in ks9}
    order=sorted(dd,key=dd.get); rank=order.index(k)+1; hit+=(rank==1)
    print(f"  TBZ{TBZ:<3}/THI{THI:<4}  최근접={order[0]:16} 정답순위={rank:2}/16  {'OK' if rank==1 else ''}")
print(f"\n  1순위 적중 {hit}/12,  평균 순위 {np.mean([sorted({kk:float(np.linalg.norm(U[('DQ9',kk)]['frac']-U[('DQ9-sus',c)]['frac'])) for kk in ks9}.items(),key=lambda x:x[1]).index(next(t for t in sorted({kk:float(np.linalg.norm(U[('DQ9',kk)]['frac']-U[('DQ9-sus',c)]['frac'])) for kk in ks9}.items(),key=lambda x:x[1]) if t[0]==k))+1 for c,k,_,_ in PAIRS]):.1f}/16")
pickle.dump(dict(PAIRS=PAIRS,A=A,B=B,dists=dists,coss=coss,inks=inks,FE=FE,nn=nn,alld=alld),
            open(os.path.join(d,"repl.pkl"),"wb"))
