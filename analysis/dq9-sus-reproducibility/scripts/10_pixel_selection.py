"""맵 안 100포인트 중 어떤 부분집합을 쓰면 액적간 일치가 좋아지나.
   목적함수 = d' = (조건간 신호) / (같은조건 액적간 잡음).  잡음만 줄고 신호도 줄면 소용없음."""
import pickle, os, numpy as np, itertools
from scipy.optimize import nnls

d=os.path.dirname(__file__)
D=pickle.load(open(os.path.join(d,"data.pkl"),"rb"))
R,wnc,wn0,data=D["R"],D["wnc"],D["wn0"],D["data"]
LV=[9,18,36,72]; conds=[f"DQ9-TB{tb}-TH{th}" for tb in LV for th in LV]
lab=lambda c:(int(c.split("-TB")[1].split("-")[0]), int(c.split("-TH")[1]))
ks9=[c for c in conds if ("DQ9",c) in data]
sus=[c for c in conds if ("DQ9-sus",c) in data]
PAIRS=[(c,f"DQ9-TB{lab(c)[1]}-TH{lab(c)[0]}") for c in sus]
PAIRS=[(a,b) for a,b in PAIRS if ("DQ9",b) in data]

CACHE=os.path.join(d,"allpix.pkl")
if os.path.exists(CACHE):
    PIX=pickle.load(open(CACHE,"rb"))
else:
    PIX={}
    for k,v in data.items():
        X=v["X"]; C=np.empty((len(X),4)); r2=np.empty(len(X))
        for i,x in enumerate(X):
            xc=np.clip(x,0,None); c,_=nnls(R,xc); C[i]=c
            ss=np.sum((xc-xc.mean())**2)
            r2[i]=1-np.sum((R@c-xc)**2)/max(ss,1e-12)
        F=C[:,:3]; s=F.sum(1,keepdims=True); s[s==0]=1
        PIX[k]=dict(frac=F/s, r2=r2, sig=np.trapezoid(np.clip(X,0,None),wnc,axis=1),
                    ink=v["ink"], raw=C)
        print("pix",k,flush=True)
    pickle.dump(PIX,open(CACHE,"wb"))

def compose(k, rule):
    """rule(P) -> boolean mask. 맵 대표 조성 = 선택 픽셀의 중앙값."""
    P=PIX[k]; m=rule(P)
    if m.sum()<5: m=np.ones(len(P["r2"]),bool)
    return np.median(P["frac"][m],axis=0), int(m.sum())

def evaluate(rule):
    # 잡음: 12 replicate 쌍의 성분별 차이
    df=[]; ns=[]
    for a,b in PAIRS:
        fa,n1=compose(("DQ9",b),rule); fb,n2=compose(("DQ9-sus",a),rule)
        df.append(fb-fa); ns += [n1,n2]
    df=np.array(df); noise=df.std(ddof=1,axis=0).mean()/np.sqrt(2)   # 액적 1개 SD
    # 신호: DQ9 안에서 한 칸(2배) 이웃 조건 간 분율 변화
    st=[]
    for x,y in itertools.combinations(ks9,2):
        if abs(np.log2(lab(x)[0]/lab(y)[0]))+abs(np.log2(lab(x)[1]/lab(y)[1]))==1:
            fx,_=compose(("DQ9",x),rule); fy,_=compose(("DQ9",y),rule)
            st.append(np.abs(fx-fy))
    sig=np.median(np.array(st),axis=0).mean()
    rmse=float(np.sqrt((df**2).mean()))
    return dict(dprime=sig/noise, noise=noise, sig=sig, rmse=rmse, npix=np.mean(ns))

def pct(P,key,keep,high=True):
    v=P[key]; q=np.percentile(v,100-keep if high else keep)
    return v>=q if high else v<=q

RULES={}
RULES["전체 100점 (선택 없음)"]      = lambda P: np.ones(len(P["r2"]),bool)
RULES["현재 방식: R² 상위 80%"]      = lambda P: pct(P,"r2",80)
for kp in [70,50,30,20,10]:
    RULES[f"재구성 R² 상위 {kp}%"]    = (lambda kp: (lambda P: pct(P,"r2",kp)))(kp)
for kp in [50,30,20]:
    RULES[f"총신호 상위 {kp}%"]       = (lambda kp: (lambda P: pct(P,"sig",kp)))(kp)
for kp in [50,30,20]:
    RULES[f"R²·신호 둘 다 상위 {kp}%"] = (lambda kp: (lambda P: pct(P,"r2",kp)&pct(P,"sig",kp)))(kp)
RULES["R² 상위50% ∩ 신호 중간50%"]   = lambda P: pct(P,"r2",50)&(P["sig"]>=np.percentile(P["sig"],25))&(P["sig"]<=np.percentile(P["sig"],75))
def consensus(P,keep=50):
    med=np.median(P["frac"],axis=0)
    dd=np.linalg.norm(P["frac"]-med,axis=1)
    return dd<=np.percentile(dd,keep)
for kp in [70,50,30]:
    RULES[f"맵 중앙조성에 가까운 {kp}%"]=(lambda kp: (lambda P: consensus(P,kp)))(kp)
RULES["R²상위50% ∩ 중앙조성근접50%"] = lambda P: pct(P,"r2",50)&consensus(P,50)

print("="*104)
print("맵 안 픽셀 선택 규칙별 성능  (신호=한 칸 이동시 분율변화, 잡음=액적1개 SD)")
print("="*104)
print(f"  {'규칙':32} {'평균픽셀수':>9} {'신호':>7} {'잡음':>7} {'d′':>7} {'replicate RMSE':>15}")
res={}
for n,r in RULES.items():
    e=evaluate(r); res[n]=e
    print(f"  {n:32} {e['npix']:9.0f} {e['sig']:7.3f} {e['noise']:7.3f} {e['dprime']:7.2f} {e['rmse']:15.3f}")

print()
best=max(res,key=lambda n:res[n]["dprime"])
base=res["현재 방식: R² 상위 80%"]
print(f"  최고 d′: 「{best}」  d′={res[best]['dprime']:.2f}  (현재 방식 {base['dprime']:.2f} 대비 "
      f"{res[best]['dprime']/base['dprime']:.2f}배)")
print(f"  → 이 d′를 액적 수로 환산하면 현재 방식 액적 {(res[best]['dprime']/base['dprime'])**2:.1f}개 평균과 동등")

print()
print("="*104)
print("맵 내부 이질성 자체 — 100포인트가 실제로 얼마나 다른가")
print("="*104)
for k in [("DQ9","DQ9-TB36-TH36"),("DQ9-sus","DQ9-TB36-TH36"),("DQ9","DQ9-TB9-TH72")]:
    P=PIX[k]
    f=P["frac"]
    print(f"  {k[0]:9}{k[1]:16} 픽셀간 분율 SD: DQ {f[:,0].std():.3f} TBZ {f[:,1].std():.3f} THI {f[:,2].std():.3f}"
          f"   R² 범위 {P['r2'].min():.2f}~{P['r2'].max():.2f}   신호 최대/최소 {P['sig'].max()/max(P['sig'].min(),1e-9):.0f}배")
allsd=np.mean([PIX[k]["frac"].std(axis=0) for k in PIX],axis=0)
print(f"\n  전 맵 평균 픽셀간 SD: DQ {allsd[0]:.3f}  TBZ {allsd[1]:.3f}  THI {allsd[2]:.3f}")
print(f"  (비교) 액적간 SD ≈ 0.091  → 픽셀간 산포가 액적간과 비슷하거나 더 큼")
