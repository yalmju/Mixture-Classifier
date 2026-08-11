"""②(파일명 두 축이 서로 바뀜) vs ③(TB축이 죽고 THI는 TH를 따라감) 정면 비교.
   + sus와 동일한 설계(TB 18/36/72, 12맵)로 잘라낸 DQ9를 '공정한 기준선'으로 사용."""
import pickle, os, numpy as np, itertools
from scipy import stats

d = os.path.dirname(__file__)
D = pickle.load(open(os.path.join(d, "data.pkl"), "rb"))
U = pickle.load(open(os.path.join(d, "unmix.pkl"), "rb"))
LV = [9,18,36,72]
conds = [f"DQ9-TB{tb}-TH{th}" for tb in LV for th in LV]
ks9  = [c for c in conds if ("DQ9",c) in D["data"]]
sus  = [c for c in conds if ("DQ9-sus",c) in D["data"]]
lab  = lambda c: (int(c.split("-TB")[1].split("-")[0]), int(c.split("-TH")[1]))

tb9 = np.array([lab(c)[0] for c in ks9],float); th9 = np.array([lab(c)[1] for c in ks9],float)
fit={}
for j,nm,cc in [(1,"TBZ",tb9),(2,"THI",th9)]:
    y=np.log([max(U[("DQ9",c)]["frac"][j],1e-3)/max(U[("DQ9",c)]["frac"][0],1e-3) for c in ks9])
    b,a=np.polyfit(np.log(cc/9.0),y,1); fit[nm]=(a,b)
est=lambda g,c:[9.0*np.exp((np.log(max(U[(g,c)]["frac"][j],1e-3)/max(U[(g,c)]["frac"][0],1e-3))-fit[n][0])/fit[n][1])
                for j,n in [(1,"TBZ"),(2,"THI")]]

def ols(y,X):
    Xd=np.column_stack([np.ones(len(y)),X]); beta,*_=np.linalg.lstsq(Xd,y,rcond=None)
    r=y-Xd@beta; dof=len(y)-Xd.shape[1]; s2=r@r/dof
    se=np.sqrt(np.diag(s2*np.linalg.inv(Xd.T@Xd)))
    return beta,se,dof,r@r

def coefs(grp,ks):
    x1=np.log2(np.array([lab(c)[0] for c in ks],float)/9)
    x2=np.log2(np.array([lab(c)[1] for c in ks],float)/9)
    E=np.array([est(grp,c) for c in ks]); out={}
    for j,nm in [(0,"TBZ"),(1,"THI")]:
        b,se,dof,_=ols(np.log2(E[:,j]),np.column_stack([x1,x2]))
        out[nm]=(b[1],b[2],se[1],se[2],dof)
    return out

# sus와 같은 설계로 자른 DQ9 (TB9 행 제거) — 계수 감쇠의 공정한 기준선
ks9_m = [c for c in ks9 if lab(c)[0] != 9]

print("="*100)
print("1. 공정한 기준선: sus와 동일 설계(TB 18/36/72, 12맵)로 자른 DQ9에서도 계수가 1에 가까운가?")
print("="*100)
for tag, grp, ks in [("DQ9 전체 (16맵)","DQ9",ks9), ("DQ9 잘라낸 (12맵, sus와 동일설계)","DQ9",ks9_m),
                     ("DQ9-sus (12맵)","DQ9-sus",sus)]:
    c = coefs(grp,ks)
    print(f"  {tag:34}")
    for nm in ["TBZ","THI"]:
        a,b,sa,sb,dof = c[nm]
        t=stats.t.ppf(0.975,dof)
        print(f"      추정{nm}:  a(TB라벨)={a:+.2f}±{t*sa:.2f}   b(TH라벨)={b:+.2f}±{t*sb:.2f}")

print()
print("="*100)
print("2. 네 계수를 각 가설의 예측치와 대조 — 어느 가설이 '한 개도 기각되지 않는가'")
print("="*100)
c = coefs("DQ9-sus",sus)
HYP = {
 "① 라벨 그대로":                    {"TBZ":(1,0), "THI":(0,1)},
 "② 파일명 두 축이 서로 바뀜(swap)":   {"TBZ":(0,1), "THI":(1,0)},
 "③ TB축 死 + THI는 TH를 따라감":     {"TBZ":(0,1), "THI":(0,1)},
}
for hn, pred in HYP.items():
    bad = []
    print(f"  {hn}")
    for nm in ["TBZ","THI"]:
        a,b,sa,sb,dof = c[nm]
        for val,obs,se,src in [(pred[nm][0],a,sa,"a(TB라벨)"), (pred[nm][1],b,sb,"b(TH라벨)")]:
            t=(obs-val)/se; p=2*(1-stats.t.cdf(abs(t),dof))
            flag = "기각" if p<0.05 else "  OK"
            if p<0.05: bad.append(f"{nm}·{src}")
            print(f"      {nm} {src:10} 예측={val}  관측={obs:+.2f}  p={p:.3f}  {flag}")
    print(f"      → 기각된 계수 {len(bad)}/4  {bad if bad else ''}\n")

print("="*100)
print("3. 잔차 기반 모형 비교 (각 가설의 계수를 '고정'했을 때 남는 오차, 낮을수록 좋음)")
print("="*100)
x1=np.log2(np.array([lab(cc)[0] for cc in sus],float)/9)
x2=np.log2(np.array([lab(cc)[1] for cc in sus],float)/9)
E=np.array([est("DQ9-sus",cc) for cc in sus])
for hn,pred in HYP.items():
    tot=0
    for j,nm in [(0,"TBZ"),(1,"THI")]:
        y=np.log2(E[:,j]); pa,pb=pred[nm]
        resid=y-(pa*x1+pb*x2)
        tot+=np.var(resid-resid.mean())          # 절편은 자유롭게 둠
    print(f"  {hn:32} 잔차분산 합={tot:.3f}")

print()
print("="*100)
print("4. '파일명 축 바꿔달기'로 sus를 되살릴 수 있는가 — 재라벨 후 DQ9와 직접 대조")
print("   재라벨 규칙(②): 파일명 TB{a}-TH{b}  →  실제 TBZ={b}, THI={a}")
print("="*100)
print(f"  {'원본 파일명':17} {'재라벨 후 실제조건':20} {'sus DQ/TBZ/THI':22} {'DQ9 DQ/TBZ/THI':22} {'거리':>6}")
ds=[]
for cc in sus:
    a,b = lab(cc)
    k=("DQ9", f"DQ9-TB{b}-TH{a}")
    if k not in U:
        print(f"  {cc:17} TBZ={b},THI={a} → DQ9에 해당 맵 없음"); continue
    fb,fa = U[("DQ9-sus",cc)]["frac"], U[k]["frac"]
    v=float(np.linalg.norm(fa-fb)); ds.append(v)
    print(f"  {cc:17} {'TBZ='+str(b)+', THI='+str(a):20} {'/'.join(f'{x:.2f}' for x in fb):22} "
          f"{'/'.join(f'{x:.2f}' for x in fa):22} {v:6.3f}")
print(f"\n  재라벨 후 평균거리 = {np.mean(ds):.3f}")
# 기준: DQ9 내부에서 '같은 조건은 없으니' 액적 재현성 대용 = 가장 가까운 이웃 조건 간 거리
nn=[]
for a,b in itertools.combinations(ks9,2):
    if abs(np.log2(lab(a)[0]/lab(b)[0]))+abs(np.log2(lab(a)[1]/lab(b)[1]))==1:   # 한 단계 이웃
        nn.append(np.linalg.norm(U[("DQ9",a)]["frac"]-U[("DQ9",b)]["frac"]))
print(f"  참고: DQ9 안에서 '한 단계 이웃 조건'끼리의 거리 중앙값 = {np.median(nn):.3f}")
print(f"        (재라벨 후 잔차가 이 수준이면 사실상 일치로 볼 수 있음)")
