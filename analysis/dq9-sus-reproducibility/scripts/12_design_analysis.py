"""추가 측정 설계: 빠진 THI=9 줄 4맵 vs 같은 조건 액적 반복 — 어느 쪽이 이득인가"""
import pickle, os, numpy as np, itertools
from scipy import stats

d=os.path.dirname(__file__)
D=pickle.load(open(os.path.join(d,"data.pkl"),"rb")); U=pickle.load(open(os.path.join(d,"unmix.pkl"),"rb"))
P=pickle.load(open(os.path.join(d,"repl.pkl"),"rb"))
A,B,dists=P["A"],P["B"],P["dists"]; PAIRS=P["PAIRS"]
LV=[9,18,36,72]; conds=[f"DQ9-TB{tb}-TH{th}" for tb in LV for th in LV]
ks9=[c for c in conds if ("DQ9",c) in D["data"]]
lab=lambda c:(int(c.split("-TB")[1].split("-")[0]), int(c.split("-TH")[1]))
rng=np.random.default_rng(0)

print("="*94)
print("1. 현재 재현성 추정의 정밀도 (n=12쌍, 부트스트랩 2000회)")
print("="*94)
df=(B-A)
def rmse(ix): return float(np.sqrt((df[ix]**2).mean()))
boot=[rmse(rng.integers(0,12,12)) for _ in range(2000)]
lo,hi=np.percentile(boot,[2.5,97.5])
print(f"  조성 RMSE = {rmse(np.arange(12)):.3f}   95% CI [{lo:.3f}, {hi:.3f}]   (폭 {hi-lo:.3f})")
for n in [12,16,24,36,48]:
    # 같은 잡음에서 n쌍일 때 기대 CI 폭 (RMSE의 상대 SE ≈ 1/sqrt(2n) * ...)
    bb=[float(np.sqrt((df[rng.integers(0,12,n)]**2).mean())) for _ in range(2000)]
    l,h=np.percentile(bb,[2.5,97.5])
    print(f"     n={n:2}쌍 → 예상 95% CI 폭 {h-l:.3f}"+("   ← 빠진 줄 4맵 추가시" if n==16 else ""))

print()
print("="*94)
print("2. 잔차가 조건에 따라 달라지나 (THI=9 줄이 특별히 위험한가)")
print("="*94)
TBZ=np.array([p[2] for p in PAIRS],float); THI=np.array([p[3] for p in PAIRS],float)
for nm,x in [("TBZ 수준",TBZ),("THI 수준",THI),("DQ분율(평균)", (A[:,0]+B[:,0])/2)]:
    r=stats.spearmanr(x,dists)
    print(f"  조성거리 ~ {nm:14} rho={r.statistic:+.3f}  p={r.pvalue:.3f}")
print("  THI 수준별 조성거리 중앙값:", {int(v): round(float(np.median(dists[THI==v])),3) for v in sorted(set(THI))})
print("  → THI=9는 아직 replicate가 없어 외삽 구간. 위 추세로 위험도를 미리 알 수는 없음")

print()
print("="*94)
print("3. 핵심 질문: 조건 하나를 가르려면 액적 몇 개를 평균해야 하나")
print("="*94)
# 액적 1개의 성분별 SD (두 액적 차이의 SD / sqrt2)
sd1=df.std(ddof=1,axis=0)/np.sqrt(2)
print(f"  액적 1개 조성분율 SD:  DQ {sd1[0]:.3f}   TBZ {sd1[1]:.3f}   THI {sd1[2]:.3f}   (평균 {sd1.mean():.3f})")
# 한 칸(2배) 이동시 성분별 분율 변화량
steps=[]
for a,b in itertools.combinations(ks9,2):
    if abs(np.log2(lab(a)[0]/lab(b)[0]))+abs(np.log2(lab(a)[1]/lab(b)[1]))==1:
        steps.append(np.abs(U[("DQ9",a)]["frac"]-U[("DQ9",b)]["frac"]))
steps=np.array(steps); sig=np.median(steps,axis=0)
print(f"  한 칸(2배) 이동시 분율 변화(중앙값): DQ {sig[0]:.3f}   TBZ {sig[1]:.3f}   THI {sig[2]:.3f}")
print()
print(f"  {'평균할 액적 수 k':16} {'유효 SE(평균성분)':>18} {'판별력 d′ = 신호/SE':>22}")
for k in [1,2,3,4,6,8]:
    se=sd1.mean()/np.sqrt(k); dprime=sig.mean()/se
    mark=""
    if dprime>=2 and (sig.mean()/(sd1.mean()/np.sqrt(k-1)) < 2 if k>1 else False): mark="  ← d′=2 도달"
    print(f"  {k:^16} {se:18.3f} {dprime:22.2f}{mark}")
print("\n  (d′ 2 이상이면 인접 조건을 어느 정도 가름, 3 이상이면 안정적)")

print()
print("="*94)
print("4. 두 설계 비교 — 같은 4맵을 어디에 쓰는 게 이득인가")
print("="*94)
print("  (a) 빠진 THI=9 줄 4맵:")
print(f"      · replicate 12 → 16쌍, 재현성 CI 폭 {hi-lo:.3f} → 위 표 참조")
print( "      · 4×4 격자 완성 (외삽 구간이던 THI=9 replicate 확보)")
print( "      · 조건당 액적은 여전히 2개 → 단일 조건 판별력은 그대로")
print("  (b) 기존 조건 중 4곳을 골라 액적 2개씩 추가 (조건당 총 4액적):")
d1=sd1.mean(); print(f"      · 그 4조건에서 유효 SE {d1:.3f} → {d1/2:.3f} (절반),  d′ {sig.mean()/d1:.2f} → {sig.mean()/(d1/2):.2f}")
print( "      · 재현성 추정 자유도도 조건당 3쌍씩 확보돼 훨씬 단단해짐")
print( "      · 대신 격자는 미완성으로 남음")
