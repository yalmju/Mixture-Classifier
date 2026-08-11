"""같은 조건인데 액적이 다르면 100픽셀 분포가 얼마나 어긋나나 — n=100의 함정."""
import pickle,os,numpy as np
from scipy import stats
d=os.path.dirname(__file__)
PIX=pickle.load(open(os.path.join(d,"allpix.pkl"),"rb")); D=pickle.load(open(os.path.join(d,"data.pkl"),"rb"))
LV=[9,18,36,72]; conds=[f"DQ9-TB{a}-TH{b}" for a in LV for b in LV]
lab=lambda c:(int(c.split("-TB")[1].split("-")[0]),int(c.split("-TH")[1]))
sus=[c for c in conds if ("DQ9-sus",c) in D["data"]]
PAIRS=[(c,f"DQ9-TB{lab(c)[1]}-TH{lab(c)[0]}",lab(c)[1],lab(c)[0]) for c in sus]
PAIRS=[p for p in PAIRS if ("DQ9",p[1]) in D["data"]]
PAIRS.sort(key=lambda p:(p[2],p[3]))
NM=["DQ","TBZ","THI"]

def ovl(a,b):
    """두 분포의 겹침 계수 (히스토그램 교집합, 0=완전분리 1=완전일치)"""
    lo,hi=min(a.min(),b.min()),max(a.max(),b.max())
    if hi<=lo: return 1.0
    bins=np.linspace(lo,hi,41)
    ha,_=np.histogram(a,bins,density=True); hb,_=np.histogram(b,bins,density=True)
    return float(np.sum(np.minimum(ha,hb))*(bins[1]-bins[0]))

print("="*106)
print("같은 조건 · 다른 액적 · 각 n=100픽셀 —  분포가 겹치나?")
print("="*106)
print(f"  {'조건 (TBZ/THI)':16} {'성분':5} {'액적1  평균±SD':>18} {'액적2  평균±SD':>18} "
      f"{'차이/SD(Cohen d)':>17} {'겹침':>7} {'t-test p':>10}")
rows=[]
for c,k,TBZ,THI in PAIRS:
    A=PIX[("DQ9",k)]["frac"]; B=PIX[("DQ9-sus",c)]["frac"]
    for j,nm in enumerate(NM):
        a,b=A[:,j],B[:,j]
        sp=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
        dd=(b.mean()-a.mean())/max(sp,1e-9)
        o=ovl(a,b); p=stats.ttest_ind(a,b,equal_var=False).pvalue
        rows.append((TBZ,THI,nm,a.mean(),a.std(ddof=1),b.mean(),b.std(ddof=1),dd,o,p))
        if nm=="TBZ" or True:
            print(f"  TBZ{TBZ:<3}/THI{THI:<4}  {nm:5} {f'{a.mean():.3f} ± {a.std(ddof=1):.3f}':>18} "
                  f"{f'{b.mean():.3f} ± {b.std(ddof=1):.3f}':>18} {dd:17.1f} {o:7.2f} {p:10.2e}")
    print()

DD=np.array([abs(r[7]) for r in rows]); OO=np.array([r[8] for r in rows]); PP=np.array([r[9] for r in rows])
print("="*106)
print("요약 (36 = 12조건 × 3성분)")
print("="*106)
print(f"  |Cohen d| : 중앙값 {np.median(DD):.1f}   범위 {DD.min():.1f}–{DD.max():.1f}")
print(f"  분포 겹침  : 중앙값 {np.median(OO):.2f}   0.5 미만인 경우 {int((OO<0.5).sum())}/36")
print(f"  t-검정에서 p<0.05 (= '두 액적이 유의하게 다르다'): {int((PP<0.05).sum())}/36")
print()
print("  ※ 같은 조건이므로 '차이 없음'이 정답인데, n=100으로 t-test하면 거의 전부 유의하게 나온다.")
print("     100픽셀은 독립 반복이 아니라 한 액적 안의 하위표본이기 때문 (pseudoreplication).")

print()
print("="*106)
print("오차막대를 무엇으로 그릴 것인가 — 같은 데이터, 세 가지 표기")
print("="*106)
sd_pix=np.mean([[PIX[k]["frac"][:,j].std(ddof=1) for j in range(3)] for k in PIX],axis=0)
drop_sd=np.array([0.082,0.087,0.105])
print(f"  {'표기':38} {'DQ':>8} {'TBZ':>8} {'THI':>8}   해석")
print(f"  {'(a) 픽셀 SD (n=100)':38} {sd_pix[0]:8.3f} {sd_pix[1]:8.3f} {sd_pix[2]:8.3f}   맵 안 불균일성 — 보여줄 값")
print(f"  {'(b) 픽셀 SEM (SD/√100)':38} {sd_pix[0]/10:8.3f} {sd_pix[1]/10:8.3f} {sd_pix[2]/10:8.3f}   "
      f"← 절대 쓰지 말 것 (10~20배 과소)")
print(f"  {'(c) 액적간 SD (실측, n=12쌍)':38} {drop_sd[0]:8.3f} {drop_sd[1]:8.3f} {drop_sd[2]:8.3f}   "
      f"재측정하면 이만큼 움직임 — 진짜 오차")
print()
print(f"  (c)/(b) = {np.mean(drop_sd/(sd_pix/10)):.0f}배.  SEM으로 그리면 오차막대가 {np.mean(drop_sd/(sd_pix/10)):.0f}배 작게 나옴.")
pickle.dump(dict(rows=rows,PAIRS=PAIRS),open(os.path.join(d,"n100.pkl"),"wb"))
