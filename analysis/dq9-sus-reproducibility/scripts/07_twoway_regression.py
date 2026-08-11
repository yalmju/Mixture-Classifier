"""2원 회귀로 판별:
   log(추정농도) = c0 + a*log(TB라벨) + b*log(TH라벨)
   정상  : TBZ(a=1,b=0)  THI(a=0,b=1)
   ② swap: TBZ(a=0,b=1)  THI(a=1,b=0)
   ③ 사용자: TBZ(a=0,b=1)  THI(a=0,b=1)      <- ②와 THI 행에서만 갈림
"""
import pickle, os, numpy as np, itertools
from scipy import stats

d = os.path.dirname(__file__)
D = pickle.load(open(os.path.join(d, "data.pkl"), "rb"))
U = pickle.load(open(os.path.join(d, "unmix.pkl"), "rb"))
LV = [9, 18, 36, 72]
conds = [f"DQ9-TB{tb}-TH{th}" for tb in LV for th in LV]
ks9 = [c for c in conds if ("DQ9", c) in D["data"]]
shared = [c for c in conds if ("DQ9-sus", c) in D["data"]]
lab = lambda c: (int(c.split("-TB")[1].split("-")[0]), int(c.split("-TH")[1]))

tb9 = np.array([lab(c)[0] for c in ks9], float); th9 = np.array([lab(c)[1] for c in ks9], float)
fit = {}
for j, nm, cc in [(1,"TBZ",tb9), (2,"THI",th9)]:
    y = np.log([max(U[("DQ9",c)]["frac"][j],1e-3)/max(U[("DQ9",c)]["frac"][0],1e-3) for c in ks9])
    b, a = np.polyfit(np.log(cc/9.0), y, 1); fit[nm] = (a, b)
def est(grp, c):
    f = U[(grp,c)]["frac"]
    return [9.0*np.exp((np.log(max(f[j],1e-3)/max(f[0],1e-3))-fit[nm][0])/fit[nm][1])
            for j, nm in [(1,"TBZ"),(2,"THI")]]

def ols(y, X):
    Xd = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    r = y - Xd@beta
    dof = len(y) - Xd.shape[1]
    s2 = r@r/dof
    cov = s2*np.linalg.inv(Xd.T@Xd)
    se = np.sqrt(np.diag(cov))
    t = stats.t.ppf(0.975, dof)
    return beta, se, t*se, dof

print("="*96)
print("2원 회귀:  log(추정농도) = c0 + a·log2(TB라벨) + b·log2(TH라벨)")
print("="*96)
for grp, ks in [("DQ9  (정상 대조군)", ks9), ("DQ9-sus", shared)]:
    kk = ks9 if "정상" in grp else shared
    x1 = np.log2(np.array([lab(c)[0] for c in kk], float)/9)
    x2 = np.log2(np.array([lab(c)[1] for c in kk], float)/9)
    E = np.array([est("DQ9" if "정상" in grp else "DQ9-sus", c) for c in kk])
    print(f"\n  {grp}   (n={len(kk)})")
    for j, nm in [(0,"추정 TBZ"), (1,"추정 THI")]:
        y = np.log2(E[:, j])
        beta, se, ci, dof = ols(y, np.column_stack([x1, x2]))
        print(f"     {nm}:  a(TB라벨)={beta[1]:+.2f} ± {ci[1]:.2f}    "
              f"b(TH라벨)={beta[2]:+.2f} ± {ci[2]:.2f}     (95% CI, dof={dof})")

print()
print("="*96)
print("판별 요약 — sus의 '추정 THI' 행이 ②와 ③를 가른다")
print("   ② swap 이면  a≈1, b≈0   /   ③ 사용자 가설이면  a≈0, b≈1")
print("="*96)
x1 = np.log2(np.array([lab(c)[0] for c in shared], float)/9)
x2 = np.log2(np.array([lab(c)[1] for c in shared], float)/9)
E = np.array([est("DQ9-sus", c) for c in shared])
for j, nm in [(0,"TBZ"), (1,"THI")]:
    y = np.log2(E[:, j])
    beta, se, ci, dof = ols(y, np.column_stack([x1, x2]))
    for k, who in [(1,"a (TB라벨)"), (2,"b (TH라벨)")]:
        t0 = (beta[k]-0)/se[k]; t1 = (beta[k]-1)/se[k]
        print(f"  {nm} {who:12}  추정={beta[k]:+.2f}   "
              f"H0(계수=0) p={2*(1-stats.t.cdf(abs(t0),dof)):.3f}   "
              f"H0(계수=1) p={2*(1-stats.t.cdf(abs(t1),dof)):.3f}")

print()
print("="*96)
print("보조 검정: 각 TH열 안에서 TB라벨이 THI를 움직이는가 (③이면 안 움직여야 함)")
print("="*96)
for v in LV:
    g = [(lab(c)[0], est("DQ9-sus", c)) for c in shared if lab(c)[1] == v]
    if len(g) < 3: continue
    g.sort()
    print(f"  TH={v:<3} TB라벨 {[x[0] for x in g]}  →  추정THI {[round(x[1][1],1) for x in g]}"
          f"   추정TBZ {[round(x[1][0],1) for x in g]}")
allp = [(lab(c)[0], lab(c)[1], *est("DQ9-sus", c)) for c in shared]
rho = stats.spearmanr([a for a,_,_,_ in allp], [t for _,_,_,t in allp])
print(f"\n  TH열 내부 순위 일관성 (TB라벨 vs 추정THI, 전체): rho={rho.statistic:+.3f} p={rho.pvalue:.3f}")
