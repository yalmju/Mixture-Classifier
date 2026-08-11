"""(a) 액적 반복 잡음 하한 추정  (b) 'TB:TH 항상 동일' 가설 검정"""
import pickle, os, numpy as np, glob, itertools
from scipy.optimize import nnls
from scipy import stats

d = os.path.dirname(__file__)
import importlib.util
spec = importlib.util.spec_from_file_location("cmp", os.path.join(d, "cmp.py"))

D = pickle.load(open(os.path.join(d, "data.pkl"), "rb"))
U = pickle.load(open(os.path.join(d, "unmix.pkl"), "rb"))
R, wn0, wnc = D["R"], D["wn0"], D["wnc"]
LV = [9, 18, 36, 72]
conds = [f"DQ9-TB{tb}-TH{th}" for tb in LV for th in LV]
shared = [c for c in conds if ("DQ9-sus", c) in D["data"]]
REF = "/Users/seungki2/Documents/GitHub/Mixture-Classifier/.claude/worktrees/ink-check-6maps-run-ad8cec/Reference"

# ---------- (a) 같은 용액을 다른 액적으로 3번 찍은 순수 표준으로 '액적 잡음' 추정 ----------
def load(path):
    lines = open(path).read().strip().split("\n")
    hdr = [v for v in lines[2].split(",") if v.strip()]
    wn = np.array([float(v) for v in hdr[2:]]); n = len(wn)
    A = np.array([[float(v) for v in r.split(",")[2:2+n]] for r in lines[3:] if r.strip()], float)
    return wn, A

def als(y, lam=1e5, p=0.01, n=10):
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    L = len(y)
    Dm = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2)); DD = lam*Dm.dot(Dm.T)
    w = np.ones(L); W = sparse.spdiags(w,0,L,L); z = y
    for _ in range(n):
        W.setdiag(w); z = spsolve((W+DD).tocsc(), w*y); w = p*(y>z)+(1-p)*(y<z)
    return z

CACHE = os.path.join(d, "refrep.pkl")
if os.path.exists(CACHE):
    REP = pickle.load(open(CACHE, "rb"))
else:
    mask = (wn0 >= 400) & (wn0 <= 1800)
    REP = {}
    for name in ["DQ", "TBZ", "THI"]:
        for f in sorted(glob.glob(os.path.join(REF, f"{name}-*.csv"))):
            w, A = load(f)
            X = np.array([(y - als(y))[mask] for y in A])
            C = np.array([nnls(R, np.clip(x,0,None))[0] for x in X])
            res = np.array([np.linalg.norm(R@c-np.clip(x,0,None))/max(np.linalg.norm(np.clip(x,0,None)),1e-9)
                            for c,x in zip(C,X)])
            good = res < np.percentile(res, 80)
            A3 = C[good][:, :3]; s = A3.sum(1, keepdims=True); s[s==0]=1
            REP[os.path.basename(f)] = np.median(A3/s, axis=0)
            print("ref", os.path.basename(f), np.round(REP[os.path.basename(f)],3), flush=True)
    pickle.dump(REP, open(CACHE, "wb"))

print("="*92)
print("A. 액적 반복성 하한 — 같은 순수 용액을 다른 액적으로 3회 (조성분율 L2 거리)")
print("="*92)
dd = []
for name in ["DQ", "TBZ", "THI"]:
    ks = [k for k in REP if k.startswith(name + "-")]
    ds = [float(np.linalg.norm(REP[a]-REP[b])) for a,b in itertools.combinations(ks,2)]
    dd += ds
    print(f"  {name:4} 액적쌍 거리: {', '.join(f'{v:.3f}' for v in ds)}")
print(f"  → 액적 잡음 수준(중앙값) ≈ {np.median(dd):.3f}, 최대 {np.max(dd):.3f}")
print(f"  (참고) DQ9 맵쌍 거리 중앙값 0.394,  DQ9-sus 맵쌍 거리 중앙값 0.285")

# ---------- (b) 'TB:TH 항상 동일' 가설 ----------
print()
print("="*92)
print("B. 'sus에서는 TB와 TH가 항상 같이 움직였다(비율 고정)' 가설")
print("   DQ9로 적합한 응답식으로 각 맵의 실제 TB/TH를 역추정한 뒤, 둘의 상관을 본다.")
print("   설계대로면 TB와 TH는 독립 격자이므로 상관 ≈ 0 이어야 한다.")
print("="*92)
ks9 = [c for c in conds if ("DQ9", c) in D["data"]]
tb9 = np.array([int(c.split("-TB")[1].split("-")[0]) for c in ks9], float)
th9 = np.array([int(c.split("-TH")[1]) for c in ks9], float)
fit = {}
for j, nm, cc in [(1,"TBZ",tb9), (2,"THI",th9)]:
    y = np.log([max(U[("DQ9",c)]["frac"][j],1e-3)/max(U[("DQ9",c)]["frac"][0],1e-3) for c in ks9])
    b, a = np.polyfit(np.log(cc/9.0), y, 1); fit[nm] = (a,b)

def est(grp, c):
    f = U[(grp,c)]["frac"]; o = []
    for j, nm in [(1,"TBZ"),(2,"THI")]:
        a, b = fit[nm]
        o.append(9.0*np.exp((np.log(max(f[j],1e-3)/max(f[0],1e-3))-a)/b))
    return o

for grp, ks in [("DQ9", ks9), ("DQ9-sus", shared)]:
    E = np.array([est(grp,c) for c in ks])
    lr = np.log(E[:,0]/E[:,1])
    r_p = stats.pearsonr(np.log(E[:,0]), np.log(E[:,1]))
    print(f"  {grp:9} corr(log추정TB, log추정TH) r={r_p.statistic:+.3f} (p={r_p.pvalue:.3f})   "
          f"추정TB/추정TH 기하평균={np.exp(lr.mean()):.2f}, 기하SD={np.exp(lr.std()):.2f}")
    # 실제 설정비와 비교
    stb = np.array([int(c.split("-TB")[1].split("-")[0]) for c in ks], float)
    sth = np.array([int(c.split("-TH")[1]) for c in ks], float)
    print(f"            설정비 log(TB/TH) 대비 추정비 상관: "
          f"{stats.pearsonr(np.log(stb/sth), lr).statistic:+.3f}")

print()
print("="*92)
print("C. 그렇다면 sus에서 실제로 변한 것은? — 맵별 (농약 총량 / DQ) 비")
print("   '한 stock을 연속 희석' 이면 이 값만 변하고 TB/TH 비는 고정된다")
print("="*92)
for grp, ks in [("DQ9", ks9), ("DQ9-sus", shared)]:
    E = np.array([est(grp,c) for c in ks])
    tot = E.sum(1)
    print(f"  {grp:9} 추정 (TB+TH): 중앙값 {np.median(tot):6.1f}  범위 {tot.min():6.1f}–{tot.max():7.1f}  "
          f"기하SD={np.exp(np.std(np.log(tot))):.2f}")

print()
print("  sus 맵별 상세 (추정 TB, TH, 비율):")
for c in shared:
    a, b = est("DQ9-sus", c)
    stb = int(c.split("-TB")[1].split("-")[0]); sth = int(c.split("-TH")[1])
    print(f"    {c:16} 추정TB={a:7.1f} 추정TH={b:7.1f}  추정TB/TH={a/b:5.2f}   설정TB/TH={stb/sth:5.2f}")
