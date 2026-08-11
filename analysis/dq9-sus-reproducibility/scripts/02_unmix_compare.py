import pickle, os, numpy as np, itertools
from scipy.optimize import nnls

D = pickle.load(open(os.path.join(os.path.dirname(__file__), "data.pkl"), "rb"))
wnc, R, data = D["wnc"], D["R"], D["data"]
NAMES = ["DQ", "TBZ", "THI", "BLK"]
conds = [f"DQ9-TB{tb}-TH{th}" for tb in (9,18,36,72) for th in (9,18,36,72)]
shared = [c for c in conds if ("DQ9-sus", c) in data]

def unmix_map(key):
    X = data[key]["X"]
    C = np.array([nnls(R, np.clip(x, 0, None))[0] for x in X])
    # reconstruction quality per pixel
    res = np.array([np.linalg.norm(R@c - np.clip(x,0,None))/max(np.linalg.norm(np.clip(x,0,None)),1e-9)
                    for c, x in zip(C, X)])
    good = res < np.percentile(res, 80)          # drop worst 20% pixels
    return C, res, good

def frac(C, good):
    A = C[good][:, :3]                            # DQ, TBZ, THI (ink excluded)
    s = A.sum(1, keepdims=True); s[s == 0] = 1
    return np.median(A/s, axis=0)

def med_spec(key, good):
    return np.median(data[key]["X"][good], axis=0)

U = {}
for k in data:
    C, res, good = unmix_map(k)
    U[k] = dict(C=C, good=good, frac=frac(C, good), med=med_spec(k, good),
                ink=np.median(data[k]["ink"][good]))

def cos(a, b):
    a = np.clip(a,0,None); b = np.clip(b,0,None)
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))

print("="*88)
print("A. 같은 조건 DQ9 vs DQ9-sus  —  조성 분율(중앙값, 잉크 제외) 및 스펙트럼 유사도")
print("="*88)
print(f"{'condition':16} {'DQ9 (DQ/TBZ/THI)':26} {'DQ9-sus (DQ/TBZ/THI)':26} {'cos':>6} {'ΔTBZ':>7} {'ΔTHI':>7}")
rows = []
for c in shared:
    a, b = U[("DQ9", c)], U[("DQ9-sus", c)]
    fa, fb = a["frac"], b["frac"]
    cs = cos(a["med"], b["med"])
    rows.append((c, fa, fb, cs))
    print(f"{c:16} {'/'.join(f'{v:.3f}' for v in fa):26} {'/'.join(f'{v:.3f}' for v in fb):26} "
          f"{cs:6.3f} {fb[1]-fa[1]:+7.3f} {fb[2]-fa[2]:+7.3f}")

print()
print("="*88)
print("B. 대조군: 같은 폴더 안에서 '다른 조건'끼리는 얼마나 다른가 (cos 분포)")
print("="*88)
for grp in ["DQ9", "DQ9-sus"]:
    ks = [c for c in conds if (grp, c) in data]
    vals = [cos(U[(grp,x)]["med"], U[(grp,y)]["med"]) for x, y in itertools.combinations(ks, 2)]
    print(f"{grp:9} 다른조건 cos: median {np.median(vals):.3f}  min {np.min(vals):.3f}  max {np.max(vals):.3f}")
same = [r[3] for r in rows]
print(f"{'SAME cond':9} DQ9 vs sus cos: median {np.median(same):.3f}  min {np.min(same):.3f}  max {np.max(same):.3f}")

print()
print("="*88)
print("C. 농도 단조성 (mixing 실수 탐지의 핵심): 설정 농도가 오르면 해당 성분 분율도 올라야 함")
print("="*88)
for grp in ["DQ9", "DQ9-sus"]:
    ks = [c for c in conds if (grp, c) in data]
    tb = np.array([int(c.split("-TB")[1].split("-")[0]) for c in ks])
    th = np.array([int(c.split("-TH")[1]) for c in ks])
    fTB = np.array([U[(grp,c)]["frac"][1] for c in ks])
    fTH = np.array([U[(grp,c)]["frac"][2] for c in ks])
    fDQ = np.array([U[(grp,c)]["frac"][0] for c in ks])
    def sp(x, y):
        from scipy.stats import spearmanr
        return spearmanr(x, y).statistic
    print(f"{grp:9} Spearman(TBZ설정, TBZ분율)={sp(tb,fTB):+.3f}   "
          f"Spearman(THI설정, THI분율)={sp(th,fTH):+.3f}   "
          f"DQ분율 평균={fDQ.mean():.3f} (설정상 모든 맵에서 DQ=9로 동일)")
    print("           TB별 TBZ분율 평균:", {int(v): round(float(fTB[tb==v].mean()),3) for v in sorted(set(tb))})
    print("           TH별 THI분율 평균:", {int(v): round(float(fTH[th==v].mean()),3) for v in sorted(set(th))})

print()
print("="*88)
print("D. 매칭 테스트: DQ9-sus 각 맵은 DQ9의 '어느' 조건과 가장 닮았나? (조성 분율 거리)")
print("="*88)
allDQ9 = [c for c in conds if ("DQ9", c) in data]
nmis = 0
for c in shared:
    fb = U[("DQ9-sus", c)]["frac"]
    d = {k: float(np.linalg.norm(U[("DQ9",k)]["frac"] - fb)) for k in allDQ9}
    order = sorted(d, key=d.get)
    hit = order[0] == c
    nmis += (not hit)
    print(f"{c:16} best={order[0]:16} d={d[order[0]]:.3f}   (self d={d[c]:.3f}, rank={order.index(c)+1}/{len(order)})  {'OK' if hit else 'MISMATCH'}")
print(f"\n  → 최근접 매칭 실패 {nmis}/{len(shared)}")

print()
print("="*88)
print("E. 잉크(2137) 세기 — 액적/전처리 차이 확인용")
print("="*88)
for c in shared:
    a, b = U[("DQ9", c)]["ink"], U[("DQ9-sus", c)]["ink"]
    print(f"{c:16} DQ9={a:10.1f}  sus={b:10.1f}  ratio={b/a:5.2f}")

pickle.dump({k: {kk: vv for kk, vv in v.items() if kk != 'C'} for k, v in U.items()},
            open(os.path.join(os.path.dirname(__file__), "unmix.pkl"), "wb"))
