import pickle, os, numpy as np, itertools
from scipy.stats import spearmanr

d = os.path.dirname(__file__)
D = pickle.load(open(os.path.join(d, "data.pkl"), "rb"))
U = pickle.load(open(os.path.join(d, "unmix.pkl"), "rb"))
wnc, refs, data = D["wnc"], D["refs"], D["data"]
conds = [f"DQ9-TB{tb}-TH{th}" for tb in (9,18,36,72) for th in (9,18,36,72)]
shared = [c for c in conds if ("DQ9-sus", c) in data]

# ---------------------------------------------------------------- 0) 순수 표준 스펙트럼 상호 유사도
print("0. 순수 표준 스펙트럼끼리 cos (NNLS 분해가 신뢰 가능한지)")
for a, b in itertools.combinations(["DQ","TBZ","THI","BLK"], 2):
    print(f"   {a:4}-{b:4} cos={refs[a]@refs[b]:.3f}")

# ---------------------------------------------------------------- 1) 독립적 마커밴드 (NNLS 안 씀)
def selective(target, others, n=3):
    t = refs[target]; o = np.max([refs[x] for x in others], axis=0)
    score = t - 1.2*o
    idx, used = [], np.zeros(len(wnc), bool)
    for _ in range(n):
        s = np.where(used, -np.inf, score)
        i = int(np.argmax(s))
        idx.append(i); used |= np.abs(wnc - wnc[i]) < 25
    return sorted(idx)

MB = {n: selective(n, [x for x in ["DQ","TBZ","THI"] if x != n] + ["BLK"]) for n in ["DQ","TBZ","THI"]}
print("\n1. 성분별 선택적 마커밴드 (다른 성분 기여가 가장 작은 위치)")
for n, ii in MB.items():
    print(f"   {n:4}: " + ", ".join(f"{wnc[i]:.0f}" for i in ii))

def band_int(key, idx, w=8.0):
    X = data[key]["X"][U[key]["good"]]
    out = []
    for i in idx:
        m = np.abs(wnc - wnc[i]) <= w
        out.append(np.median(X[:, m].mean(1)))
    return float(np.mean(out))

BI = {}
for k in data:
    v = {n: band_int(k, MB[n]) for n in MB}
    s = sum(max(x, 0) for x in v.values()) or 1
    BI[k] = {n: max(v[n], 0)/s for n in v}

print("\n2. 마커밴드 기반 상대세기 (NNLS 무관, 독립 검증)")
print(f"{'condition':16} {'DQ9  DQ/TBZ/THI':24} {'DQ9-sus  DQ/TBZ/THI':24}")
for c in shared:
    a, b = BI[("DQ9",c)], BI[("DQ9-sus",c)]
    print(f"{c:16} {'/'.join(f'{a[n]:.3f}' for n in ['DQ','TBZ','THI']):24} "
          f"{'/'.join(f'{b[n]:.3f}' for n in ['DQ','TBZ','THI']):24}")

print("\n3. 마커밴드 기준 농도 단조성 (Spearman)")
for grp in ["DQ9", "DQ9-sus"]:
    ks = [c for c in conds if (grp, c) in data]
    tb = np.array([int(c.split("-TB")[1].split("-")[0]) for c in ks])
    th = np.array([int(c.split("-TH")[1]) for c in ks])
    fTB = np.array([BI[(grp,c)]["TBZ"] for c in ks])
    fTH = np.array([BI[(grp,c)]["THI"] for c in ks])
    print(f"   {grp:9} TB설정→TBZ {spearmanr(tb,fTB).statistic:+.3f}   TH설정→THI {spearmanr(th,fTH).statistic:+.3f}"
          f"   | 교차: TB설정→THI {spearmanr(tb,fTH).statistic:+.3f}   TH설정→TBZ {spearmanr(th,fTB).statistic:+.3f}")

print("\n4. NNLS 기준 교차 단조성 (TBZ↔THI 뒤바뀜 가설 검정)")
for grp in ["DQ9", "DQ9-sus"]:
    ks = [c for c in conds if (grp, c) in data]
    tb = np.array([int(c.split("-TB")[1].split("-")[0]) for c in ks])
    th = np.array([int(c.split("-TH")[1]) for c in ks])
    fTB = np.array([U[(grp,c)]["frac"][1] for c in ks])
    fTH = np.array([U[(grp,c)]["frac"][2] for c in ks])
    print(f"   {grp:9} 정상: TB→TBZ {spearmanr(tb,fTB).statistic:+.3f}, TH→THI {spearmanr(th,fTH).statistic:+.3f}"
          f"   | 교차: TB→THI {spearmanr(tb,fTH).statistic:+.3f}, TH→TBZ {spearmanr(th,fTB).statistic:+.3f}")

print("\n5. 라벨 매핑 가설 비교 (sus 맵 ↔ DQ9 맵, 조성분율 유클리드 거리 평균)")
def dist(susc, dq9c):
    return float(np.linalg.norm(U[("DQ9-sus",susc)]["frac"] - U[("DQ9",dq9c)]["frac"]))
hyp = {}
ident = [(c, c) for c in shared]
swap  = [(c, f"DQ9-TB{c.split('-TH')[1]}-TH{c.split('-TB')[1].split('-')[0]}") for c in shared]
swap  = [(a,b) for a,b in swap if ("DQ9",b) in data]
hyp["동일 라벨 (identity)"] = ident
hyp["TBZ↔THI 라벨 뒤바뀜 (swap)"] = swap
for name, pairs in hyp.items():
    ds = [dist(a,b) for a,b in pairs]
    print(f"   {name:32} n={len(pairs):2}  평균거리={np.mean(ds):.3f}")
# 무작위 기준
rnd = [dist(a,b) for a in shared for b in conds if ("DQ9",b) in data]
print(f"   {'무작위 짝 (기준선)':32} n={len(rnd):2}  평균거리={np.mean(rnd):.3f}")

print("\n6. 맵 내부 산포 (같은 맵 100픽셀의 조성 분율 표준편차) — 비교의 잡음 하한")
D2 = pickle.load(open(os.path.join(d, "data.pkl"), "rb"))
