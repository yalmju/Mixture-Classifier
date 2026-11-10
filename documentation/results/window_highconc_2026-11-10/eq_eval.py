import numpy as np, re, warnings
from scipy.optimize import nnls
from sklearn.cross_decomposition import PLSRegression
warnings.filterwarnings("ignore")
z = np.load("all102.npz", allow_pickle=True)
wn = z["wn"]; m = (wn >= 500) & (wn <= 2500)
RAW = np.clip(z["X"][:, m], 0, None); X = np.log1p(RAW); P = z["P"][:, m]
g = z["g"].astype(str); C = z["C"]; Y = C / C.sum(1, keepdims=True)
maps = np.array(sorted(set(g)))
Cm = np.array([C[g == k][0] for k in maps]); Ym = Cm / Cm.sum(1, keepdims=True)
GRID = {3, 6, 12, 24}
def tag(k, c):
    if "(2)" in k: return "eq"
    if all(v in GRID for v in c): return "grid"
    if c[0] == c[1] == c[2] and c[0] in (10, 33, 100): return "eq"
    if sorted(c)[0] in (3, 5) and max(c) >= 333: return "extreme"
    return "hi"
T = np.array([tag(k, c) for k, c in zip(maps, Cm)])
print({t: int((T == t).sum()) for t in set(T)})
def dev(p, y): return 50 * np.abs(p - y).sum(-1)
def pool(pred, gg): return np.array([pred[gg == k].mean(0) for k in maps if (gg == k).any()])
A = np.array([nnls(P.T, r)[0] for r in RAW]); An = A / np.clip(A.sum(1, keepdims=True), 1e-12, None)
NN = np.array([An[g == k].mean(0) for k in maps])
def pls(tr, te, nc=8):
    p = PLSRegression(nc).fit(X[tr], Y[tr]).predict(X[te]); p = np.clip(p, 0, None)
    return p / np.clip(p.sum(1, keepdims=True), 1e-9, None)
# (a) sanity: 조건 held-out 5-fold, 89맵(grid+hi)
base = np.isin(T, ["grid", "hi"]); bm = maps[base]
out = []
for rep in range(10):
    rng = np.random.default_rng(1000 + rep); perm = rng.permutation(len(bm)); pm = np.zeros((len(maps), 3))
    for f in range(5):
        tem = bm[perm[f::5]]; te = np.isin(g, tem); tr = np.isin(g, bm) & ~te
        pr = pls(tr, te)
        for k in tem: pm[maps == k] = pr[g[te] == k].mean(0)
    out.append(pm)
PM = np.mean(out, 0)
for t in ("grid", "hi"):
    s = T == t
    print(f"CV {t:5s} n={s.sum():2d}  NNLS {dev(NN[s], Ym[s]).mean():5.1f}  PLS {np.mean([dev(o[s], Ym[s]).mean() for o in out]):5.1f}")
# (b) 89맵으로 학습 → 나머지 13맵 (완전 held-out) ; (c) grid 64맵만으로 학습 → 나머지
for name, trset in (("train=grid+hi(89)", ["grid", "hi"]), ("train=grid(64)", ["grid"])):
    tr = np.isin(g, maps[np.isin(T, trset)])
    print("\n" + name)
    for k, c, y, t, nn in zip(maps, Cm, Ym, T, NN):
        if t in ("eq", "extreme"):
            p = pls(tr, g == k).mean(0)
            print(f"  {k[:-4]:24s} true {np.round(100*y).astype(int)}  NNLS {np.round(100*nn).astype(int)} ({dev(nn,y):4.1f})  PLS {np.round(100*p).astype(int)} ({dev(p,y):4.1f})")
# 격자 안 등몰(3,6,12,24): CV 예측
print("\n격자 등몰 (CV)")
for k, c, y, nn, p in zip(maps, Cm, Ym, NN, PM):
    if c[0] == c[1] == c[2] and c[0] in GRID:
        print(f"  {k[:-4]:24s} NNLS {np.round(100*nn).astype(int)} ({dev(nn,y):4.1f})  PLS {np.round(100*p).astype(int)} ({dev(p,y):4.1f})")
# 333 vs 333(2)
np.savez("eq_eval.npz", maps=maps, Cm=Cm, T=T, NN=NN, PM=PM)
