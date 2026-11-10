"""고농도 복원용 모델 후보 — 89맵 번들, 조건 held-out 반복 5-fold (eval_learners 와 같은 fold)."""
import sys, time, warnings, numpy as np
warnings.filterwarnings("ignore")
S = "/private/tmp/claude-501/-Users-seungki2-Documents-GitHub-Mixture-Classifier/12cdd5f8-e873-4010-ad13-620bc00ee54e/scratchpad"
from joblib import Parallel, delayed
from scipy.optimize import nnls
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline

z = np.load(S + "/bundle_89maps_24px.npz", allow_pickle=True)
ax = 100.0 + 1.55 * np.arange(2001); m = (ax >= 500) & (ax <= 2500)
X = z["X"][:, m].astype(np.float64); Y = z["Y"].astype(np.float64)
P = z["P_ref"][:, m]; MAPS = z["mapkey"].astype(object); CONC = z["concentration"]
UM = np.array(list(dict.fromkeys(MAPS.tolist())), object)
MI = np.array([np.where(UM == k)[0][0] for k in MAPS])
first = np.array([np.where(MI == i)[0][0] for i in range(len(UM))])
YM = Y[first]; CM = CONC[first]; GRID = CM.max(1) <= 24.5
T = np.log10(CONC.sum(1))[:, None]
RAW = np.expm1(np.clip(X, 0, None))
A = np.array([nnls(P.T, r)[0] for r in RAW])                      # 픽셀별 NNLS 계수
la = np.log(A + 1e-3 * A.sum(1, keepdims=True) + 1e-9)
NF = np.column_stack([la - la.mean(1, keepdims=True), np.log(A.sum(1) + 1e-9), np.log1p(RAW.sum(1))])

def norm(p):
    p = np.clip(np.asarray(p, float), 0, None); s = p.sum(1, keepdims=True)
    return np.where(s > 0, p / np.where(s > 0, s, 1), 1.0 / p.shape[1])
def clr(Yv, eps=0.02):
    L = np.log((Yv + eps) / (1 + 3 * eps)); return L - L.mean(1, keepdims=True)
def sm(L):
    e = np.exp(L - L.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def pool(p, mi, n):
    out = np.full((n, p.shape[1]), np.nan)
    for i in np.unique(mi): out[i] = p[mi == i].mean(0)
    return out
def folds(groups, k, seed):
    u = np.array(sorted(set(groups.tolist())), object)
    rng = np.random.default_rng(seed); rng.shuffle(u)
    f = {g: i % k for i, g in enumerate(u)}
    return np.array([f[g] for g in groups])
def gbm_multi(Ftr, Ltr, Fte):
    return np.column_stack([HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_depth=3,
            l2_regularization=1.0, random_state=0).fit(Ftr, Ltr[:, j]).predict(Fte) for j in range(3)])

def models(tr, te):
    o = {}
    pls = PLSRegression(n_components=8).fit(X[tr], Y[tr]); o["pls8"] = norm(pls.predict(X[te]))
    p12 = PLSRegression(n_components=12).fit(X[tr], Y[tr])
    St, Se = p12.transform(X[tr]), p12.transform(X[te])
    sc = StandardScaler().fit(St); St, Se = sc.transform(St), sc.transform(Se)
    L = clr(Y[tr])
    o["pls8_clr"] = sm(PLSRegression(n_components=8).fit(X[tr], L).predict(X[te]))
    o["pls12_clr"] = sm(PLSRegression(n_components=12).fit(X[tr], L).predict(X[te]))
    o["ridge_clr"] = sm(RidgeCV(alphas=[1, 10, 100, 1000]).fit(X[tr], L).predict(X[te]))
    o["nnlsF_gbm"] = sm(gbm_multi(NF[tr], L, NF[te]))
    o["nnlsF_poly2"] = sm(make_pipeline(StandardScaler(), PolynomialFeatures(2), RidgeCV(alphas=[.1, 1, 10, 100]))
                          .fit(NF[tr], L).predict(NF[te]))
    o["sc_krr"] = norm(KernelRidge(kernel="rbf", alpha=0.3, gamma=1 / 24).fit(St, Y[tr]).predict(Se))
    o["sc_knn"] = norm(KNeighborsRegressor(15, weights="distance").fit(St, Y[tr]).predict(Se))
    o["sc_gbm"] = sm(gbm_multi(St, L, Se))
    o["sc+nnlsF_gbm"] = sm(gbm_multi(np.hstack([St, NF[tr]]), L, np.hstack([Se, NF[te]])))
    # 선언 총농도를 입력으로
    o["sc_krr+T"] = norm(KernelRidge(kernel="rbf", alpha=0.3, gamma=1 / 26).fit(np.hstack([St, 2 * T[tr]]), Y[tr])
                         .predict(np.hstack([Se, 2 * T[te]])))
    o["sc_gbm+T"] = sm(gbm_multi(np.hstack([St, T[tr]]), L, np.hstack([Se, T[te]])))
    o["nnlsF_gbm+T"] = sm(gbm_multi(np.hstack([NF[tr], T[tr]]), L, np.hstack([NF[te], T[te]])))
    # 구간 분리(오라클: 시험 맵이 격자인지 고농도인지 안다고 가정)
    g = GRID[MI]; ps = np.zeros((te.sum(), 3)); gte = g[te]
    for flag in (True, False):
        a = tr & (g == flag)
        if gte.__eq__(flag).any() and a.sum() > 50:
            ps[gte == flag] = norm(PLSRegression(n_components=min(8, 5 if not flag else 8)).fit(X[a], Y[a]).predict(X[te][gte == flag]))
    o["pls_split(oracle)"] = ps
    # 고농도 3배 가중(복제)
    idx = np.concatenate([np.where(tr)[0]] + [np.where(tr & ~g)[0]] * 2)
    o["pls8_hi×3"] = norm(PLSRegression(n_components=8).fit(X[idx], Y[idx]).predict(X[te]))
    return o

def run(rep):
    f = folds(MAPS, 5, 1000 + rep); pred = {}
    for k in range(5):
        tr = f != k; te = f == k
        for n, p in models(tr, te).items():
            pm = pool(p, MI[te], len(UM)); ok = ~np.isnan(pm[:, 0])
            pred.setdefault(n, np.full((len(UM), 3), np.nan))[ok] = pm[ok]
    return pred

if __name__ == "__main__":
    t0 = time.time(); R = Parallel(n_jobs=7)(delayed(run)(r) for r in range(10))
    k = (CM > 0).sum(1)
    SUB = {"grid64": GRID, "hi-bin": ~GRID & (k == 2), "hi-tern": ~GRID & (k == 3), "hi-all": ~GRID}
    dev = lambda p: 50 * np.abs(p - YM).sum(1)
    print(f"{'model':20s}" + "".join(f"{q:>10s}" for q in SUB) + "   hi major-ok%")
    for n in R[0]:
        D = np.array([dev(r[n]) for r in R])
        maj = np.mean([(r[n].argmax(1) == YM.argmax(1))[~GRID & (YM.max(1) > 0.6)].mean() for r in R]) * 100
        print(f"{n:20s}" + "".join(f"{D[:, s].mean():10.1f}" for s in SUB.values()) + f"   {maj:5.0f}")
    np.savez(S + "/zoo.npz", YM=YM, CM=CM, **{f"{r}|{n}": R[r][n] for r in range(10) for n in R[0]})
    print("done", time.time() - t0)
