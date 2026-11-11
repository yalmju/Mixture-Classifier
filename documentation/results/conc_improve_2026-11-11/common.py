"""공통: 캐시 읽기 · 입력 만들기 · 반복 조건 묶음 CV (2026-11-11 농도 개선 탐색)."""
import os, sys, warnings, numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(f"{HERE}/../../..")
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/documentation/scripts")
from sklearn.cross_decomposition import PLSRegression

z = np.load(f"{HERE}/cache.npz", allow_pickle=True)
PA_all, PR_all, T_all, R_all, M_all = z["PA"].astype(float), z["PR"].astype(float), z["T"], z["R"], z["M"]
paths_all, ax, INK = z["paths"], z["ax"], float(z["ink"])
GRID = {3, 6, 12, 24}
ing = np.array([all(v in GRID for v in t) for t in T_all])
PA, PR, T, R, M, paths = PA_all[ing], PR_all[ing], T_all[ing], R_all[ing], M_all[ing], paths_all[ing]
LT = np.log10(T); N = len(T)
cond = np.array(["-".join("%g" % v for v in t) for t in T])
SUB = ["DQ", "TBZ", "THI"]
L2, L15 = np.log10(2) + 1e-9, np.log10(1.5) + 1e-9


def band(X, c, hw=10.0):
    mk = np.abs(ax - c) <= hw
    return np.clip(X, 0, None)[..., mk].max(-1)


def sel_of(i):
    s = np.where(M[i] >= 0.5)[0]
    return s if len(s) else np.where(M[i] >= 0)[0]


def base_features(P=None, ink_from=None, divide=True):
    """채택 식의 입력: ln(1 + 선택 픽셀 평균 스펙트럼) − ln(1 + 평균 잉크 밴드)."""
    P = PA if P is None else P; I = PA if ink_from is None else ink_from
    X = []
    for i in range(N):
        s = sel_of(i); row = np.log1p(np.clip(P[i][s], 0, None).mean(0))
        if divide:
            row = row - np.log1p(band(I[i][s], INK).mean())
        X.append(row)
    return np.array(X)


def pls_fit_predict(k):
    def f(Xa, Ya, Xb):
        kk = int(min(k, len(Xa) - 1, Xa.shape[1]))
        return np.asarray(PLSRegression(n_components=kk, scale=False).fit(Xa, Ya).predict(Xb), float).reshape(len(Xb), -1)
    return f


def const_fit_predict(Xa, Ya, Xb):
    return np.repeat(Ya.mean(0)[None], len(Xb), 0)


def folds(rep, n_fold=5):
    uc = np.unique(cond); rng = np.random.default_rng(rep)
    f = dict(zip(uc, rng.permutation(len(uc)) % n_fold))
    return np.array([f[c] for c in cond])


def cv_predict(X, fp, reps=20, Y=None):
    """held-out 예측 (reps, N, 3) — log10 µM."""
    Y = LT if Y is None else Y; out = np.zeros((reps, N, Y.shape[1]))
    for r in range(reps):
        fo = folds(r)
        for k in range(5):
            te = fo == k; out[r][te] = fp(X[~te], Y[~te], X[te])
    return out


def metrics(P):
    """P (reps, N, 3) → 반복별 지표 dict (각 값은 길이 reps 배열)."""
    E = np.abs(P - LT[None]); d = {}
    d["w2"] = 100 * (E <= L2).mean((1, 2)); d["w15"] = 100 * (E <= L15).mean((1, 2))
    d["med"] = 10 ** np.median(E.reshape(len(P), -1), 1); d["rmse"] = np.sqrt(((P - LT[None]) ** 2).mean((1, 2)))
    d["slope"] = np.array([np.polyfit(LT.ravel(), p.ravel(), 1)[0] for p in P])
    d["all3"] = (E <= L2).all(2).sum(1)
    d["w2c"] = 100 * (E <= L2).mean(1)            # (reps, 3)
    return d


def line(name, d, ref=None):
    s = (f"{name:<46s} 2배 {d['w2'].mean():5.1f} ± {d['w2'].std():3.1f} | 1.5배 {d['w15'].mean():5.1f} | {d['med'].mean():4.2f}× | "
         f"RMSE {d['rmse'].mean():.3f} | 기울기 {d['slope'].mean():.2f} | 셋 다 {d['all3'].mean():4.1f}/64 | "
         f"DQ {d['w2c'][:, 0].mean():4.1f} TBZ {d['w2c'][:, 1].mean():4.1f} THI {d['w2c'][:, 2].mean():4.1f}")
    if ref is not None:
        n = len(d["w2"])
        s += (f" || 기준보다 나음 2배 {int((d['w2'] > ref['w2']).sum())}/{n} · 1.5배 {int((d['w15'] > ref['w15']).sum())}/{n} · "
              f"RMSE {int((d['rmse'] < ref['rmse']).sum())}/{n}")
    return s


class Log:
    def __init__(self, path): self.path, self.rows = path, []
    def __call__(self, s=""):
        print(s, flush=True); self.rows.append(s)
        open(self.path, "w").write("\n".join(self.rows) + "\n")
