import numpy as np, glob, os, itertools, json
from scipy.optimize import nnls

BASE = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB/Ratio/Baseline260808"
REF  = "/Users/seungki2/Documents/GitHub/Mixture-Classifier/.claude/worktrees/ink-check-6maps-run-ad8cec/Reference"

def load(path):
    with open(path) as f:
        lines = f.read().strip().split("\n")
    hdr = [v for v in lines[2].split(",") if v.strip()]
    wn = np.array([float(v) for v in hdr[2:]])
    n = len(wn)
    rows = [l.split(",") for l in lines[3:] if l.strip()]
    A = np.array([[float(v) for v in r[2:2+n]] for r in rows], float)
    return wn, A

def als(y, lam=1e5, p=0.01, n=10):
    """asymmetric least squares baseline"""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    DD = lam*D.dot(D.T)
    w = np.ones(L); W = sparse.spdiags(w,0,L,L)
    z = y
    for _ in range(n):
        W.setdiag(w)
        z = spsolve((W+DD).tocsc(), w*y)
        w = p*(y>z) + (1-p)*(y<z)
    return z

def prep(A, wn, mask):
    """baseline-correct + crop each spectrum"""
    out = np.empty((A.shape[0], mask.sum()))
    for i, y in enumerate(A):
        out[i] = (y - als(y))[mask]
    return out

# ---- common grid ----
wn0, _ = load(os.path.join(BASE, "DQ9", "DQ9-TB18-TH18_corrected.csv"))
mask = (wn0 >= 400) & (wn0 <= 1800)
maskI = (wn0 >= 2100) & (wn0 <= 2180)          # ink band 2137
wnc = wn0[mask]

# ---- references (pure components) ----
refs = {}
for name in ["DQ", "TBZ", "THI", "BLK"]:
    S = []
    for f in sorted(glob.glob(os.path.join(REF, f"{name}-*.csv"))):
        w, A = load(f)
        assert np.allclose(w, wn0)
        S.append(np.median(prep(A, w, mask), axis=0))
    r = np.median(np.array(S), axis=0)
    r = np.clip(r, 0, None)
    refs[name] = r / np.linalg.norm(r)
R = np.column_stack([refs[n] for n in ["DQ", "TBZ", "THI", "BLK"]])

# ---- maps ----
conds = [f"DQ9-TB{tb}-TH{th}" for tb in (9,18,36,72) for th in (9,18,36,72)]
data = {}
for grp in ["DQ9", "DQ9-sus"]:
    for c in conds:
        p = os.path.join(BASE, grp, c + "_corrected.csv")
        if not os.path.exists(p): continue
        w, A = load(p)
        ink = np.array([np.trapezoid((y-als(y))[maskI], wn0[maskI]) for y in A])
        X = prep(A, w, mask)
        data[(grp, c)] = dict(X=X, ink=ink)
        print("loaded", grp, c, X.shape, flush=True)

np.save("/tmp/none.npy", np.zeros(1))
import pickle
with open(os.path.join(os.path.dirname(__file__), "data.pkl"), "wb") as f:
    pickle.dump(dict(wnc=wnc, R=R, refs=refs, data=data, wn0=wn0), f)
print("done")
