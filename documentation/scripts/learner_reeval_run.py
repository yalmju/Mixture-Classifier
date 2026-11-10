"""학습기 재평가 — 한 프로토콜: 반복 그룹 5-fold, 모든 학습기 같은 fold, MLP 는 seed 3개.
scheme: cond(89맵 조건 held-out) / ratio(같은 비율 통째 held-out) / grid(64격자만으로 학습·평가)

입력: bundle_89maps_24px.npz (89맵 x 24픽셀, 500-2500 cm-1 로 자른 1,290점, 3출력 DQ/TBZ/THI)
출력: ../results/learner_reeval_2026-11-10/eval_learners.npz  (약 10분, 7코어)
집계: learner_reeval_analyze.py  ->  summary.txt
주의: 배포 모델(4출력, blank·순수 꼭짓점 포함)과 같은 학습이 아니다. 학습기끼리의 비교용."""
import sys, time, json, numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "learner_reeval_2026-11-10")
DRIVE = os.environ.get("UNMIXR_CLONE", os.path.expanduser(
    "~/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/github/Mixture Classifier"))
# 드라이브 IO 가 멈추면 dl_model.py 등 *.py 와 번들을 로컬로 복사하고 UNMIXR_CODE / UNMIXR_BUNDLE 로 지정
CODE = os.environ.get("UNMIXR_CODE", DRIVE)
BUNDLE = os.environ.get("UNMIXR_BUNDLE", os.path.join(
    DRIVE, "documentation/results/composition_all_no100/v2_search/bundle_89maps_24px.npz"))
sys.path.insert(0, CODE)
from joblib import Parallel, delayed

z = np.load(BUNDLE, allow_pickle=True)
ax = 100.0 + 1.55 * np.arange(2001); m = (ax >= 500) & (ax <= 2500)
X = z["X"][:, m].astype(np.float32); Y = z["Y"].astype(np.float32)
P = z["P_ref"][:, m]; MAPS = z["mapkey"].astype(object); G = z["gkey"].astype(object)
CONC = z["concentration"]
UM = np.array(list(dict.fromkeys(MAPS.tolist())), object)
MI = np.array([np.where(UM == k)[0][0] for k in MAPS])           # 픽셀 → 맵 번호
first = np.array([np.where(MI == i)[0][0] for i in range(len(UM))])
YM = Y[first]; CM = CONC[first]; GM = G[first]
GRID = CM.max(1) <= 24.5


def norm(p):
    p = np.clip(np.asarray(p, float), 0, None); s = p.sum(1, keepdims=True)
    return np.where(s > 0, p / np.where(s > 0, s, 1), 1.0 / p.shape[1])


def pool(p, mi, n):
    out = np.full((n, p.shape[1]), np.nan)
    for i in np.unique(mi):
        out[i] = p[mi == i].mean(0)
    return out


def folds(groups, k, seed):
    u = np.array(sorted(set(groups.tolist())), object)
    rng = np.random.default_rng(seed); rng.shuffle(u)
    f = {g: i % k for i, g in enumerate(u)}
    return np.array([f[g] for g in groups])


def fit(method, tr, te, seed, **kw):
    import dl_model
    return norm(dl_model._fit_predict(method, X[tr], Y[tr], X[te], seed=seed, n_trees=300,
                                      rf_max_features="sqrt", P_ref=P, train_maps=MAPS[tr], **kw))


def inner_pick(kind, tr, grid, seed):
    """학습 맵만으로 3-fold 내부 CV → 하이퍼파라미터 선택 (맵 평균 조성 편차 최소)."""
    from sklearn.linear_model import Ridge
    f = folds(MAPS[tr], 3, seed + 99); idx = np.where(tr)[0]; sc = []
    for h in grid:
        e = []
        for j in range(3):
            a = idx[f != j]; b = idx[f == j]
            if kind == "pls":
                from sklearn.cross_decomposition import PLSRegression
                p = norm(PLSRegression(n_components=h).fit(X[a], Y[a]).predict(X[b]))
            else:
                p = norm(Ridge(alpha=h).fit(X[a], Y[a]).predict(X[b]))
            pm = pool(p, MI[b], len(UM)); ok = ~np.isnan(pm[:, 0])
            e.append(50 * np.abs(pm[ok] - YM[ok]).sum(1))
        sc.append(np.concatenate(e).mean())
    return grid[int(np.argmin(sc))]


def run(scheme, rep):
    import torch; torch.set_num_threads(1)
    from sklearn.linear_model import Ridge
    from sklearn.cross_decomposition import PLSRegression
    use = np.ones(len(X), bool) if scheme != "grid" else GRID[MI]
    grp = {"cond": MAPS, "ratio": G, "grid": MAPS}[scheme]
    f = np.full(len(X), -1); f[use] = folds(grp[use], 5, 1000 + rep)
    names = ["nnls", "pls8", "plsT", "ridgeT", "rf", "mlp_s0", "mlp_s1", "mlp_s2", "mlp350"]
    pred = {n: np.full((len(UM), 3), np.nan) for n in names}; picks = []
    for k in range(5):
        tr = use & (f != k); te = use & (f == k); s = 7000 + 10 * rep + k
        out = {}
        out["nnls"] = fit("nnls", tr, te, s)
        out["pls8"] = fit("pls", tr, te, s, n_components=8)
        nc = inner_pick("pls", tr, [3, 5, 8, 12, 16, 22], s)
        out["plsT"] = norm(PLSRegression(n_components=nc).fit(X[tr], Y[tr]).predict(X[te]))
        al = inner_pick("ridge", tr, [0.1, 1, 10, 100, 1000], s)
        out["ridgeT"] = norm(Ridge(alpha=al).fit(X[tr], Y[tr]).predict(X[te]))
        out["rf"] = fit("rf", tr, te, s)
        for j in range(3):
            out[f"mlp_s{j}"] = fit("mlp", tr, te, s + 100 * (j + 1), epochs=120)
        out["mlp350"] = fit("mlp", tr, te, s + 100, epochs=350)
        picks.append((nc, al))
        for n in names:
            pm = pool(out[n], MI[te], len(UM)); ok = ~np.isnan(pm[:, 0]); pred[n][ok] = pm[ok]
    return scheme, rep, pred, picks


if __name__ == "__main__":
    tasks = [("cond", r) for r in range(20)] + [("ratio", r) for r in range(10)] + [("grid", r) for r in range(10)]
    t0 = time.time()
    res = Parallel(n_jobs=7, verbose=10)(delayed(run)(s, r) for s, r in tasks)
    out = {}
    for s, r, pred, picks in res:
        for n, p in pred.items():
            out[f"{s}|{r}|{n}"] = p
        out[f"{s}|{r}|picks"] = np.array(picks, float)
    np.savez(os.path.join(OUT, "eval_learners.npz"), YM=YM, CM=CM, GM=GM.astype(str), UM=UM.astype(str), **out)
    print("done", time.time() - t0)
