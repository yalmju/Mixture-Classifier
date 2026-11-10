"""농도 단계 비교 — 같은 12개 맵 특징, 같은 목표(Δlog10), 같은 fold. 신경망(128-32) 대 선형(Ridge) 대 PLS."""
import os, sys, numpy as np
REPO = "/Users/seungki2/Documents/GitHub/Mixture-Classifier"
sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(f"{HERE}/head_inputs.npz", allow_pickle=True)
F, CC, T, COND = z["F"], z["Ccal"], z["T"], z["cond"].astype(object)
GRID = np.isin(T, [3, 6, 12, 24]).all(1)
LT = np.log10(np.clip(T, 0.05, None)) - np.log10(np.clip(CC, 0.05, None))
print("maps", len(F), "conditions", len(set(COND.tolist())), "grid", GRID.sum(), flush=True)


def folds(groups, k, seed):
    u = np.array(sorted(set(groups.tolist())), object)
    rng = np.random.default_rng(seed); rng.shuffle(u)
    f = {g: i % k for i, g in enumerate(u)}
    return np.array([f[g] for g in groups])


def lin(kind, tr, te, seed):
    """성분별로 따로 맞춘다(없는 성분 T=0 은 뺀다). 하이퍼파라미터는 학습 쪽 3-fold 로 고른다."""
    from sklearn.linear_model import Ridge
    from sklearn.cross_decomposition import PLSRegression
    mu = F[tr].mean(0); sd = F[tr].std(0) + 1e-6
    A = (F - mu) / sd
    grid = [0.1, 1, 10, 100] if kind == "ridge" else [1, 2, 3, 4, 6]
    mk = (lambda h: Ridge(alpha=h)) if kind == "ridge" else (lambda h: PLSRegression(n_components=h))
    out = np.zeros((len(te), T.shape[1]))
    f = folds(COND[tr], 3, seed + 99)
    for j in range(T.shape[1]):
        ok = T[tr, j] > 0; a = tr[ok]; fa = f[ok]; sc = []
        for h in grid:
            e = []
            for q in range(3):
                p = np.ravel(mk(h).fit(A[a[fa != q]], LT[a[fa != q], j]).predict(A[a[fa == q]]))
                e.append(np.abs(p - LT[a[fa == q], j]))
            sc.append(np.concatenate(e).mean())
        h = grid[int(np.argmin(sc))]
        out[:, j] = np.ravel(mk(h).fit(A[a], LT[a, j]).predict(A[te]))
    return np.clip(CC[te] * 10.0 ** np.clip(out, -2, 2), 1e-3, 5e3)


def net(tr, te, seed):
    import torch, dl_model
    torch.set_num_threads(1)
    n, mu, sd, _, _, _ = dl_model._fit_residual_net(F[tr], CC[tr], T[tr], COND[tr], seed=seed)
    return dl_model._predict_residual_uM(n, mu, sd, F[te], CC[te])


def cv(rep, k=5):
    f = folds(COND, k, 1000 + rep)
    P = {n: np.full(T.shape, np.nan) for n in ("cal", "ridge", "pls", "net")}
    for q in range(k):
        tr = np.where(f != q)[0]; te = np.where(f == q)[0]; s = 7000 + 10 * rep + q
        P["cal"][te] = CC[te]; P["ridge"][te] = lin("ridge", tr, te, s)
        P["pls"][te] = lin("pls", tr, te, s); P["net"][te] = net(tr, te, s)
    return P


def loo():
    """모델 파일에 기록된 평가와 같은 방식 — 조건 하나씩 빼기, 같은 seed."""
    P = {n: np.full(T.shape, np.nan) for n in ("cal", "ridge", "pls", "net")}
    for i, held in enumerate(dict.fromkeys(COND.tolist())):
        te = np.where(COND == held)[0]; tr = np.where(COND != held)[0]
        P["cal"][te] = CC[te]; P["ridge"][te] = lin("ridge", tr, te, i)
        P["pls"][te] = lin("pls", tr, te, i); P["net"][te] = net(tr, te, 1000 + i)
    return P


def score(p, m=None):
    m = GRID if m is None else m
    t = T[m]; q = p[m]; ok = t > 0
    fold = np.maximum(q[ok] / t[ok], t[ok] / q[ok])
    return dict(w15=100 * (fold <= 1.5).mean(), w2=100 * (fold <= 2).mean(), w3=100 * (fold <= 3).mean(),
                med=np.median(fold), mae=np.abs(q[ok] - t[ok]).mean(), rec=100 * np.median(q[ok] / t[ok]))


if __name__ == "__main__":
    from joblib import Parallel, delayed
    L = loo()
    lines = ["농도 단계 비교 — 논문용 PLS 모델의 학습 82맵, 12개 맵 특징 동일, 조성 입력은 전체로 학습한 PLS(모델 기록과 같은 조건)",
             f"64격자 평가 맵 {GRID.sum()}장 · 성분 {int((T[GRID] > 0).sum())}개", "",
             "[조건 하나씩 빼기 — 모델 파일 기록과 같은 방식]"]
    for n in ("cal", "ridge", "pls", "net"):
        s = score(L[n])
        lines.append(f"  {n:6s} 1.5배 {s['w15']:5.1f} % · 2배 {s['w2']:5.1f} % · 3배 {s['w3']:5.1f} % · 중앙 배수 {s['med']:.2f}×"
                     f" · MAE {s['mae']:.2f} µM · 중앙 회수율 {s['rec']:.0f} %")
    R = Parallel(n_jobs=7)(delayed(cv)(r) for r in range(20))
    lines += ["", "[반복 조건 묶음 5-fold × 20회]"]
    S = {n: [score(P[n]) for P in R] for n in ("cal", "ridge", "pls", "net")}
    for n in S:
        g = lambda k: np.array([s[k] for s in S[n]])
        lines.append(f"  {n:6s} 1.5배 {g('w15').mean():5.1f} ± {g('w15').std():.1f} % · 2배 {g('w2').mean():5.1f} ± {g('w2').std():.1f} %"
                     f" · 3배 {g('w3').mean():5.1f} % · 중앙 배수 {g('med').mean():.2f}× · MAE {g('mae').mean():.2f} µM"
                     f" · 중앙 회수율 {g('rec').mean():.0f} %")
    for a in ("ridge", "pls"):
        for k, nm in (("w2", "2배 이내"), ("w15", "1.5배 이내"), ("mae", "MAE")):
            d = np.array([x[k] for x in S["net"]]) - np.array([x[k] for x in S[a]])
            better = (d > 0).sum() if k != "mae" else (d < 0).sum()
            lines.append(f"  신경망 − {a:5s} {nm}: {d.mean():+.2f} (20회 중 신경망이 나은 횟수 {better})")
    # 성분별 (반복 평균)
    lines += ["", "[성분별 2배 이내 %, 반복 평균]"]
    for n in ("cal", "ridge", "pls", "net"):
        v = []
        for j in range(T.shape[1]):
            w = []
            for P in R:
                t = T[GRID, j]; q = P[n][GRID, j]; w.append(100 * (np.maximum(q / t, t / q) <= 2).mean())
            v.append(np.mean(w))
        lines.append(f"  {n:6s} " + " · ".join(f"{s} {x:5.1f}" for s, x in zip(z['subs'], v)))
    txt = "\n".join(lines); print(txt)
    open(f"{HERE}/head_cv_summary.txt", "w", encoding="utf-8").write(txt + "\n")
    np.savez(f"{HERE}/head_cv_preds.npz", T=T, GRID=GRID, **{f"loo_{n}": L[n] for n in L},
             **{f"cv{r}_{n}": R[r][n] for r in range(20) for n in R[r]})
