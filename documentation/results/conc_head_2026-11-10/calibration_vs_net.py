"""농도 단계: 검량선으로 되는가, 신경망이 필요한가 (2026-11-10, 사용자 질문).

(1) 단일성분 검량선에 혼합물 밴드 높이를 그대로 넣기 — fig4 held-out 64 조건.
(2) 혼합물로 맞춘 검량선 log10 C = a·log10(밴드) + b·(PLS 몫) + c — 조건 하나씩 빼기, PLS 몫도 held-out.
(3) head_cv.py 와 같은 52맵 · 같은 fold(반복 조건 묶음 5-fold × 20) 에서 신경망과 직접 비교.
    선형 쪽은 3–24 µM 격자 맵만으로 맞춘다(신경망 기록은 격자 밖 30맵까지 넣고 학습한 것).
"""
import csv, os, sys, numpy as np
REPO = "/Users/seungki2/Documents/GitHub/Mixture-Classifier"
sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
FIG4 = f"{REPO}/documentation/results/fig4_paper_pls_2026-11-10"
S = ["DQ", "TBZ", "THI"]
OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


def stat(C, T):
    q = np.maximum(C / T, T / C)
    return (100 * (q <= 2).mean(), *(100 * (q <= 2).mean(0)), 100 * (q <= 3).mean(), np.median(q))


def line(tag, C, T):
    a, d, t, h, w3, med = stat(C, T)
    say(f"  {tag:34s} 2배 이내 {a:5.1f} % (DQ {d:5.1f} · TBZ {t:5.1f} · THI {h:5.1f}) · 3배 {w3:5.1f} % · 중앙 배수 {med:.2f}×")


def fit_predict(Xtr, ytr, Xte):
    A = np.c_[Xtr, np.ones(len(Xtr))]
    w = np.linalg.lstsq(A, ytr, rcond=None)[0]
    return np.c_[Xte, np.ones(len(Xte))] @ w


# ── (1)(2) fig4 held-out 64 조건: 밴드 높이(ALS 뒤) + held-out PLS 몫 ─────────────────────
cal = {}
for r in csv.DictReader(open(f"{FIG4}/band_calibration.csv", encoding="utf-8-sig")):
    cal.setdefault(r["compound"], []).append((float(r["C_uM"]), float(r["band_height"])))
rows = [r for r in csv.DictReader(open(f"{FIG4}/heldout_maps.csv", encoding="utf-8-sig")) if r["grid64"] == "1"]
T = np.array([[float(r[s + "_true_uM"]) for s in S] for r in rows])
B = np.array([[float(r[s + "_band"]) for s in S] for r in rows])
P = np.array([[float(r[s + "_PLS"]) for s in S] for r in rows]); P = P / P.sum(1, keepdims=True)
cond = np.array([r["condition"] for r in rows])
lb = np.log10(np.clip(B, 1, None)); lt = np.log10(T)

say(f"[1] 단일성분 검량선에 혼합물 밴드 높이를 넣기 — held-out 64 조건 · 성분 {T.size}개")
Cl = np.zeros_like(T)
for j, s in enumerate(S):
    c = np.array(sorted(cal[s])); m = (c[:, 0] >= 0.1) & (c[:, 0] <= 10)
    a, b = np.polyfit(c[m, 0], c[m, 1], 1)
    Cl[:, j] = np.clip((B[:, j] - b) / a, 0.05, None)
    mix = " / ".join(f"{np.median(B[T[:, j] == v, j]):.0f}" for v in (3, 6, 12, 24))
    one = " / ".join(f"{a * v + b:.0f}" for v in (3, 6, 12, 24))
    r2 = np.corrcoef(lt[:, j], lb[:, j])[0, 1] ** 2
    say(f"  {s:3s} 밴드 높이 3/6/12/24 µM — 혼합물 중앙 {mix} · 단일 검량 직선 {one} · r²(log 밴드, log 자기 농도) {r2:.2f}")
line("상수(기하평균)", np.full_like(T, 10 ** lt.mean()), T)
line("단일 검량 직선(0.1–10 µM)", Cl, T)


def loo(X_of):
    C = np.zeros_like(T)
    for j in range(3):
        X = X_of(j)
        for i in range(len(T)):
            tr = cond != cond[i]
            C[i, j] = 10 ** fit_predict(X[tr], lt[tr, j], X[[i]])[0]
    return C


say()
say("[2] 혼합물로 맞춘 검량선 (조건 하나씩 빼기, PLS 몫도 그 조건을 빼고 학습한 값)")
line("PLS 몫만 (대조)", loo(lambda j: P[:, [j]]), T)
line("자기 밴드만", loo(lambda j: lb[:, [j]]), T)
line("자기 밴드 + 자기 몫", loo(lambda j: np.c_[lb[:, j], P[:, j]]), T)
line("세 밴드 + 몫", loo(lambda j: np.c_[lb, P[:, :2]]), T)

# ── (3) head_cv.py 와 같은 52맵 · 같은 fold ───────────────────────────────────────────
z = np.load(f"{HERE}/head_inputs.npz", allow_pickle=True)
F, CC, TT, COND = z["F"], z["Ccal"], z["T"], z["cond"].astype(object)
del z
GRID = np.isin(TT, [3, 6, 12, 24]).all(1)
_pz = np.load(f"{HERE}/head_cv_preds.npz", allow_pickle=True)
prev = {k: np.array(_pz[k]) for k in _pz.files if k.startswith("cv")}; del _pz
RAT = F[:, 3:6]; LB = np.log10(np.expm1(F[:, 6:9]) + 1.0); IQ = F[:, 9:12]; LTT = np.log10(np.clip(TT, 0.05, None))


def folds(groups, k, seed):
    u = np.array(sorted(set(groups.tolist())), object)
    rng = np.random.default_rng(seed); rng.shuffle(u)
    f = {g: i % k for i, g in enumerate(u)}
    return np.array([f[g] for g in groups])


FEATS = {
    "PLS 몫만 (대조)": lambda j: RAT[:, [j]],
    "자기 밴드만": lambda j: LB[:, [j]],
    "자기 밴드 + 자기 몫": lambda j: np.c_[LB[:, j], RAT[:, j]],
    "세 밴드 + 몫": lambda j: np.c_[LB, RAT[:, :2]],
    "세 밴드 + 몫 + 총세기 분위수": lambda j: np.c_[LB, RAT[:, :2], IQ],
}


def net_grid(tr, te, seed):
    import torch, dl_model
    torch.set_num_threads(1)
    n, mu, sd, _, _, _ = dl_model._fit_residual_net(F[tr], CC[tr], TT[tr], COND[tr], seed=seed)
    return dl_model._predict_residual_uM(n, mu, sd, F[te], CC[te])


def cv(rep, k=5):
    f = folds(COND, k, 1000 + rep)                       # head_cv.py 와 같은 fold
    Pd = {n: np.full(TT.shape, np.nan) for n in list(FEATS) + ["신경망(격자 맵만 학습)"]}
    for q in range(k):
        te = np.where((f == q) & GRID)[0]; tr = np.where((f != q) & GRID)[0]
        if not len(te):
            continue
        for n, X_of in FEATS.items():
            for j in range(3):
                X = X_of(j)
                Pd[n][te, j] = 10 ** fit_predict(X[tr], LTT[tr, j], X[te])
        Pd["신경망(격자 맵만 학습)"][te] = net_grid(tr, te, 7000 + 10 * rep + q)
    Pd["신경망(82맵 학습, 기록)"] = prev[f"cv{rep}_net"]
    Pd["Ridge 12특징(82맵 학습, 기록)"] = prev[f"cv{rep}_ridge"]
    return Pd


if __name__ == "__main__":
    from joblib import Parallel, delayed
    R = Parallel(n_jobs=7)(delayed(cv)(r) for r in range(20))
    say()
    say(f"[3] head_cv.py 와 같은 {GRID.sum()}맵 · 같은 fold — 반복 조건 묶음 5-fold × 20회 (조성 입력은 전체로 학습한 PLS)")
    names = list(R[0])
    W = {n: np.array([stat(P_[n][GRID], TT[GRID]) for P_ in R]) for n in names}
    for n in names:
        m = W[n].mean(0); sd = W[n][:, 0].std()
        say(f"  {n:30s} 2배 이내 {m[0]:5.1f} ± {sd:.1f} % (DQ {m[1]:5.1f} · TBZ {m[2]:5.1f} · THI {m[3]:5.1f}) · 3배 {m[4]:5.1f} % · 중앙 배수 {m[5]:.2f}×")
    ref = W["신경망(82맵 학습, 기록)"][:, 0]
    say()
    for n in names:
        if n.startswith("신경망(82"):
            continue
        d = ref - W[n][:, 0]
        say(f"  신경망(82맵) − {n}: {d.mean():+.1f} %p · 20회 중 신경망이 나은 횟수 {(d > 0).sum()} · 같음 {(d == 0).sum()}")
    open(f"{HERE}/calibration_vs_net.txt", "w", encoding="utf-8").write("\n".join(OUT) + "\n")
