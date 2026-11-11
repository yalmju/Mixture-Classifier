"""E4. E3 에서 꾸준히 나았던 것들을 ① 겹쳐 보고 ② 범위·변환을 훑어 봉우리가 넓은지 보고 ③ 고르는 일까지 학습 쪽
안에서 하는 중첩 CV 로 다시 잰다(고른 뒤의 수치가 부풀지 않았는지).  ④ 조성 단계 몫을 섞는 안 ⑤ 정방향(고전) 검량.
  실행:  .venv/bin/python3 documentation/results/conc_improve_2026-11-11/e4_combine_nested.py
"""
import csv
from common import *
say = Log(f"{HERE}/e4_combine_nested.txt")


def feats(power=None, lo=500, hi=2500, bright=None):
    """power None → ln(1 + 평균/…) 채택 식 · 아니면 (평균 / 잉크)^power.  bright=(q0, q1) → 밝기 분위 구간 픽셀만."""
    X = []
    for i in range(N):
        s = sel_of(i)
        if bright:
            t = PA[i][s].sum(1); q = np.quantile(t, bright); s = s[(t >= q[0]) & (t <= q[1])]
        m = np.clip(PA[i][s], 0, None).mean(0); k = band(PA[i][s], INK).mean()
        X.append(np.log1p(m) - np.log1p(k) if power is None else (m / k) ** power)
    return np.array(X)[:, (ax >= lo) & (ax <= hi)]


X0 = feats(); P0 = cv_predict(X0, pls_fit_predict(8)); ref = metrics(P0)
say(line("대조: 상수", metrics(cv_predict(X0, const_fit_predict)))); say(line("기준: 스펙트럼 ÷ 2137, PLS 8 (500–2500)", ref))

say("\n── ① 범위 훑기 (ln, PLS 8) — 위 끝")
for hi in (1500, 1600, 1700, 1800, 1900, 2000, 2050, 2250, 2500):
    say(line(f"500–{hi}", metrics(cv_predict(feats(hi=hi), pls_fit_predict(8))), ref))
say("── 아래 끝 (위 끝 1800)")
for lo in (500, 600, 700, 800, 1000):
    say(line(f"{lo}–1800", metrics(cv_predict(feats(lo=lo, hi=1800), pls_fit_predict(8))), ref))
say("\n── ② 변환 훑기 (500–1800, PLS 8)")
for pw in (None, 0.75, 0.5, 0.33, 0.25):
    say(line("ln(1+x)" if pw is None else f"(평균/잉크)^{pw}", metrics(cv_predict(feats(power=pw, hi=1800), pls_fit_predict(8))), ref))
say("\n── ③ 겹치기")
C = {"기준": (X0, 8), "500–1800": (feats(hi=1800), 8), "제곱근": (feats(power=0.5), 8), "500–1800 · 제곱근": (feats(power=0.5, hi=1800), 8),
     "500–1800 · PLS 12": (feats(hi=1800), 12), "500–1800 · 제곱근 · PLS 12": (feats(power=0.5, hi=1800), 12),
     "500–1800 · 밝은 절반": (feats(hi=1800, bright=(0.5, 1.0)), 8), "500–1800 · 제곱근 · 밝은 절반": (feats(power=0.5, hi=1800, bright=(0.5, 1.0)), 8),
     "500–1700": (feats(hi=1700), 8), "500–2000": (feats(hi=2000), 8), "500–1800 · 세제곱근": (feats(power=0.33, hi=1800), 8),
     "500–1800 · PLS 6": (feats(hi=1800), 6), "500–1800 · PLS 10": (feats(hi=1800), 10)}
PR_ = {}
for n, (X, k) in C.items():
    PR_[n] = cv_predict(X, pls_fit_predict(k))
    if n != "기준":
        say(line(n, metrics(PR_[n]), ref))

# 밴드 식(세 밴드 ÷ 2137 + 몫)과 평균 — 밴드 식 입력은 11-10 기록에서
zc = np.load(f"{HERE}/../conc_head_2026-11-10/specpls_cv_cache.npz", allow_pickle=True)
key = {tuple(np.round(t, 3)): i for i, t in enumerate(zc["T"])}; XA = zc["XA"][[key[tuple(np.round(t, 3))] for t in T]]


def linfit(Xa, Ya, Xb):
    A = np.c_[Xa, np.ones(len(Xa))]; return np.c_[Xb, np.ones(len(Xb))] @ np.linalg.lstsq(A, Ya, rcond=None)[0]


PB = cv_predict(XA, linfit)
say(line("밴드 식 단독", metrics(PB), ref))
for n in ("기준", "500–1800", "500–1800 · 제곱근"):
    say(line(f"({n}) + 밴드 식 평균", metrics((PR_[n] + PB) / 2), ref))
say(line("(500–1800) + (500–1800 · 제곱근) 평균", metrics((PR_["500–1800"] + PR_["500–1800 · 제곱근"]) / 2), ref))

say("\n── ④ 중첩 CV: 후보 고르기를 학습 쪽 안(조건 묶음 4-fold, RMSE)에서")
names = list(C)


def nested(pool, reps=20):
    out = np.zeros((reps, N, 3)); pick = {n: 0 for n in pool}
    for r in range(reps):
        fo = folds(r)
        for q in range(5):
            tr = np.where(fo != q)[0]; te = np.where(fo == q)[0]
            uc = np.unique(cond[tr]); g = dict(zip(uc, np.random.default_rng(1000 * r + q).permutation(len(uc)) % 4)); fi = np.array([g[c] for c in cond[tr]])
            best, bn = np.inf, None
            for n in pool:
                X, k = C[n]; e = 0.0
                for u in range(4):
                    a, b = tr[fi != u], tr[fi == u]; e += ((pls_fit_predict(k)(X[a], LT[a], X[b]) - LT[b]) ** 2).sum()
                if e < best:
                    best, bn = e, n
            X, k = C[bn]; out[r, te] = pls_fit_predict(k)(X[tr], LT[tr], X[te]); pick[bn] += 1
    return out, pick


for label, pool in (("전체 후보", names), ("기준 대 500–1800", ["기준", "500–1800"]), ("기준 대 500–1800 · 제곱근", ["기준", "500–1800 · 제곱근"])):
    Pn, pick = nested(pool); say(line(f"중첩: {label}", metrics(Pn), ref))
    say("    고른 횟수(100 fold): " + " · ".join(f"{n} {c}" for n, c in sorted(pick.items(), key=lambda kv: -kv[1]) if c))

say("\n── ⑤ 조성 단계의 held-out 몫을 섞기 (몫은 Fig 4 기록 — 그 맵을 안 본 PLS 조성 모델의 판독)")
rows = {(float(r["DQ_true_uM"]), float(r["TBZ_true_uM"]), float(r["THI_true_uM"])): [float(r[f"{s}_PLS"]) for s in SUB]
        for r in csv.DictReader(open(f"{HERE}/../fig4_paper_pls_2026-11-10/heldout_maps.csv")) if r["grid64"] == "1"}
S = np.array([rows[tuple(float(v) for v in t)] for t in T]); S = np.clip(S, 1e-3, None); S = S / S.sum(1, keepdims=True)
for n in ("기준", "500–1800"):
    P = PR_[n]; tot = np.log10((10 ** P).sum(2, keepdims=True)); Pc = np.log10(S)[None] + tot
    say(line(f"({n}) 몫 × 판독 합", metrics(Pc), ref))
    for w in (0.25, 0.5, 0.75):
        say(line(f"({n}) 농도 판독 {w:.2f} + 몫 × 합 {1 - w:.2f}", metrics(w * P + (1 - w) * Pc), ref))

say("\n── ⑥ 정방향(고전) 검량: PLS 점수 = A + B·log C 로 맞추고 뒤집어 읽기 — 눌림은 없애지만 산포가 커지는가")


def classical(k):
    def f(Xa, Ya, Xb):
        p = PLSRegression(n_components=k, scale=False).fit(Xa, Ya); Sa, Sb = p.transform(Xa), p.transform(Xb)
        A = np.c_[Ya, np.ones(len(Ya))]; W = np.linalg.lstsq(A, Sa, rcond=None)[0]; B, a = W[:-1].T, W[-1]
        Rz = Sa - A @ W; V = np.cov(Rz.T) + 1e-9 * np.eye(k); Vi = np.linalg.inv(V)
        return np.linalg.solve(B.T @ Vi @ B, B.T @ Vi @ (Sb - a).T).T
    return f


for n in ("기준", "500–1800"):
    for k in (8, 12):
        say(line(f"({n}) 정방향 검량, 점수 {k}", metrics(cv_predict(C[n][0], classical(k))), ref))
np.savez_compressed(f"{HERE}/e4_predictions.npz", names=np.array(names, object), P=np.array([PR_[n] for n in names]), PB=PB, LT=LT, cond=cond)
