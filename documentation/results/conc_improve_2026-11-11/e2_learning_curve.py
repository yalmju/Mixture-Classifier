"""E2. 학습 조건 수에 따른 held-out 오차 (채택 식). 조건 묶음 5-fold × 40: 학습 쪽 51–52 조건에서 n 개만 무작위로 써서 맞춘다.
  오차² = a + b/n 로 맞춰 n → ∞ 의 바닥(a)을 본다 — 학습 혼합물을 더 재면 얼마나 내려가는가.
  실행:  .venv/bin/python3 documentation/results/conc_improve_2026-11-11/e2_learning_curve.py
"""
from common import *
say = Log(f"{HERE}/e2_learning_curve.txt")
X = base_features(); NS = [12, 16, 20, 28, 36, 44, 51]; REPS = 40
res = {n: [] for n in NS}
for k_lv in (8,):
    for n in NS:
        P = np.zeros((REPS, N, 3))
        for r in range(REPS):
            fo = folds(r); rng = np.random.default_rng(1000 + r)
            for k in range(5):
                te = fo == k; tr = np.where(~te)[0]; tr = rng.choice(tr, size=min(n, len(tr)), replace=False)
                P[r][te] = pls_fit_predict(k_lv)(X[tr], LT[tr], X[te])
        res[n] = metrics(P)
        say(line(f"학습 {n:2d} 조건", res[n]))
n = np.array(NS, float); mse = np.array([(res[k]["rmse"] ** 2).mean() for k in NS])
for lo in (0, 2):
    A = np.c_[np.ones(len(n) - lo), 1 / n[lo:]]; a, b = np.linalg.lstsq(A, mse[lo:], rcond=None)[0]
    say(f"\nRMSE² = a + b/n (n ≥ {int(n[lo])}): 바닥 √a = {np.sqrt(max(a, 0)):.3f} dex · "
        + " · ".join(f"n={m}: {np.sqrt(a + b / m):.3f}" for m in (51, 64, 128, 256, 1000)))
# 눌림 기울기도 같이
say("\n기울기(판독 대 참, log–log): " + " · ".join(f"n={k}: {res[k]['slope'].mean():.2f}" for k in NS))
# 잠재변수 수를 n 과 같이 키우면?
say("\n잠재변수 수 (학습 51 조건): " + " · ".join(
    f"LV{k}: {metrics(cv_predict(X, pls_fit_predict(k)))['rmse'].mean():.3f}" for k in (4, 6, 8, 10, 12, 16, 20)))
