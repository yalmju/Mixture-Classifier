"""전체 스펙트럼 회귀로 농도를 읽는 경우 — PLS 회귀와 무작위 숲(random forest) (2026-11-10).
more_methods.py 의 캐시(맵 평균 log 스펙트럼 ÷ 2137, 밴드÷2137, 몫)를 그대로 쓴다. 64 조건 묶음 5-fold × 20회.
  실행:  .venv/bin/python3 documentation/results/conc_head_2026-11-10/full_spectrum_rf.py
"""
import os, warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(f"{HERE}/more_methods_cache.npz"); T, C0, SH, SPn = z["T"], z["C0"], z["SH"], z["SPn"]
LT = np.log10(T); cond = np.array(["-".join("%g" % v for v in t) for t in T]); tot = T.sum(1); XC = np.c_[C0, SH]; OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


def const(Xa, Ya, Xb): return np.repeat(Ya.mean(0)[None], len(Xb), 0)
def lin(Xa, Ya, Xb): return np.c_[Xb, np.ones(len(Xb))] @ np.linalg.lstsq(np.c_[Xa, np.ones(len(Xa))], Ya, rcond=None)[0]
def pls8(Xa, Ya, Xb): return PLSRegression(n_components=8, scale=False).fit(Xa, Ya).predict(Xb)
def rf(Xa, Ya, Xb): return RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1).fit(Xa, Ya).predict(Xb)


V = {"대조: 상수": (XC, const), "C  밴드÷2137 + 몫 (선형)": (XC, lin), "G  전체 스펙트럼÷2137 PLS 8": (SPn, pls8),
     "RF 전체 스펙트럼÷2137 무작위 숲": (SPn, rf), "RF 밴드÷2137 + 몫 무작위 숲": (XC, rf)}
uc = np.unique(cond); acc = {k: [] for k in V}; eq = {k: [] for k in V}; i24 = int(np.where((T == 24).all(1))[0][0])
for rep in range(20):
    rng = np.random.default_rng(rep); fold = dict(zip(uc, rng.permutation(len(uc)) % 5)); fo = np.array([fold[c] for c in cond])
    for k, (X, fit) in V.items():
        P = np.zeros_like(LT)
        for f in range(5):
            P[fo == f] = fit(X[fo != f], LT[fo != f], X[fo == f])
        r = np.abs(P - LT); eq[k].append(10 ** P[i24])
        acc[k].append((100 * np.mean(r <= np.log10(2)), 100 * np.mean(r <= np.log10(1.5)), 10 ** np.median(r),
                       np.polyfit(LT.ravel(), P.ravel(), 1)[0], 100 * np.mean(r[tot > 42] <= np.log10(2)),
                       *(100 * np.mean(r[:, j] <= np.log10(2)) for j in range(3))))
say("64 조건 묶음 5-fold × 20회 (2배 이내 % ± sd | 1.5배 % | 중앙 배수 | 기울기 | 총 >42 µM 2배 % | DQ·TBZ·THI 2배 % | 24/24/24 판독 µM | C 보다 나은 횟수)")
base = np.array(acc["C  밴드÷2137 + 몫 (선형)"])[:, 0]
for k in V:
    a = np.array(acc[k]); e = np.median(np.array(eq[k]), 0)
    say(f"  {k:30s} | {a[:, 0].mean():5.1f} ± {a[:, 0].std():.1f} | {a[:, 1].mean():5.1f} | {a[:, 2].mean():.2f}× | {a[:, 3].mean():.2f} | "
        f"{a[:, 4].mean():5.1f} | {a[:, 5].mean():.0f}·{a[:, 6].mean():.0f}·{a[:, 7].mean():.0f} | "
        f"{e[0]:.0f}/{e[1]:.0f}/{e[2]:.0f} | {int((a[:, 0] > base).sum())}/20")
open(f"{HERE}/full_spectrum_rf.txt", "w", encoding="utf-8").write("\n".join(OUT) + "\n")
