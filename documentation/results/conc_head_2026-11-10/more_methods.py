"""농도 판독을 더 낫게 할 다른 방법 (2026-11-10, more_inputs.py 다음).

[1] 한 맵 안에서 더 할 수 있는 것 (64 조건 묶음 5-fold × 20회)
    C   밴드÷2137 + 몫 (맵 평균끼리 나눔)          — more_inputs.py 의 권장안
    C1  픽셀마다 나눈 뒤 log 평균                    C2  픽셀마다 나눈 값의 중앙값
    Q   C + 픽셀 분포(90 % 분위수)                   N2  C + 제곱·곱 항 (ridge)
    K   C 입력으로 커널 회귀(RBF, 안쪽 분할로 조절)   G   전체 스펙트럼÷2137 PLS 8
[2] 맵을 두 장 쓰는 방법 — 원액과 2배 희석을 같이 읽기.  격자에 성분 모두 절반인 조건이 있는 27 조건
    (예 24/24/24 ↔ 12/12/12). 두 조건을 모두 뺀 맞춤으로 각각 읽고 log 평균: C = sqrt(읽음(C) · 2·읽음(C/2)).

  실행:  .venv/bin/python3 documentation/results/conc_head_2026-11-10/more_methods.py
"""
import os, sys, warnings, numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(f"{HERE}/../../..")
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/documentation/scripts")
import dl_model, train_paper_pls as tp
from dataset import load_mixture_list
from real_data import load_map
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV

CACHE = f"{HERE}/more_methods_cache.npz"
if os.path.exists(CACHE):
    z = np.load(CACHE); T, C0, C1, C2, Q90, SH, SPn = (z[k] for k in ("T", "C0", "C1", "C2", "Q90", "SH", "SPn"))
else:
    m = dl_model.load_model(tp.OUT); u = m["uM"]; S3 = list(u["subs"]); subs_all = list(m["subs"])
    bands = list(np.asarray(u["bands_cm"], float)); rows = []
    for it in load_mixture_list(tp.PURE, filename_truth=True):
        p = tp._here(it)[0]; c = tp.conc_of(p)
        if c is None or not all(v in tp.GRID for v in c):
            continue
        wn, cube, _a, _b = load_map(p)
        wn = np.asarray(wn, float); X = np.asarray(cube, float).reshape(-1, len(wn))
        mask = (wn >= m["lo"]) & (wn <= m["hi"]); Xm = np.clip(X[:, mask], 0, None); ax = wn[mask]
        pk = np.clip(np.asarray(dl_model.apply_model_pixels(m, wn, X), float), 0, None)
        cols = [subs_all.index(s) for s in S3]; mass = pk[:, cols].sum(1); R = pk[:, cols] / (mass[:, None] + 1e-12)
        sel = np.where(mass >= 0.5)[0]; sel = sel if len(sel) else np.arange(len(Xm))
        B = dl_model._band_signal(Xm[sel], ax, bands + [2137.0]); w = R[sel] + 1e-6
        ratio = (B[:, :3] + 1) / (B[:, 3:4] + 1)
        sh = R[sel].mean(0); sh = sh / sh.sum()
        rows.append((np.asarray(c, float),
                     np.log10((w * B[:, :3]).sum(0) / w.sum(0) + 1) - np.log10(B[:, 3].mean() + 1),
                     (w * np.log10(ratio)).sum(0) / w.sum(0),
                     np.log10(np.median(ratio, 0)), np.log10(np.percentile(ratio, 90, 0)), sh[:2],
                     np.log1p(Xm[sel].mean(0)) - np.log1p(B[:, 3].mean())))
    T, C0, C1, C2, Q90, SH, SPn = (np.array([r[i] for r in rows]) for i in range(7))
    np.savez(CACHE, T=T, C0=C0, C1=C1, C2=C2, Q90=Q90, SH=SH, SPn=SPn)
LT = np.log10(T); cond = np.array(["-".join("%g" % v for v in t) for t in T]); tot = T.sum(1); OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


def lin(Xa, Ya, Xb):
    return np.c_[Xb, np.ones(len(Xb))] @ np.linalg.lstsq(np.c_[Xa, np.ones(len(Xa))], Ya, rcond=None)[0]


def pls8(Xa, Ya, Xb):
    return PLSRegression(n_components=8, scale=False).fit(Xa, Ya).predict(Xb)


def quad(X):
    i, j = np.triu_indices(X.shape[1]); return np.c_[X, X[:, i] * X[:, j]]


def ridge2(Xa, Ya, Xb):
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 2, 12))).fit(quad(Xa), Ya).predict(quad(Xb))


def krr(Xa, Ya, Xb):
    g = GridSearchCV(make_pipeline(StandardScaler(), KernelRidge(kernel="rbf")),
                     {"kernelridge__alpha": [0.01, 0.1, 1.0], "kernelridge__gamma": [0.02, 0.05, 0.1, 0.2]}, cv=5)
    return g.fit(Xa, Ya).predict(Xb)


XC = np.c_[C0, SH]
V = {"C  밴드÷2137 + 몫 (맵 평균)": (XC, lin), "C1 픽셀마다 나눈 뒤 log 평균": (np.c_[C1, SH], lin),
     "C2 픽셀마다 나눈 값의 중앙값": (np.c_[C2, SH], lin), "Q  C + 90 % 분위수": (np.c_[C0, Q90, SH], lin),
     "N2 C + 제곱·곱 항 (ridge)": (XC, ridge2), "K  커널 회귀 (RBF)": (XC, krr), "G  전체 스펙트럼÷2137 PLS 8": (SPn, pls8)}
uc = np.unique(cond); acc = {k: [] for k in V}
for rep in range(20):
    rng = np.random.default_rng(rep); fold = dict(zip(uc, rng.permutation(len(uc)) % 5)); fo = np.array([fold[c] for c in cond])
    for k, (X, fit) in V.items():
        P = np.zeros_like(LT)
        for f in range(5):
            P[fo == f] = fit(X[fo != f], LT[fo != f], X[fo == f])
        r = np.abs(P - LT)
        acc[k].append((100 * np.mean(r <= np.log10(2)), 100 * np.mean(r <= np.log10(1.5)), 10 ** np.median(r),
                       np.polyfit(LT.ravel(), P.ravel(), 1)[0], 100 * np.mean(r[tot > 42] <= np.log10(2))))
say("[1] 한 맵 안에서 — 64 조건 묶음 5-fold × 20회 (2배 이내 % ± sd | 1.5배 % | 중앙 배수 | 기울기 | 총 >42 µM 2배 % | C 보다 나은 횟수)")
base = np.array(acc["C  밴드÷2137 + 몫 (맵 평균)"])[:, 0]
for k in V:
    a = np.array(acc[k])
    say(f"  {k:32s} | {a[:, 0].mean():5.1f} ± {a[:, 0].std():.1f} | {a[:, 1].mean():5.1f} | {a[:, 2].mean():.2f}× | {a[:, 3].mean():.2f} | "
        f"{a[:, 4].mean():5.1f} | {int((a[:, 0] > base).sum())}/20")

say("\n[2] 맵 두 장 — 원액 + 2배 희석 (두 조건을 모두 뺀 맞춤)")
idx = {c: i for i, c in enumerate(cond)}


def score(name, P, Tt):
    r = np.abs(np.log10(P) - np.log10(Tt))
    b = np.polyfit(np.log10(Tt).ravel(), np.log10(P).ravel(), 1)[0]
    say(f"  {name:44s} | {100 * np.mean(r <= np.log10(2)):5.1f} | {100 * np.mean(r <= np.log10(1.5)):5.1f} | {10 ** np.median(r):.2f}× | {b:.2f}")


for label, (X, fit) in (("C 밴드÷2137 + 몫", (XC, lin)), ("G 전체 스펙트럼÷2137 PLS 8", (SPn, pls8))):
    for div, tag in ((2, "2배 희석"), (4, "4배 희석")):
        one, two, tru, eq = [], [], [], None
        for i, t in enumerate(T):
            key = "-".join("%g" % v for v in t / div)
            if key not in idx:
                continue
            j = idx[key]; a = np.ones(len(T), bool); a[[i, j]] = False
            p = fit(X[a], LT[a], X[[i, j]])
            one.append(10 ** p[0]); two.append(10 ** (0.5 * (p[0] + p[1] + np.log10(div)))); tru.append(t)
            if (t == 24).all():
                eq = (one[-1], two[-1])
        one, two, tru = map(np.array, (one, two, tru))
        say(f" {label} · {tag}: {len(tru)} 조건 (2배 이내 % | 1.5배 % | 중앙 배수 | 기울기)")
        score("   원액 한 장", one, tru); score(f"   원액 + {tag} 두 장", two, tru)
        hi = tru.sum(1) > 42
        if hi.sum() > 2:
            score(f"   그 중 총 >42 µM ({hi.sum()} 조건) 한 장", one[hi], tru[hi]); score("   같은 조건 두 장", two[hi], tru[hi])
        if eq is not None:
            say("   24/24/24: 한 장 " + "/".join(f"{v:.0f}" for v in eq[0]) + " → 두 장 " + "/".join(f"{v:.0f}" for v in eq[1]) + " µM")
open(f"{HERE}/more_methods.txt", "w", encoding="utf-8").write("\n".join(OUT) + "\n")

# ---------------------------------------------------------------- [3] 두 장 읽기의 대조·성분별·세 장
say("\n[3] 두 장 읽기 — 대조와 성분별 (2배 희석 27 조건, 두 조건 모두 뺀 맞춤)")
say("    (2배 이내 % | 1.5배 % | √2배 이내 % = 가장 가까운 격자 수준 | 중앙 배수 | RMSE dex)")


def const(Xa, Ya, Xb):
    return np.repeat(Ya.mean(0)[None], len(Xb), 0)


def pairs(X, fit, n=2):
    E1, E2, Tt = [], [], []
    for i, t in enumerate(T):
        js = [idx.get("-".join("%g" % v for v in t / 2 ** k)) for k in range(n)]
        if any(j is None for j in js):
            continue
        a = np.ones(len(T), bool); a[js] = False
        p = fit(X[a], LT[a], X[js]) + np.log10(2.0) * np.arange(n)[:, None]
        E1.append(p[0] - LT[i]); E2.append(p.mean(0) - LT[i]); Tt.append(t)
    return np.array(E1), np.array(E2), np.array(Tt)


def line(name, E):
    r = np.abs(E)
    say(f"  {name:40s} | {100 * np.mean(r <= np.log10(2) + 1e-9):5.1f} | {100 * np.mean(r <= np.log10(1.5)):5.1f} | "
        f"{100 * np.mean(r <= 0.5 * np.log10(2)):5.1f} | {10 ** np.median(r):.2f}× | {np.sqrt(np.mean(E ** 2)):.3f}")


for label, X, fit in (("대조: 상수", XC, const), ("대조: 몫만", SH, lin), ("C 밴드÷2137 + 몫", XC, lin), ("G 전체 스펙트럼÷2137 PLS 8", SPn, pls8)):
    E1, E2, Tt = pairs(X, fit)
    line(label + " · 한 장", E1); line(label + " · 두 장", E2)
    if label.startswith("C "):
        for k, s in enumerate(("DQ", "TBZ", "THI")):
            line(f"     {s} 한 장", E1[:, k]); line(f"     {s} 두 장", E2[:, k])
        say("     한 장 오차와 희석 장 오차의 상관 r = %.2f, 한 장 평균 오차 %+.3f dex, 희석 장(×2) 평균 오차 %+.3f dex"
            % (np.corrcoef(E1.ravel(), (2 * E2 - E1).ravel())[0, 1], E1.mean(), (2 * E2 - E1).mean()))
say("\n 세 장 (원액 + 2배 + 4배 희석, 8 조건)")
for label, X, fit in (("대조: 상수", XC, const), ("C 밴드÷2137 + 몫", XC, lin), ("G 전체 스펙트럼÷2137 PLS 8", SPn, pls8)):
    E1, E3, Tt = pairs(X, fit, 3)
    line(label + " · 한 장", E1); line(label + " · 세 장", E3)
open(f"{HERE}/more_methods.txt", "w", encoding="utf-8").write("\n".join(OUT) + "\n")
