"""more_inputs.py 의 확인: (1) 2137 이 실제 봉우리인가, 배경 높이인가 (2) 반복 분할에서도 차이가 유지되는가
(3) 전체 스펙트럼 회귀의 잠재변수 수를 안쪽 분할로 고르면.   실행: .venv/bin/python3 <이 파일>"""
import os, sys, warnings, numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(f"{HERE}/../../..")
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/documentation/scripts")
import dl_model, train_paper_pls as tp
from dataset import load_mixture_list
from real_data import load_map
from sklearn.cross_decomposition import PLSRegression

m = dl_model.load_model(tp.OUT); u = m["uM"]; S3 = list(u["subs"]); subs_all = list(m["subs"])
bands = np.asarray(u["bands_cm"], float); rows = []
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
    F = dl_model._band_linear_context(Xm[sel], R[sel], np.zeros(len(sel), int), ax, bands)[1][0]
    mean = Xm[sel].mean(0)
    pk_top = dl_model._band_signal(Xm[sel], ax, [2137.0]).mean()
    side = mean[((ax > 2060) & (ax < 2090)) | ((ax > 2190) & (ax < 2220))].mean()
    flat = mean[(ax > 1800) & (ax < 1900)].mean()
    rows.append(dict(c=np.asarray(c, float), F=F, top=pk_top, side=side, flat=flat, spec=np.log1p(mean)))
T = np.array([r["c"] for r in rows]); LT = np.log10(T); F = np.array([r["F"] for r in rows]); SP = np.array([r["spec"] for r in rows])
top = np.array([r["top"] for r in rows]); side = np.array([r["side"] for r in rows]); flat = np.array([r["flat"] for r in rows])
cond = np.array(["-".join("%g" % v for v in t) for t in T]); OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


say(f"[1] 2137 cm⁻¹: 봉우리 꼭대기 중앙 {np.median(top):.0f} · 양옆 배경(2060–2090, 2190–2220) {np.median(side):.0f} · "
    f"봉우리 높이/배경 중앙 {np.median((top - side) / side):.1f}배 · 1800–1900 평탄부 {np.median(flat):.0f}")
ink_top = np.log10(top + 1); ink_h = np.log10(np.clip(top - side, 1, None)); bg = np.log10(flat + 1)


def lin(Xa, Ya, Xb):
    return np.c_[Xb, np.ones(len(Xb))] @ np.linalg.lstsq(np.c_[Xa, np.ones(len(Xa))], Ya, rcond=None)[0]


def pls(k):
    return lambda Xa, Ya, Xb: PLSRegression(n_components=k, scale=False).fit(Xa, Ya).predict(Xb)


def pls_inner(Xa, Ya, Xb, ks=(2, 4, 6, 8, 10)):
    rng = np.random.default_rng(0); fold = rng.integers(0, 5, len(Xa)); best = None
    for k in ks:
        e = 0.0
        for f in range(5):
            a = fold != f
            e += ((PLSRegression(n_components=k, scale=False).fit(Xa[a], Ya[a]).predict(Xa[~a]) - Ya[~a]) ** 2).sum()
        if best is None or e < best[0]:
            best = (e, k)
    return PLSRegression(n_components=best[1], scale=False).fit(Xa, Ya).predict(Xb)


V = {"A 세 밴드 + 몫 (현재)": (F, lin),
     "B A + 2137 꼭대기": (np.c_[F, ink_top], lin),
     "B' A + 2137 봉우리 높이(배경 뺌)": (np.c_[F, ink_h], lin),
     "B'' A + 1800–1900 배경 높이(잉크 아님)": (np.c_[F, bg], lin),
     "C 밴드/2137 + 몫": (np.c_[F[:, :3] - ink_top[:, None], F[:, 3:]], lin),
     "E 전체 스펙트럼 PLS 8": (SP, pls(8)),
     "E' 전체 스펙트럼 PLS, 잠재변수 안쪽 분할로 선택": (SP, pls_inner),
     "G 전체 스펙트럼/2137 PLS 8": (SP - np.log1p(top)[:, None], pls(8))}
uc = np.unique(cond); res = {k: [] for k in V}; slope = {k: [] for k in V}; hi = {k: [] for k in V}; f15 = {k: [] for k in V}
tot = T.sum(1)
for rep in range(20):
    rng = np.random.default_rng(rep); fold = dict(zip(uc, rng.permutation(len(uc)) % 5)); fo = np.array([fold[c] for c in cond])
    for k, (X, fit) in V.items():
        P = np.zeros_like(LT)
        for f in range(5):
            P[fo == f] = fit(X[fo != f], LT[fo != f], X[fo == f])
        r = np.abs(P - LT)
        res[k].append(100 * np.mean(r <= np.log10(2))); f15[k].append(100 * np.mean(r <= np.log10(1.5)))
        slope[k].append(np.polyfit(LT.ravel(), P.ravel(), 1)[0]); hi[k].append(100 * np.mean(r[tot > 42] <= np.log10(2)))
say("\n[2] 64 조건 묶음 5-fold × 20회 (2배 이내 % 평균 ± sd | 1.5배 % | 기울기 | 총 >42 µM 2배 % | 현재 식보다 나은 횟수)")
base = np.array(res["A 세 밴드 + 몫 (현재)"])
for k in V:
    a = np.array(res[k])
    say(f"  {k:40s} | {a.mean():5.1f} ± {a.std():.1f} | {np.mean(f15[k]):5.1f} | {np.mean(slope[k]):.2f} | {np.mean(hi[k]):5.1f} | {int((a > base).sum())}/20")
open(f"{HERE}/more_inputs_check.txt", "w", encoding="utf-8").write("\n".join(OUT) + "\n")
