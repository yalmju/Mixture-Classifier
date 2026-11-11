"""농도 단계 두 후보를 같은 전처리 · 같은 분할로 반복 비교 (2026-11-11).

  A  세 밴드 ÷ 2137 + 몫  (다변량 선형회귀)          — 11-10 채택 식
  B  전체 스펙트럼 ÷ 2137  PLS 회귀 (잠재변수 8)     — 사용자 선택 후보("94.3 %")

64 격자 조건(학습 52 + 시험 12맵)의 맵 입력을 모델과 똑같이 만든다: dl_model._map_spectra
(포화 보정 → 맵당 100픽셀 → ALS 기준선 제거) → _band_linear_context / _spectrum_context.
조건 묶음 5-fold × 20회. 몫은 저장된 조성 단계의 판독(분할마다 다시 맞추지 않음 — A 에만 들어간다).

  실행:  .venv/bin/python3 documentation/results/conc_head_2026-11-10/specpls_cv.py
"""
import os, sys, warnings, numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(f"{HERE}/../../..")
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/documentation/scripts")
sys.argv = sys.argv[:1]
import dl_model, train_paper_pls as tp
from dataset import load_mixture_list
from real_data import load_map
OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


CACHE = f"{HERE}/specpls_cv_cache.npz"
if os.path.exists(CACHE):
    z = np.load(CACHE); T, XA, XB = z["T"], z["XA"], z["XB"]
else:
    m = dl_model.load_model(tp.OUT); u = m["uM"]; S3 = list(u["subs"]); subs_all = list(m["subs"])
    bands = np.asarray(u["bands_cm"], float); ink = float(u["ink_ref_cm"]); rows = []
    for it in load_mixture_list(tp.PURE, filename_truth=True):
        p = tp._here(it)[0]; c = tp.conc_of(p)
        if c is None or not all(v in tp.GRID for v in c):
            continue
        wn, cube, _a, _b = load_map(p); wn = np.asarray(wn, float)
        mask = (wn >= m["lo"]) & (wn <= m["hi"]); ax = wn[mask]
        X = np.array(dl_model._map_spectra(np.asarray(cube, float).reshape(-1, len(wn)), mask, int(m["px_per_map"]),
                                           baseline_correct=True, sampling=m["pixel_sampling"],
                                           sampling_seed=int(m["sampling_seed"]), map_id=p))
        pk = np.clip(np.asarray(dl_model.apply_model_pixels(m, ax, X), float), 0, None)
        cols = [subs_all.index(s) for s in S3]; mass = pk[:, cols].sum(1); R = pk[:, cols] / (mass[:, None] + 1e-12)
        sel = np.where(mass >= 0.5)[0]; sel = sel if len(sel) else np.arange(len(X)); one = np.zeros(len(sel), int)
        rows.append((np.asarray(c, float), dl_model._band_linear_context(X[sel], R[sel], one, ax, bands, ink_cm=ink)[1][0],
                     dl_model._spectrum_context(X[sel], one, ax, ink)[1][0]))
    T, XA, XB = (np.array([r[i] for r in rows]) for i in range(3))
    np.savez(CACHE, T=T, XA=XA, XB=XB)
LT = np.log10(T); cond = np.array(["-".join("%g" % v for v in t) for t in T]); uc = np.unique(cond)
say(f"맵 {len(T)}개 · 조건 {len(uc)}개 · 성분 {T.size}개")


def const(Xa, Ta, Xb): return np.repeat(np.log10(Ta).mean(0)[None], len(Xb), 0)
def lin(Xa, Ta, Xb): return np.log10(dl_model._predict_band_linear(dl_model._fit_band_linear(Xa, Ta), Xb))
def pls(k): return lambda Xa, Ta, Xb: np.log10(dl_model._predict_band_linear(dl_model._fit_spectrum_pls(Xa, Ta, k), Xb))


V = {"대조: 상수": (XA, const), "A  세 밴드 ÷ 2137 + 몫": (XA, lin), "B  전체 스펙트럼 ÷ 2137 PLS 8": (XB, pls(8)),
     "   (잠재변수 4)": (XB, pls(4)), "   (잠재변수 6)": (XB, pls(6)), "   (잠재변수 10)": (XB, pls(10)), "   (잠재변수 12)": (XB, pls(12))}
acc = {k: [] for k in V}; lev = {k: [] for k in V}; eq = {k: [] for k in V}; i24 = int(np.where((T == 24).all(1))[0][0])
for rep in range(20):
    rng = np.random.default_rng(rep); fold = dict(zip(uc, rng.permutation(len(uc)) % 5)); fo = np.array([fold[c] for c in cond])
    for k, (X, fit) in V.items():
        P = np.zeros_like(LT)
        for f in range(5):
            P[fo == f] = fit(X[fo != f], T[fo != f], X[fo == f])
        r = np.abs(P - LT); eq[k].append(10 ** P[i24])
        acc[k].append((100 * np.mean(r <= np.log10(2) + 1e-9), 100 * np.mean(r <= np.log10(1.5)), 10 ** np.median(r),
                       np.sqrt(np.mean(r ** 2)), np.polyfit(LT.ravel(), P.ravel(), 1)[0],
                       *(100 * np.mean(r[:, j] <= np.log10(2) + 1e-9) for j in range(3))))
        lev[k].append([100 * np.mean(r[T == v] <= np.log10(2) + 1e-9) for v in (3, 6, 12, 24)])
say("\n조건 묶음 5-fold × 20회   (2배 이내 % ± sd | 1.5배 % | 중앙 배수 | RMSE dex | 기울기 | DQ·TBZ·THI 2배 % | "
    "참 3·6·12·24 µM 성분 2배 % | 24/24/24 판독 µM | A 보다 나은 횟수)")
base = np.array(acc["A  세 밴드 ÷ 2137 + 몫"])
for k in V:
    a = np.array(acc[k]); e = np.median(np.array(eq[k]), 0); L = np.array(lev[k]).mean(0)
    say(f"  {k:30s} | {a[:, 0].mean():5.1f} ± {a[:, 0].std():.1f} | {a[:, 1].mean():5.1f} | {a[:, 2].mean():.2f}× | {a[:, 3].mean():.3f} | "
        f"{a[:, 4].mean():.2f} | {a[:, 5].mean():.0f}·{a[:, 6].mean():.0f}·{a[:, 7].mean():.0f} | "
        f"{L[0]:.0f}·{L[1]:.0f}·{L[2]:.0f}·{L[3]:.0f} | {e[0]:.0f}/{e[1]:.0f}/{e[2]:.0f} | "
        f"2배 {int((a[:, 0] > base[:, 0]).sum())}/20 · 1.5배 {int((a[:, 1] > base[:, 1]).sum())}/20 · RMSE {int((a[:, 3] < base[:, 3]).sum())}/20")
open(f"{HERE}/specpls_cv.txt", "w", encoding="utf-8").write("\n".join(OUT) + "\n")
