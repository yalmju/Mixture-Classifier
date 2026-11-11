"""E5. 같은 조건을 따로 한 번 더 조제해 찍은 맵이 있는 유일한 자료로 — 한 번 더 찍어 평균하면 얼마나 좋아지는가.
  자료: 08-08 조제 격자(Baseline260808, 성분당 9–72 µM, 64 조건) + 같은 조건 재조제 12 맵(DQ9-sus-relabeled, DQ 9 µM 줄).
  ※ 최종 조제(08-14, 3–24 µM)가 아니다. 논문 수치로 쓰지 않는다 — 오차가 어디서 오는지 보는 용도.
  식은 채택 식과 같다(같은 전처리 → 맵 평균 스펙트럼 ÷ 2137 → PLS 8). 08-08 격자 안에서 조건 하나씩 빼고 맞춘다
  (그 조건의 두 맵 모두 학습에서 뺌).
  실행:  .venv/bin/python3 documentation/results/conc_improve_2026-11-11/e5_replicate_pairs.py
"""
import os, sys, re, glob, warnings, numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(f"{HERE}/../../..")
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/documentation/scripts"); sys.argv = sys.argv[:1]
import dl_model, train_paper_pls as tp
from real_data import load_map
from sklearn.cross_decomposition import PLSRegression
ROOT = os.path.dirname(tp.FINAL); CACHE = f"{HERE}/e5_cache.npz"; INK = 2137.0
NAME = re.compile(r"DQ(\d+)-TB(\d+)-THI?(\d+)")
rows, OUT = [], []


def say(s=""):
    print(s, flush=True); OUT.append(s); open(f"{HERE}/e5_replicate_pairs.txt", "w").write("\n".join(OUT) + "\n")


if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True); T, X, rep, Cp = z["T"], z["X"], z["rep"], z["Cp"]
else:
    m = dl_model.load_model(tp.OUT)
    files = [(p, 0) for d in ("DQ9", "DQ18", "DQ36", "DQ72") for p in sorted(glob.glob(f"{ROOT}/Baseline260808/{d}/DQ*.csv"))]
    files += [(p, 1) for p in sorted(glob.glob(f"{ROOT}/mis/DQ9-sus-relabeled/DQ*.csv"))]
    T, X, rep, Cp = [], [], [], []
    for p, r in files:
        g = NAME.match(os.path.basename(p)); c = [float(g.group(i)) for i in (1, 2, 3)]
        wn, cube, _a, _b = load_map(p); wn = np.asarray(wn, float); mask = (wn >= m["lo"]) & (wn <= m["hi"]); ax = wn[mask]
        px = np.array(dl_model._map_spectra(np.asarray(cube, float).reshape(-1, len(wn)), mask, int(m["px_per_map"]), baseline_correct=True,
                                            sampling=m["pixel_sampling"], sampling_seed=int(m["sampling_seed"]), map_id=p))
        X.append(dl_model._spectrum_context(px, np.zeros(len(px), int), ax, INK)[1][0])
        Cp.append(np.full(3, np.nan))
        T.append(c); rep.append(r); print(len(T), os.path.basename(p), c, r, flush=True)
    T, X, rep, Cp = np.array(T), np.array(X), np.array(rep), np.array(Cp)
    np.savez_compressed(CACHE, T=T, X=X, rep=rep, Cp=Cp)
LT = np.log10(T); cond = np.array(["-".join("%g" % v for v in t) for t in T])
main = rep == 0; say(f"08-08 격자 {int(main.sum())} 맵 · 조건 {len(np.unique(cond[main]))} · 재조제 {int((~main).sum())} 맵")
dup = [c for c in np.unique(cond[main]) if (cond[main] == c).sum() > 1]
if dup:
    say(f"  같은 조건 파일이 둘 이상인 격자 조건: {dup}")
L2, L15 = np.log10(2) + 1e-9, np.log10(1.5) + 1e-9


def fit_pred(tr, te, k=8):
    return np.asarray(PLSRegression(n_components=k, scale=False).fit(X[tr], LT[tr]).predict(X[te]), float).reshape(len(te), -1)


P = np.zeros_like(LT)
for c in np.unique(cond):
    te = np.where(cond == c)[0]; tr = np.where(cond != c)[0]; P[te] = fit_pred(tr, te)
E = P - LT


def summ(e):
    a = np.abs(e).ravel(); return f"2배 {100 * (a <= L2).mean():5.1f} % · 1.5배 {100 * (a <= L15).mean():5.1f} % · 중앙 {10 ** np.median(a):.2f}× · RMSE {np.sqrt((e ** 2).mean()):.3f} dex"


say(f"\n08-08 격자 전체(조건 하나씩 빼기, 맵 {int(main.sum())}): {summ(E[main])}")
cm = LT[main].mean(0); say(f"  대조 상수: {summ(np.repeat(cm[None], int(main.sum()), 0) - LT[main])}")
say("  성분별 2배 이내: " + " · ".join(f"{s} {100 * (np.abs(E[main][:, j]) <= L2).mean():.1f}" for j, s in enumerate(("DQ", "TBZ", "THI"))))
# 재조제 쌍
ia, ib = [], []
for i in np.where(~main)[0]:
    j = np.where(main & (cond == cond[i]))[0]
    if len(j):
        ia.append(j[0]); ib.append(i)
ia, ib = np.array(ia), np.array(ib); say(f"\n재조제 쌍 {len(ia)} 개 (DQ 9 µM, TBZ · THI 9–72 µM) — 성분 {len(ia) * 3} 개")
e1, e2 = E[ia], E[ib]; em = (e1 + e2) / 2; dd = e1 - e2
say(f"  한 맵(처음 조제)      : {summ(e1)}")
say(f"  한 맵(재조제)         : {summ(e2)}")
say(f"  한 맵(둘을 합쳐 24 맵) : {summ(np.r_[e1, e2])}")
say(f"  두 맵 판독의 기하평균 : {summ(em)}")
say(f"  두 맵 판독끼리의 차이 : 중앙 {10 ** np.median(np.abs(dd)):.2f}× · 표준편차 {dd.std():.3f} dex → 맵 하나의 무작위 부분 {dd.std() / np.sqrt(2):.3f} dex")
vt = (np.r_[e1, e2] ** 2).mean(); vr = (dd ** 2).mean() / 2; vb = vt - vr
say(f"  제곱오차 분해(한 맵 {np.sqrt(vt):.3f} dex): 다시 조제하면 바뀌는 부분 {100 * vr / vt:.0f} % ({np.sqrt(vr):.3f} dex) · 조건에 붙어 안 바뀌는 부분 {100 * max(vb, 0) / vt:.0f} % ({np.sqrt(max(vb, 0)):.3f} dex)")
say(f"  → n 번 찍어 평균할 때 예상 RMSE: " + " · ".join(f"n={n}: {np.sqrt(max(vb, 0) + vr / n):.3f}" for n in (1, 2, 3, 5)))
for j, s in enumerate(("DQ", "TBZ", "THI")):
    say(f"  {s}: 한 맵 RMSE {np.sqrt((np.r_[e1[:, j], e2[:, j]] ** 2).mean()):.3f} · 두 맵 평균 {np.sqrt((em[:, j] ** 2).mean()):.3f} · 두 판독 차이 SD {dd[:, j].std():.3f} · 오차 상관 r {np.corrcoef(e1[:, j], e2[:, j])[0, 1]:+.2f}")
say(f"  두 맵 오차의 상관(성분 합쳐): r = {np.corrcoef(e1.ravel(), e2.ravel())[0, 1]:+.2f}")
say("\n  쌍별 판독 µM (참 → 처음 / 재조제)")
for a, b in zip(ia, ib):
    say("   " + cond[a].ljust(10) + "  " + "  ".join(f"{s} {T[a, j]:g} → {10 ** P[a, j]:5.1f} / {10 ** P[b, j]:5.1f}" for j, s in enumerate(("DQ", "TBZ", "THI"))))
# 두 판독의 차이가 세 성분에 같이 실리는가(전달량) 따로 실리는가
dc = dd.mean(1, keepdims=True); ds = dd - dc
say(f"\n  두 판독 차이의 제곱 분해: 세 성분 공통 {100 * (dc ** 2).mean() / (dd ** 2).mean():.0f} % (SD {dc.std():.3f} dex) · 성분 고유 {100 * (ds ** 2).mean() / (dd ** 2).mean():.0f} %")
say("  차이의 성분 간 상관: " + " · ".join(f"{a}–{b} {np.corrcoef(dd[:, i], dd[:, j])[0, 1]:+.2f}" for (i, a), (j, b) in (((0, 'DQ'), (1, 'TBZ')), ((0, 'DQ'), (2, 'THI')), ((1, 'TBZ'), (2, 'THI')))))
# 몫(조성)으로 바꿔 보면: 판독 농도 → 몫, 두 맵의 몫 차이
sh = lambda p: 10 ** p / (10 ** p).sum(1, keepdims=True)
ts = T[ia] / T[ia].sum(1, keepdims=True)
say(f"  (참고) 판독 농도를 몫으로 바꾼 조성 오차: 한 맵 {100 * np.abs(np.r_[sh(P[ia]), sh(P[ib])] - np.r_[ts, ts]).mean():.1f} %p · 두 맵 평균 {100 * np.abs(sh((P[ia] + P[ib]) / 2) - ts).mean():.1f} %p")
# 쌍이 12 개뿐이라 — 쌍을 다시 뽑아(부트스트랩 2000 회) 두 맵 평균의 이득이 얼마나 흔들리는지
rng = np.random.default_rng(0); rat, r2 = [], []
for _ in range(2000):
    k = rng.integers(0, len(ia), len(ia)); a = np.sqrt((np.r_[e1[k], e2[k]] ** 2).mean()); b = np.sqrt((em[k] ** 2).mean()); rat.append(b / a); r2.append(b)
say(f"  부트스트랩(쌍 12 개): 두 맵 평균 RMSE {np.mean(r2):.3f} (90 % 구간 {np.quantile(r2, 0.05):.3f}–{np.quantile(r2, 0.95):.3f}) · 한 맵 대비 비 {np.mean(rat):.2f} ({np.quantile(rat, 0.05):.2f}–{np.quantile(rat, 0.95):.2f}); 오차가 완전히 독립이면 0.71")
