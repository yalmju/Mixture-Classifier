"""(이전 판, 2026-11-10) 세 밴드 ÷ 2137 + 몫 농도 단계의 수치 — 11-11 에 전체 스펙트럼 ÷ 2137 PLS 로 바뀌어
지금 논문 모델의 수치는 paper_numbers.py 에 있다. 이 스크립트는 _bandink 모델을 읽는다.

  모델  pls_composition_261110_bandink.dlm       log10 C_j = Σ a_jk·log10((I_k+1)/(I_ink+1)) + b_j·s + c_j
  비교  pls_composition_261110_bandonly.dlm   이전 식(2137 로 안 나눔)

학습 52 격자 맵은 조건 하나씩 빼기(모델 기록과 같은 값인지 [0] 에서 확인), 시험 12 격자 맵과 격자 밖 맵은 저장 모델 적용.
스펙트럼은 학습과 같은 전처리(dl_model._map_spectra: 포화 보정 → 맵당 100픽셀 → ALS 기준선 제거)를 거쳐 넣는다.
전처리 없이 넣으면 밴드 입력이 학습 때와 달라진다(평균 0.05 dex) — [0] 의 확인이 이것을 잡는다.

  실행:  .venv/bin/python3 documentation/results/conc_head_2026-11-10/paper_numbers_bandink.py
"""
import os, sys, csv, warnings, numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(f"{HERE}/../../..")
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/documentation/scripts")
sys.argv = sys.argv[:1]
import dl_model, train_paper_pls as tp
from dataset import load_mixture_list, load_mixture_roles
from real_data import load_map
OUT = []
S = ["DQ", "TBZ", "THI"]


def say(s=""):
    print(s, flush=True); OUT.append(s)


NEW = dl_model.load_model(tp.OUT.replace(".dlm", "_bandink.dlm"))
OLD = dl_model.load_model(tp.OUT.replace(".dlm", "_bandonly.dlm"))
assert NEW["uM"].get("ink_ref_cm") and not OLD["uM"].get("ink_ref_cm")


def like_training(m, wn, X, map_id):
    mask = (wn >= m["lo"]) & (wn <= m["hi"])
    return np.array(dl_model._map_spectra(X, mask, int(m["px_per_map"]), baseline_correct=not m.get("nnls_screen"),
                                          sampling=m["pixel_sampling"], sampling_seed=int(m["sampling_seed"]), map_id=map_id))


def feats(m, wn, X):
    u = m["uM"]
    um, subs, meta = dl_model.apply_uM_pixels(m, wn, X, return_meta=True)
    F = np.asarray(u["mu"], float) + np.asarray(meta["feature_z"], float) * np.asarray(u["sd"], float)
    return F, np.array([meta["map_uM"][s] for s in subs]), meta


ev = NEW["uM"]["loo_eval"]
roles = load_mixture_roles(tp.PURE)
maps = [(p, np.asarray(t, float), np.asarray(q, float), "loo" if ok else "out") for p, t, q, ok in
        zip(ev["paths"], ev["true_uM"], ev["pred_uM"], ev["in_fit_range"])]
for it in load_mixture_list(tp.PURE, filename_truth=True):
    if roles.get(it[0], "train") == "test":
        p = tp._here(it)[0]; c = tp.conc_of(p)
        if c and all(v in tp.GRID for v in c):
            maps.append((p, np.asarray(c, float), None, "test"))
R = []                                   # true, record read, split, F_new, apply_new, meta_new, F_old, apply_old
for p, t, q, sp in maps:
    wn, cube, _a, _b = load_map(p); wn = np.asarray(wn, float)
    X = like_training(NEW, wn, np.asarray(cube, float).reshape(-1, len(wn)), p)
    fn, an, mn = feats(NEW, wn, X); fo, ao, _ = feats(OLD, wn, X)
    R.append(dict(p=p, T=t, rec=q, sp=sp, Fn=fn, An=an, meta=mn, Fo=fo, Ao=ao))
sp = np.array([r["sp"] for r in R]); T = np.array([r["T"] for r in R])
Fn = np.array([r["Fn"] for r in R]); Fo = np.array([r["Fo"] for r in R]); An = np.array([r["An"] for r in R])
loo, tst, out = sp == "loo", sp == "test", sp == "out"
grid = loo | tst
LT = np.log10(np.clip(T, 0.05, None))


def fit(X, y):
    return np.linalg.lstsq(np.c_[X, np.ones(len(X))], y, rcond=None)[0]


def app(w, X):
    return 10 ** (np.c_[X, np.ones(len(X))] @ w)


def loo_pred(X, idx):
    """idx 의 맵을 하나씩 빼고(조건 = 맵 하나) 나머지 idx 로 맞춤."""
    P = np.full((len(T), 3), np.nan)
    for i in idx:
        tr = idx[idx != i]
        for j in range(3):
            P[i, j] = app(fit(X[tr], LT[tr, j]), X[[i]])[0]
    return P


def fold(p, t):
    return np.maximum(p / t, t / p)


def st(P, Tt):
    q = fold(P, Tt)
    return 100 * (q <= 2).mean(), 100 * (q <= 1.5).mean(), np.median(q)


iL = np.where(loo)[0]
Pl = loo_pred(Fn, iL)
rec = np.array([r["rec"] for r in R if r["sp"] == "loo"])
w_full = np.array([fit(Fn[iL], LT[iL, j]) for j in range(3)]).T
say("[0] 확인 — 학습과 같은 전처리로 다시 맞춘 값이 모델 기록과 같은가")
say(f"  조건 하나씩 빼기 판독 대 모델 기록: 최대 차 {np.abs(np.log10(Pl[iL]) - np.log10(rec)).max():.4f} dex · "
    f"계수 최대 차 {np.abs(w_full - np.asarray(NEW['uM']['coef'], float)).max():.4f} · "
    f"격자 밖 {out.sum()}맵 저장 모델 적용 대 기록: 최대 차 {np.abs(np.log10(An[out]) - np.log10(np.array([r['rec'] for r in R if r['sp'] == 'out']))).max():.4f} dex")

# 판독값 모음: 학습 52 = 모델 기록(조건 하나씩 빼기), 시험 12 = 저장 모델 적용
P = np.full((len(T), 3), np.nan); P[iL] = rec; P[tst] = An[tst]
gm = 10 ** LT[loo].mean()

say(f"\n[1] 학습 격자 {loo.sum()}맵 · 성분 {3 * loo.sum()}개, 조건 하나씩 빼기   (2배 이내 % | 1.5배 % | 중앙 배수 | MAE µM | DQ·TBZ·THI 2배 %)")
tab = {"채택: 세 밴드 ÷ 2137 + 몫": P[loo],
       "세 밴드 ÷ 2137 만": loo_pred(Fn[:, :3], iL)[loo],
       "몫만": loo_pred(Fn[:, 3:], iL)[loo],
       "이전: 세 밴드 + 몫": loo_pred(Fo, iL)[loo],
       "세 밴드만 (나누지 않음)": loo_pred(Fo[:, :3], iL)[loo],
       "대조: 상수 %.1f µM" % gm: np.full((loo.sum(), 3), gm)}
for n, Q in tab.items():
    a, b, c = st(Q, T[loo]); pc = 100 * (fold(Q, T[loo]) <= 2).mean(0)
    say(f"  {n:26s} | {a:5.1f} | {b:5.1f} | {c:.2f}× | {np.abs(Q - T[loo]).mean():.1f} | {pc[0]:.1f} · {pc[1]:.1f} · {pc[2]:.1f}")

say("\n[2] 참 농도별 2배 이내 % / 판독 중앙값 µM")
for nm, g in (("학습 52맵", loo), ("64 조건", grid)):
    cells = []
    for v in (3, 6, 12, 24):
        k = g[:, None] & (T == v)
        cells.append(f"{v} µM {100 * (fold(P[k], T[k]) <= 2).mean():.1f} % / {np.median(P[k]):.1f} (10–90 %: {np.percentile(P[k], 10):.1f}–{np.percentile(P[k], 90):.1f})")
    say(f"  {nm}: " + " · ".join(cells))
say("  성분별 판독 범위(참 3 → 24 µM 의 판독 중앙값, 학습 52맵): " + " · ".join(
    f"{s} {np.median(P[loo & (T[:, j] == 3), j]):.1f} → {np.median(P[loo & (T[:, j] == 24), j]):.1f}" for j, s in enumerate(S)))

say(f"\n[3] 64 격자 조건 (조건 빼기 {loo.sum()} + 시험 {tst.sum()})")
Tg, Pg = T[grid], P[grid]; Ct = Tg.sum(1); f = fold(Pg, Tg); f0 = fold(np.full_like(Tg, gm), Tg)
a, b, c = st(Pg, Tg)
say(f"  전체: 2배 이내 {a:.1f} % · 1.5배 {b:.1f} % · 중앙 {c:.2f}× · RMSE {np.sqrt(np.mean(np.log10(f) ** 2)):.3f} dex | 대조 상수 {100 * (f0 <= 2).mean():.1f} %"
    f" | 성분별 2배 " + " · ".join(f"{100 * (f[:, j] <= 2).mean():.1f}" for j in range(3)))
a, b, c = st(P[tst], T[tst]); a0 = 100 * (fold(np.full_like(T[tst], gm), T[tst]) <= 2).mean()
say(f"  시험 12맵(저장 모델 적용): 2배 이내 {a:.1f} % · 1.5배 {b:.1f} % · 중앙 {c:.2f}× | 대조 상수 {a0:.1f} %")
for lo, hi in ((9, 42), (45, 72)):
    g = (Ct >= lo) & (Ct <= hi)
    say(f"  총 {lo}–{hi} µM | 조건 {g.sum():2d} | 2배 이내 {100 * (f[g] <= 2).mean():.1f} % · 1.5배 {100 * (f[g] <= 1.5).mean():.1f} % · 중앙 {np.median(f[g]):.2f}× | 대조 {100 * (f0[g] <= 2).mean():.1f} %")


def ols(x, y):
    b, a = np.polyfit(x, y, 1); r = y - (a + b * x)
    return b, a, 1 - r.var() / y.var()


say("  판독 대 참 log–log 기울기 · r²: " + " · ".join("%s %.2f · %.2f" % ((s,) + ols(np.log10(Tg[:, j]), np.log10(Pg[:, j]))[::2]) for j, s in enumerate(S)))
bt, at, r2t = ols(np.log10(Ct), np.log10(Pg.sum(1)))
bs, _, r2s = ols(np.log10(Tg / Tg.sum(1, keepdims=True)).ravel(), np.log10(Pg / Pg.sum(1, keepdims=True)).ravel())
ba, _, r2a = ols(np.log10(Tg).ravel(), np.log10(Pg).ravel())
say(f"    세 성분 합침 {ba:.2f} · {r2a:.2f} | 총합 {bt:.2f} · {r2t:.2f} | 판독 농도끼리의 몫 {bs:.2f} · {r2s:.2f}")
say("  총 농도 구간별 판독 총합/참 총합 (실측 | 당겨짐만 있을 때 기대): " + " · ".join(
    f"{lo}–{hi}: {np.median(Pg[g].sum(1) / Ct[g]):.2f} | {np.median(10 ** (at + bt * np.log10(Ct[g])) / Ct[g]):.2f}"
    for lo, hi in ((9, 18), (21, 30), (33, 42), (45, 54), (57, 72)) for g in [(Ct >= lo) & (Ct <= hi)]))
k = np.where(Tg == 24); oth = Tg[k[0]].sum(1) - 24
say("  24 µM 성분의 판독 중앙값 — 나머지 두 성분 합에 따라: " + " · ".join(
    f"{lo}–{hi} µM: {np.median(Pg[k][(oth >= lo) & (oth <= hi)]):.1f} (2배 이내 {100 * np.mean(Pg[k][(oth >= lo) & (oth <= hi)] >= 12):.0f} %)"
    for lo, hi in ((6, 15), (18, 30), (36, 48))))
for c in (3, 6, 12, 24):
    i = np.where((Tg == c).all(1))[0][0]
    say(f"  {c}/{c}/{c} µM → " + " / ".join(f"{v:.1f}" for v in Pg[i]) + " µM · 판독 농도의 몫 " + " / ".join(f"{100 * v:.0f}" for v in Pg[i] / Pg[i].sum()) + " %")
L = np.log10(Pg); Lt = np.log10(Tg); D = np.empty_like(L)
for j in range(3):
    b, a, _ = ols(Lt[:, j], L[:, j]); D[:, j] = (L[:, j] - a) / b
fd = 10 ** np.abs(D - Lt); g24 = Tg == 24
say(f"  눌림을 펴면(기울기 1 로 되돌림, 진단용): 2배 이내 {100 * (f <= 2).mean():.1f} → {100 * (fd <= 2).mean():.1f} % · log 오차 "
    f"{np.sqrt(np.mean((L - Lt) ** 2)):.2f} → {np.sqrt(np.mean((D - Lt) ** 2)):.2f} dex · 24 µM 중앙 {np.median(Pg[g24]):.1f} → {np.median(10 ** D[g24]):.1f} µM "
    f"(10–90 %: {np.percentile(10 ** D[g24], 10):.1f}–{np.percentile(10 ** D[g24], 90):.1f})")

say("\n[4] 묶음 5-fold × 20회 (학습 52맵, 조성 입력은 전체로 학습한 PLS)   2배 이내 % ± 표준편차 | 1.5배 % | 중앙 배수")


def cv(X, rep, k=5):
    rng = np.random.default_rng(1000 + rep); o = rng.permutation(iL); Q = np.full((len(T), 3), np.nan)
    for q in range(k):
        te = o[q::k]; tr = np.setdiff1d(iL, te)
        for j in range(3):
            Q[te, j] = app(fit(X[tr], LT[tr, j]), X[te])
    return st(Q[loo], T[loo])


for n, X in (("세 밴드 ÷ 2137 + 몫", Fn), ("이전: 세 밴드 + 몫", Fo)):
    W = np.array([cv(X, r) for r in range(20)])
    say(f"  {n:20s} | {W[:, 0].mean():.1f} ± {W[:, 0].std():.1f} | {W[:, 1].mean():.1f} | {W[:, 2].mean():.2f}×")

say("\n[5] 조성까지 held-out (64 조건: 몫은 그 조건을 빼고 학습한 PLS — fig4 heldout_maps.csv, 조건 하나씩 빼기)")
rows = [r for r in csv.DictReader(open(f"{REPO}/documentation/results/fig4_paper_pls_2026-11-10/heldout_maps.csv", encoding="utf-8-sig")) if r["grid64"] == "1"]
H = {tuple(int(float(r[f"{s}_true_uM"])) for s in S): np.array([float(r[f"{s}_PLS"]) for s in S]) for r in rows}
iG = np.where(grid)[0]
SH = np.zeros((len(T), 2))
for i in iG:
    h = H[tuple(int(v) for v in T[i])]; SH[i] = (h / h.sum())[:2]
for n, B in (("세 밴드 ÷ 2137 + 몫", Fn[:, :3]), ("이전: 세 밴드 + 몫", Fo[:, :3])):
    Q = loo_pred(np.c_[B, SH], iG)[grid]; a, b, c = st(Q, Tg)
    say(f"  {n:20s} | 2배 이내 {a:.1f} % · 1.5배 {b:.1f} % · 중앙 {c:.2f}×")
a, b, c = st(loo_pred(SH, iG)[grid], Tg)
say(f"  {'몫만':20s} | 2배 이내 {a:.1f} % · 1.5배 {b:.1f} % · 중앙 {c:.2f}×   (대조 상수 {100 * (f0 <= 2).mean():.1f} %)")

say(f"\n[6] 격자 밖 {out.sum()}맵 (구간 위·이원, 저장 모델 적용) — 있는 성분의 2배 이내 %, 참 농도별")
To, Po = T[out], An[out]
say("  " + " · ".join(f"{v:g}: {100 * (fold(Po[To == v], To[To == v]) <= 2).mean():.0f} % (n={int((To == v).sum())})" for v in sorted(set(To[To > 0].tolist()))))
k = (To >= 33)
say(f"  성분 33 µM 이상 전체: 2배 이내 {100 * (fold(Po[k], To[k]) <= 2).mean():.1f} % (n={int(k.sum())}) · 없는 성분(참 0)을 읽는 값 중앙 {np.median(Po[To == 0]):.1f} / 최대 {Po[To == 0].max():.1f} µM (n={int((To == 0).sum())})")
for r in R:
    if r["sp"] == "out" and len(set(r["T"].tolist())) == 1:
        say(f"  등몰 {r['T'][0]:g} µM → " + " / ".join(f"{v:.0f}" for v in r["An"]) + f" µM · 입력 |z| 최대 {r['meta']['feature_zmax']:.1f} · 적용영역 표시 {'뜸' if r['meta']['batch_mismatch'] else '안 뜸'}")

say("\n[7] 채택 식의 계수 (맞춤 52맵)   log10(I_DQ/I_ink) · log10(I_TBZ/I_ink) · log10(I_THI/I_ink) · 몫_DQ · 몫_TBZ · 절편")
for j, s in enumerate(S):
    say(f"  log10 C_{s:3s} = " + "  ".join(f"{v:+.2f}" for v in np.asarray(NEW["uM"]["coef"], float)[:, j]))
say("  자기 농도에 대한 입력의 log–log 기울기 · r² (64 조건): " + " · ".join(
    "%s %+.2f · %.2f" % ((s,) + ols(np.log10(Tg[:, j]), Fn[grid][:, j])[::2]) for j, s in enumerate(S))
    + " | 나누기 전: " + " · ".join("%s %+.2f · %.2f" % ((s,) + ols(np.log10(Tg[:, j]), Fo[grid][:, j])[::2]) for j, s in enumerate(S)))
open(f"{HERE}/paper_numbers_bandink.txt", "w", encoding="utf-8").write("\n".join(OUT) + "\n")
