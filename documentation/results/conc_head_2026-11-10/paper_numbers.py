"""논문 모델 농도 단계(지문 영역 500–2000 cm⁻¹ ÷ 2137 PLS 회귀, 2026-11-11 확정)의 본문 수치.

  모델  pls_composition_261110.dlm    log10 C = PLS 8 [ ln(1 + 맵 평균 스펙트럼) − ln(1 + I_ink) ],  500–2000 cm⁻¹
  비교  …_spec2500.dlm (같은 식, 500–2500 cm⁻¹ 전체) · …_bandink.dlm (세 밴드 ÷ 2137 + 몫, 11-10 식) · …_bandonly.dlm (세 밴드 + 몫)

학습 52 격자 맵은 조건 하나씩 빼기(모델 기록과 같은 값인지 [0] 에서 확인), 시험 12 격자 맵과 격자 밖 맵은 저장 모델 적용.
스펙트럼은 학습과 같은 전처리(dl_model._map_spectra: 포화 보정 → 맵당 100픽셀 → ALS 기준선 제거)를 거쳐 넣는다.
농도는 성분별로만 센다 — 세 물질이 다른 화합물이라 총 농도로 묶은 수치는 내지 않는다(사용자 결정 11-11).

  실행:  .venv/bin/python3 documentation/results/conc_head_2026-11-10/paper_numbers.py
"""
import os, sys, warnings, numpy as np
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


NEW = dl_model.load_model(tp.OUT); u = NEW["uM"]
assert u["kind"] == "spectrum_pls_uM_v1" and u.get("ink_ref_cm"), "논문 모델의 농도 단계가 전체 스펙트럼 PLS 가 아니다"
LV = int(u["n_components"]); INK = float(u["ink_ref_cm"]); RANGE = u.get("spectrum_range_cm")


def like_training(m, wn, X, map_id):
    mask = (wn >= m["lo"]) & (wn <= m["hi"])
    return wn[mask], np.array(dl_model._map_spectra(X, mask, int(m["px_per_map"]), baseline_correct=not m.get("nnls_screen"),
                                                    sampling=m["pixel_sampling"], sampling_seed=int(m["sampling_seed"]), map_id=map_id))


ev = u["loo_eval"]
roles = load_mixture_roles(tp.PURE)
maps = [(p, np.asarray(t, float), np.asarray(q, float), "loo" if ok else "out") for p, t, q, ok in
        zip(ev["paths"], ev["true_uM"], ev["pred_uM"], ev["in_fit_range"])]
for it in load_mixture_list(tp.PURE, filename_truth=True):
    if roles.get(it[0], "train") == "test":
        p = tp._here(it)[0]; c = tp.conc_of(p)
        if c and all(v in tp.GRID for v in c):
            maps.append((p, np.asarray(c, float), None, "test"))
R = []
for p, t, q, sp in maps:
    wn, cube, _a, _b = load_map(p); wn = np.asarray(wn, float)
    ax, X = like_training(NEW, wn, np.asarray(cube, float).reshape(-1, len(wn)), p)
    um, subs, meta = dl_model.apply_uM_pixels(NEW, ax, X, return_meta=True)
    R.append(dict(p=p, T=t, rec=q, sp=sp, A=np.array([meta["map_uM"][s] for s in subs]), meta=meta,
                  F=dl_model._spectrum_context(X, np.zeros(len(X), int), ax, INK)[1][0]))
sp = np.array([r["sp"] for r in R]); T = np.array([r["T"] for r in R]); F = np.array([r["F"] for r in R]); A = np.array([r["A"] for r in R])
loo, tst, out = sp == "loo", sp == "test", sp == "out"
KEEP = dl_model._spectrum_keep(ax, RANGE)          # 농도 회귀가 보는 스펙트럼 범위 (모델에 적힌 값)
grid = loo | tst
iL = np.where(loo)[0]


def fold(p, t):
    return np.maximum(p / t, t / p)


def st(P, Tt):
    q = fold(P, Tt)
    return 100 * (q <= 2).mean(), 100 * (q <= 1.5).mean(), np.median(q)


def ols(x, y):
    b, a = np.polyfit(x, y, 1); r = y - (a + b * x)
    return b, a, 1 - r.var() / y.var()


Pl = np.full((len(T), 3), np.nan)
for i in iL:
    tr = iL[iL != i]
    Pl[i] = dl_model._predict_band_linear(dl_model._fit_spectrum_pls(F[tr], T[tr], LV, KEEP), F[[i]])[0]
rec = np.array([r["rec"] for r in R if r["sp"] == "loo"])
Wf = dl_model._fit_spectrum_pls(F[iL], T[iL], LV, KEEP)
insample = dl_model._predict_band_linear(Wf, F[iL])
say("[0] 확인 — 학습과 같은 전처리로 다시 맞춘 값이 모델 기록과 같은가")
say(f"  조건 하나씩 빼기 판독 대 모델 기록: 최대 차 {np.abs(np.log10(Pl[iL]) - np.log10(rec)).max():.4f} dex · "
    f"회귀 계수 최대 차 {np.abs(Wf - np.asarray(u['coef'], float)).max():.5f} · "
    f"저장 모델 적용 대 같은 맵의 맞춤값(학습 52맵): 최대 차 {np.abs(np.log10(A[iL]) - np.log10(insample)).max():.4f} dex · "
    f"격자 밖 {out.sum()}맵 적용 대 기록: 최대 차 {np.abs(np.log10(A[out]) - np.log10(np.array([r['rec'] for r in R if r['sp'] == 'out']))).max():.4f} dex")

P = np.full((len(T), 3), np.nan); P[iL] = rec; P[tst] = A[tst]       # 학습 52 = 조건 하나씩 빼기 기록, 시험 12 = 저장 모델 적용
gm = 10 ** np.log10(T[loo]).mean()

say(f"\n[1] 학습 격자 {loo.sum()}맵 · 성분 {3 * loo.sum()}개, 조건 하나씩 빼기   (2배 이내 % | 1.5배 % | 중앙 배수 | MAE µM | DQ·TBZ·THI 2배 %)")
tab = {"채택: 500–2000 cm⁻¹ ÷ 2137 PLS": P[loo]}
for name, tag in (("500–2500 cm⁻¹ 전체 ÷ 2137 PLS", "_spec2500"), ("11-10 식: 세 밴드 ÷ 2137 + 몫", "_bandink"), ("세 밴드 + 몫 (나누지 않음)", "_bandonly")):
    e = dl_model.load_model(tp.OUT.replace(".dlm", f"{tag}.dlm"))["uM"]["loo_eval"]
    d = {os.path.basename(str(p)): np.asarray(q, float) for p, q in zip(e["paths"], e["pred_uM"])}
    tab[name] = np.array([d[os.path.basename(str(r["p"]))] for r in R if r["sp"] == "loo"])
tab["대조: 상수 %.1f µM" % gm] = np.full((loo.sum(), 3), gm)
for n, Q in tab.items():
    a, b, c = st(Q, T[loo]); pc = 100 * (fold(Q, T[loo]) <= 2).mean(0)
    say(f"  {n:28s} | {a:5.1f} | {b:5.1f} | {c:.2f}× | {np.abs(Q - T[loo]).mean():.1f} | {pc[0]:.1f} · {pc[1]:.1f} · {pc[2]:.1f}")

say("\n[2] 참 농도별 2배 이내 % / 판독 중앙값 µM (세 성분 합쳐서)")
for nm, g in (("학습 52맵", loo), ("64 조건", grid)):
    cells = []
    for v in (3, 6, 12, 24):
        k = g[:, None] & (T == v)
        cells.append(f"{v} µM {100 * (fold(P[k], T[k]) <= 2).mean():.1f} % / {np.median(P[k]):.1f} (10–90 %: {np.percentile(P[k], 10):.1f}–{np.percentile(P[k], 90):.1f})")
    say(f"  {nm}: " + " · ".join(cells))
say("  성분별 판독 중앙값, 참 3 · 6 · 12 · 24 µM (64 조건): " + " | ".join(
    f"{s} " + " · ".join(f"{np.median(P[grid & (T[:, j] == v), j]):.1f}" for v in (3, 6, 12, 24)) for j, s in enumerate(S)))
say("  성분별 2배 이내 %, 참 3 · 6 · 12 · 24 µM (64 조건): " + " | ".join(
    f"{s} " + " · ".join(f"{100 * (fold(P[grid & (T[:, j] == v), j], v) <= 2).mean():.0f}" for v in (3, 6, 12, 24)) for j, s in enumerate(S)))

say(f"\n[3] 64 격자 조건 (조건 빼기 {loo.sum()} + 시험 {tst.sum()})")
Tg, Pg = T[grid], P[grid]; f = fold(Pg, Tg); f0 = fold(np.full_like(Tg, gm), Tg)
a, b, c = st(Pg, Tg)
say(f"  전체: 2배 이내 {a:.1f} % ({int((f <= 2).sum())}/{f.size}) · 1.5배 {b:.1f} % · 중앙 {c:.2f}× · RMSE {np.sqrt(np.mean(np.log10(f) ** 2)):.3f} dex | "
    f"대조 상수 {100 * (f0 <= 2).mean():.1f} % · {100 * (f0 <= 1.5).mean():.1f} % · {np.median(f0):.2f}×")
say("  성분별 2배 · 1.5배 %: " + " | ".join(f"{s} {100 * (f[:, j] <= 2).mean():.1f} · {100 * (f[:, j] <= 1.5).mean():.1f}" for j, s in enumerate(S)))
say(f"  세 성분 모두 2배 이내인 조건: {int((f <= 2).all(1).sum())}/{len(f)} (대조 상수 {int((f0 <= 2).all(1).sum())}/{len(f)})")
a, b, c = st(P[tst], T[tst]); f0t = fold(np.full_like(T[tst], gm), T[tst])
say(f"  시험 12맵(저장 모델 적용): 2배 이내 {a:.1f} % · 1.5배 {b:.1f} % · 중앙 {c:.2f}× | 대조 상수 {100 * (f0t <= 2).mean():.1f} %")
say("  판독 대 참 log–log 기울기 · r²: " + " · ".join("%s %.2f · %.2f" % ((s,) + ols(np.log10(Tg[:, j]), np.log10(Pg[:, j]))[::2]) for j, s in enumerate(S))
    + " | 세 성분 합침 %.2f · %.2f" % ols(np.log10(Tg).ravel(), np.log10(Pg).ravel())[::2])
for c in (3, 6, 12, 24):
    i = np.where((Tg == c).all(1))[0][0]
    say(f"  {c}/{c}/{c} µM → " + " / ".join(f"{v:.1f}" for v in Pg[i]) + " µM")
say("  어느 성분이 가장 많은지(참값에 최다 성분이 하나인 조건): " + (lambda k: f"{int((Pg[k].argmax(1) == Tg[k].argmax(1)).sum())}/{int(k.sum())}")(
    np.array([(t == t.max()).sum() == 1 for t in Tg])))

say(f"\n[4] 격자 밖 {out.sum()}맵 (구간 위·이원, 저장 모델 적용) — 있는 성분의 2배 이내 %, 참 농도별")
To, Po = T[out], A[out]
say("  " + " · ".join(f"{v:g}: {100 * (fold(Po[To == v], To[To == v]) <= 2).mean():.0f} % (n={int((To == v).sum())})" for v in sorted(set(To[To > 0].tolist()))))
k = (To >= 33)
say(f"  성분 33 µM 이상 전체: 2배 이내 {100 * (fold(Po[k], To[k]) <= 2).mean():.1f} % (n={int(k.sum())}) · 없는 성분(참 0)을 읽는 값 중앙 {np.median(Po[To == 0]):.1f} / 최대 {Po[To == 0].max():.1f} µM (n={int((To == 0).sum())})")
for r in R:
    if r["sp"] == "out" and len(set(r["T"].tolist())) == 1:
        say(f"  등몰 {r['T'][0]:g} µM → " + " / ".join(f"{v:.0f}" for v in r["A"]) + f" µM · 입력 |z| 최대 {r['meta']['feature_zmax']:.1f} · 적용영역 표시 {'뜸' if r['meta']['batch_mismatch'] else '안 뜸'}")

say("\n[5] 회귀 계수가 큰 자리 (맞춤 52맵, 성분별 양의 계수 상위 3곳 / 음의 계수 상위 2곳, cm⁻¹)")
wn0, cube0, _a, _b = load_map(R[0]["p"]); ax = np.asarray(wn0, float); ax = ax[(ax >= NEW["lo"]) & (ax <= NEW["hi"])]
W = np.asarray(u["coef"], float)[:-1]


def peaks(w, n):
    idx = [i for i in range(1, len(w) - 1) if w[i] > 0 and w[i] >= w[i - 1] and w[i] >= w[i + 1]]
    idx = sorted(idx, key=lambda i: -w[i]); keep = []
    for i in idx:
        if all(abs(ax[i] - ax[j]) > 25 for j in keep):
            keep.append(i)
        if len(keep) == n:
            break
    return " · ".join(f"{ax[i]:.0f}" for i in keep)


for j, s in enumerate(S):
    say(f"  {s:3s}  + {peaks(W[:, j], 3)}   − {peaks(-W[:, j], 2)}")
say("  (표지 밴드: DQ 1570 · TBZ 1270 · THI 1367, 잉크 2137 cm⁻¹)")
open(f"{HERE}/paper_numbers.txt", "w", encoding="utf-8").write("\n".join(OUT) + "\n")
