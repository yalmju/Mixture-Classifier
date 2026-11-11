"""농도 판독을 더 낫게 할 입력이 있는가 (2026-11-10).

눌림의 원인은 같은 총 농도에서 맵 간 세기 산포(×1.38). 이를 줄일 수 있는 입력을 같은 채점으로 비교한다.
  A  현재 식: 세 밴드 세기 + PLS 몫
  B  + 잉크 2137 cm⁻¹ 밴드 세기 (내부 기준)
  C  밴드 세기를 2137 로 나눈 값 + 몫
  D  + 2137 + PLS 의 BLK 몫
  E  전체 스펙트럼(맵 평균, log1p) → log µM, PLS 회귀
채점: 학습 격자 52맵은 조건 하나씩 빼기, 시험 격자 12맵은 52맵 전체 맞춤을 적용 (모델 기록과 같은 방식).

  실행:  .venv/bin/python3 documentation/results/conc_head_2026-11-10/more_inputs.py
"""
import os, sys, warnings, numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(f"{HERE}/../../..")
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/documentation/scripts")
import dl_model, train_paper_pls as tp
from dataset import load_mixture_list, load_mixture_roles
from real_data import load_map
from sklearn.cross_decomposition import PLSRegression

m = dl_model.load_model(tp.OUT); u = m["uM"]; S3 = list(u["subs"]); subs_all = list(m["subs"])
bands = np.asarray(u["bands_cm"], float)
roles = load_mixture_roles(tp.PURE)
rows = []
for it in load_mixture_list(tp.PURE, filename_truth=True):
    p = tp._here(it)[0]; c = tp.conc_of(p)
    if c is None or not all(v in tp.GRID for v in c):
        continue
    wn, cube, _a, _b = load_map(p)
    wn = np.asarray(wn, float); X = np.asarray(cube, float).reshape(-1, len(wn))
    mask = (wn >= m["lo"]) & (wn <= m["hi"]); Xm = np.clip(X[:, mask], 0, None); ax = wn[mask]
    pk = np.clip(np.asarray(dl_model.apply_model_pixels(m, wn, X), float), 0, None)
    cols = [subs_all.index(s) for s in S3]
    mass = pk[:, cols].sum(1); R = pk[:, cols] / (mass[:, None] + 1e-12)
    sel = np.where(mass >= 0.5)[0]
    sel = sel if len(sel) else np.arange(len(Xm))
    F = dl_model._band_linear_context(Xm[sel], R[sel], np.zeros(len(sel), int), ax, bands)[1][0]
    ink = dl_model._band_signal(Xm[sel], ax, [2137.0]).mean()
    blk = float((pk[sel].sum(1) - mass[sel]).mean() / pk[sel].sum(1).mean())
    rows.append(dict(c=np.asarray(c, float), F=F, ink=np.log10(ink + 1), blk=blk,
                     spec=np.log1p(Xm[sel].mean(0)), test=roles.get(it[0], "train") == "test"))
T = np.array([r["c"] for r in rows]); F = np.array([r["F"] for r in rows]); ink = np.array([r["ink"] for r in rows])
blk = np.array([r["blk"] for r in rows]); SP = np.array([r["spec"] for r in rows]); te = np.array([r["test"] for r in rows])
cond = np.array(["-".join("%g" % v for v in t) for t in T]); tr = ~te
LT = np.log10(T); OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


def lin(Xa, Ya, Xb):
    A = np.c_[Xa, np.ones(len(Xa))]
    return np.c_[Xb, np.ones(len(Xb))] @ np.linalg.lstsq(A, Ya, rcond=None)[0]


def pls(k):
    def f(Xa, Ya, Xb):
        return PLSRegression(n_components=k, scale=False).fit(Xa, Ya).predict(Xb)
    return f


def heldout(X, fit):
    P = np.zeros_like(LT)
    for i in np.where(tr)[0]:
        a = tr & (cond != cond[i])
        P[i] = fit(X[a], LT[a], X[i:i + 1])[0]
    P[te] = fit(X[tr], LT[tr], X[te])
    return 10 ** P


def row(name, P):
    f = np.maximum(P / T, T / P); tot = T.sum(1); lo = tot <= 42
    b = np.polyfit(LT.ravel(), np.log10(P).ravel(), 1)[0]
    e = np.where((T == 24).all(1))[0][0]
    say(f"  {name:44s} | {100 * np.mean(f <= 2):5.1f} | {100 * np.mean(f <= 1.5):5.1f} | {np.median(f):.2f}× | {b:.2f} | "
        f"{100 * np.mean(f[lo] <= 2):5.1f} | {100 * np.mean(f[~lo] <= 2):5.1f} | " + "/".join(f"{v:.0f}" for v in P[e]))


say(f"64 격자 조건 (조건 하나씩 빼기 {tr.sum()} · 시험 {te.sum()}), 성분 {T.size}개")
I3 = F[:, :3]; tot = np.log10(T.sum(1))
r0 = np.log10((10 ** I3 - 1).sum(1)); b0 = np.polyfit(tot, r0, 1); s0 = (r0 - np.polyval(b0, tot)).std()
r1 = r0 - ink; b1 = np.polyfit(tot, r1, 1); s1 = (r1 - np.polyval(b1, tot)).std()
bi = np.polyfit(tot, ink, 1)
say(f"[1] 같은 총 농도에서 맵 간 산포: 세 밴드 합 {s0:.3f} dex (×{10 ** s0:.2f}, 기울기 {b0[0]:.2f}) → 2137 로 나누면 "
    f"{s1:.3f} dex (×{10 ** s1:.2f}, 기울기 {b1[0]:.2f})")
say(f"    2137 세기: 맵 간 산포 {ink.std():.3f} dex (×{10 ** ink.std():.2f}), 총 농도에 대한 기울기 {bi[0]:+.2f}, "
    f"세 밴드 합 잔차와의 상관 r = {np.corrcoef(r0 - np.polyval(b0, tot), ink - np.polyval(bi, tot))[0, 1]:+.2f}")
say("\n[2] 판독 (2배 이내 % | 1.5배 % | 중앙 배수 | log–log 기울기 | 총 ≤42 µM 2배 % | 총 >42 µM 2배 % | 24/24/24 판독)")
g = 10 ** np.mean(LT[tr]); row(f"대조: 상수 {g:.1f} µM", np.full_like(T, g))
row("A 세 밴드 + 몫 (현재)", heldout(F, lin))
row("B A + 2137 밴드", heldout(np.c_[F, ink], lin))
row("C 밴드/2137 + 몫", heldout(np.c_[I3 - ink[:, None], F[:, 3:]], lin))
row("D A + 2137 + BLK 몫", heldout(np.c_[F, ink, blk], lin))
for k in (2, 4, 6, 8):
    row(f"E 전체 스펙트럼 PLS 회귀, 잠재변수 {k}", heldout(SP, pls(k)))
for k in (4, 6):
    row(f"F 전체 스펙트럼 + 몫 PLS, 잠재변수 {k}", heldout(np.c_[SP, 5 * F[:, 3:]], pls(k)))
open(f"{HERE}/more_inputs.txt", "w", encoding="utf-8").write("\n".join(OUT) + "\n")
