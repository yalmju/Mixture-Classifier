"""농도 판독이 눌리는 이유 — 세기가 농도를 얼마나 따라가는가 (2026-11-10).

64 격자 조건(학습 52맵은 조건 하나씩 빼기 판독, 시험 12맵은 저장 모델 그대로 적용).
맵 단위 입력은 모델과 같은 정의: 세 특성 밴드 세기 I_k(조성 가중 맵 평균), PLS 몫.

  실행:  .venv/bin/python3 documentation/results/conc_head_2026-11-10/compression_cause.py
"""
import os, sys, warnings, numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(f"{HERE}/../../..")
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/documentation/scripts")
import dl_model, train_paper_pls as tp
from dataset import load_mixture_list, load_mixture_roles
from real_data import load_map

m = dl_model.load_model(tp.OUT.replace(".dlm", "_bandonly.dlm")); u = m["uM"]; S3 = list(u["subs"])  # 이전 식(세 밴드 + 몫) 모델로 고정 — 이 기록은 그 식에 대한 것
base = lambda p: os.path.basename(str(p).replace(chr(92), "/"))
z = np.load(f"{HERE}/head_inputs.npz", allow_pickle=True)
feat = {base(n): np.log10(np.expm1(f[6:9]) + 1.0) for n, f in zip(z["names"], z["F"])}
ev = u["loo_eval"]
rows = []                                   # (name, true µM, read µM, log10 band I, split)
for p, t, q, ok in zip(ev["paths"], ev["true_uM"], ev["pred_uM"], ev["in_fit_range"]):
    if ok:
        rows.append((base(p), np.asarray(t, float), np.asarray(q, float), feat[base(p)], "loo"))
roles = load_mixture_roles(tp.PURE)
for it in load_mixture_list(tp.PURE, filename_truth=True):
    if roles.get(it[0], "train") != "test":
        continue
    p = tp._here(it)[0]; c = tp.conc_of(p)
    if c is None or not all(v in tp.GRID for v in c):
        continue
    wn, cube, _a, _b = load_map(p)
    meta = dl_model.apply_uM_pixels(m, wn, np.asarray(cube, float), return_meta=True)[2]
    X = np.asarray(u["mu"], float) + np.asarray(meta["feature_z"], float) * np.asarray(u["sd"], float)
    rows.append((base(p), np.asarray(c, float), np.array([meta["map_uM"][s] for s in S3]), X[:3], "test"))
T = np.array([r[1] for r in rows]); P = np.array([r[2] for r in rows]); LB = np.array([r[3] for r in rows])
I = 10 ** LB - 1.0
OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


def ols(x, y):
    b, a = np.polyfit(x, y, 1); r = y - (a + b * x)
    return b, a, 1 - r.var() / y.var(), r.std()


Ct, It = T.sum(1), I.sum(1)
say(f"농도 판독이 눌리는 이유 — 64 격자 조건 (loo {sum(r[4] == 'loo' for r in rows)} · test {sum(r[4] == 'test' for r in rows)})")
say("\n[1] 세기는 농도를 얼마나 따라가는가 (log–log, 기울기 1 = 농도 2배에 세기 2배)")
b, a, r2, sd = ols(np.log10(Ct), np.log10(It))
say(f"  세 밴드 세기 합 ~ 총 농도        기울기 {b:.2f} · r² {r2:.2f} · 조건 간 산포 {sd:.2f} dex (×{10 ** sd:.2f})")
say(f"    → 세기 하나로 총 농도를 되짚으면 산포 {sd / b:.2f} dex (×{10 ** (sd / b):.1f}) — 격자 전체 폭이 8배(0.90 dex)")
for k, s in enumerate(S3):
    b1, _, r21, sd1 = ols(np.log10(T[:, k]), LB[:, k])
    b2, _, r22, _ = ols(np.log10(Ct), LB[:, k])
    say(f"  {s:3s} 밴드 ~ 자기 농도           기울기 {b1:+.2f} · r² {r21:.2f} · 산포 {sd1:.2f} dex | ~ 총 농도: 기울기 {b2:+.2f} · r² {r22:.2f}")
say("\n[2] 총 농도 구간별 — 세기와 판독")
say("  총 농도 µM | n | 세기 합 중앙 | 세기/µM 중앙 | 판독 총합/참 총합 중앙 | 성분 2배 이내 %")
for lo, hi in ((9, 18), (21, 30), (33, 42), (45, 54), (57, 72)):
    g = (Ct >= lo) & (Ct <= hi); f = np.maximum(P[g] / T[g], T[g] / P[g])
    say(f"  {lo:3d}–{hi:<3d}    | {g.sum():2d} | {np.median(It[g]):8.0f} | {np.median(It[g] / Ct[g]):6.1f} | "
        f"{np.median(P[g].sum(1) / Ct[g]):.2f} | {100 * np.mean(f <= 2):5.1f}")
say("\n[3] 등몰 대각선 (조성이 같아 세기만이 정보)")
for c in (3, 6, 12, 24):
    g = np.where((T == c).all(1))[0]
    for i in g:
        say(f"  {c:2d}/{c}/{c} µM ({rows[i][4]}) | 밴드 세기 DQ {I[i, 0]:6.0f} · TBZ {I[i, 1]:6.0f} · THI {I[i, 2]:6.0f} · 합 {It[i]:6.0f} | 판독 "
            + " / ".join(f"{v:.1f}" for v in P[i]))
say("\n[4] 눌림의 크기 — 판독 대 참 (log–log 기울기, 1 이면 안 눌림)")
for k, s in enumerate(S3):
    b, _, r2, _ = ols(np.log10(T[:, k]), np.log10(P[:, k]))
    say(f"  {s:3s} 기울기 {b:.2f} · r² {r2:.2f}")
b, _, r2, _ = ols(np.log10(Ct), np.log10(P.sum(1)))
say(f"  총합 기울기 {b:.2f} · r² {r2:.2f}")
sh_t, sh_p = T / T.sum(1, keepdims=True), P / P.sum(1, keepdims=True)
b, _, r2, _ = ols(np.log10(sh_t).ravel(), np.log10(sh_p).ravel())
say(f"  판독 농도끼리의 몫(조성) 기울기 {b:.2f} · r² {r2:.2f}")
say("\n[5] 참 농도 수준별 판독 중앙값(µM) · 2배 이내 %")
for c in (3, 6, 12, 24):
    g = T == c; f = np.maximum(P[g] / c, c / P[g])
    say(f"  {c:2d} µM | 판독 중앙 {np.median(P[g]):5.1f} (10–90 %: {np.percentile(P[g], 10):.1f}–{np.percentile(P[g], 90):.1f}) | {100 * np.mean(f <= 2):5.1f}")
say("\n[6] 24 µM 성분이 낮게 읽히는 조건 — 나머지 두 성분의 합에 따라")
g = np.where(T == 24)
oth = T[g[0]].sum(1) - 24
for lo, hi in ((6, 15), (18, 30), (36, 48)):
    q = (oth >= lo) & (oth <= hi)
    if q.any():
        v = P[g][q]
        say(f"  나머지 합 {lo:2d}–{hi:<2d} µM | n {q.sum():2d} | 24 µM 판독 중앙 {np.median(v):5.1f} | 2배 이내 {100 * np.mean(v >= 12):5.1f} %")

say("\n[7] 포화인가, 잡음인가")
x, y = np.log10(Ct), np.log10(It)
c2 = np.polyfit(x, y, 2); r1 = y - np.polyval(np.polyfit(x, y, 1), x); r2_ = y - np.polyval(c2, x)
say(f"  log 세기 합 ~ log 총 농도: 1차 잔차 {r1.std():.3f} dex · 2차 항 넣으면 {r2_.std():.3f} dex (2차 계수 {c2[0]:+.2f})")
say(f"    → 휘어짐(포화)은 따로 안 보인다. 처음부터 끝까지 지수 {np.polyfit(x, y, 1)[0]:.2f} 의 완만한 응답")
bt, at, _, _ = ols(np.log10(Ct), np.log10(P.sum(1)))
say("  총 농도 구간별 '판독 총합/참 총합' — 실측 대 '가운데로 당겨짐(기울기 %.2f)만 있을 때' 기대" % bt)
for lo, hi in ((9, 18), (21, 30), (33, 42), (45, 54), (57, 72)):
    g = (Ct >= lo) & (Ct <= hi)
    say(f"    {lo:3d}–{hi:<3d} µM | 실측 {np.median(P[g].sum(1) / Ct[g]):.2f} | 기대 {np.median(10 ** (at + bt * np.log10(Ct[g])) / Ct[g]):.2f}")
say("\n[8] 총 농도 구간별 성분 2배 이내 % — 판독 대 대조(상수 8.4 µM)")
for lo, hi in ((9, 42), (45, 72), (9, 72)):
    g = (Ct >= lo) & (Ct <= hi); f = np.maximum(P[g] / T[g], T[g] / P[g]); f0 = np.maximum(8.4 / T[g], T[g] / 8.4)
    say(f"  총 {lo}–{hi} µM | 조건 {g.sum():2d} | 판독 {100 * np.mean(f <= 2):5.1f} % · 1.5배 {100 * np.mean(f <= 1.5):5.1f} % · 중앙 {np.median(f):.2f}× | 대조 {100 * np.mean(f0 <= 2):5.1f} %")
say("\n[9] 눌림을 펴면(기울기 1 로 되돌리면) 어떻게 되나 — 진단용, 기울기는 64 조건 전체에서 잡음")
L = np.log10(P); Lt = np.log10(T); D = np.empty_like(L)
for k in range(3):
    b, a, _, _ = ols(Lt[:, k], L[:, k]); D[:, k] = (L[:, k] - a) / b
f = np.maximum(P / T, T / P); fd = 10 ** np.abs(D - Lt)
say(f"  그대로(눌림 있음) | 2배 이내 {100 * np.mean(f <= 2):5.1f} % · 중앙 {np.median(f):.2f}× · log 오차 RMSE {np.sqrt(np.mean((L - Lt) ** 2)):.2f} dex")
say(f"  편 것(눌림 없음)  | 2배 이내 {100 * np.mean(fd <= 2):5.1f} % · 중앙 {np.median(fd):.2f}× · log 오차 RMSE {np.sqrt(np.mean((D - Lt) ** 2)):.2f} dex")
for c in (3, 24):
    g = T == c
    say(f"    참 {c:2d} µM: 그대로 중앙 {np.median(P[g]):5.1f} (10–90 %: {np.percentile(P[g], 10):.1f}–{np.percentile(P[g], 90):.1f}) | 편 것 중앙 {np.median(10 ** D[g]):5.1f} (10–90 %: {np.percentile(10 ** D[g], 10):.1f}–{np.percentile(10 ** D[g], 90):.1f})")
say("\n[10] 24/24/24 의 조성 판독(판독 농도의 몫)")
i = np.where((T == 24).all(1))[0][0]
say("  몫 " + " / ".join(f"{100 * v:.0f} %" for v in P[i] / P[i].sum()) + " (참 33/33/33)")
i12 = np.where((T == 12).all(1))[0][0]
say(f"  12/12/12 대비 세기 합 {It[i] / It[i12]:.2f}배 (농도는 2배) · 조건 간 세기 산포 ×{10 ** r1.std():.2f}")
open(f"{HERE}/compression_cause.txt", "w", encoding="utf-8").write("\n".join(OUT) + "\n")
