"""E6. 살아남은 한 가지(지문 영역 500–1800 cm⁻¹ 만 쓰기)를 ① 모델 기록과 같은 방식(조건 하나씩 빼기 · 52 → 시험 12)으로,
② 성분별 · 농도 수준별로, ③ 다른 조제(08-08 격자, 9–72 µM — 논문 수치 아님)에서도 같은 방향인지 확인.
  실행:  .venv/bin/python3 documentation/results/conc_improve_2026-11-11/e6_range_check.py
"""
import csv
from common import *
say = Log(f"{HERE}/e6_range_check.txt")
X0 = base_features(); V = {"채택 식 (500–2500)": X0, "500–2000": X0[:, ax <= 2000], "500–1800": X0[:, ax <= 1800]}
split = {(float(r["DQ_true_uM"]), float(r["TBZ_true_uM"]), float(r["THI_true_uM"])): r["split"]
         for r in csv.DictReader(open(f"{HERE}/../fig4_paper_pls_2026-11-10/heldout_maps.csv")) if r["grid64"] == "1"}
is_test = np.array([split[tuple(float(v) for v in t)] == "test" for t in T]); tr, te = np.where(~is_test)[0], np.where(is_test)[0]
say(f"격자 맵 {N} · 학습 {len(tr)} · 시험 {len(te)}")


def s(e):
    a = np.abs(e).ravel(); return f"2배 {100 * (a <= L2).mean():5.1f} % · 1.5배 {100 * (a <= L15).mean():5.1f} % · 중앙 {10 ** np.median(a):.2f}× · RMSE {np.sqrt((e ** 2).mean()):.3f}"


f8 = pls_fit_predict(8); PLOO = {}
say("\n① 모델 기록과 같은 방식")
for n, X in V.items():
    P = np.zeros_like(LT)
    for i in tr:                                   # 학습 52 맵: 조건 하나씩 빼기
        a = tr[tr != i]; P[i] = f8(X[a], LT[a], X[[i]])[0]
    P[te] = f8(X[tr], LT[tr], X[te]); PLOO[n] = P; E = P - LT          # 시험 12 맵: 52 맵으로 맞춘 식
    say(f"  {n:<20s} 64 조건: {s(E)} · 기울기 {np.polyfit(LT.ravel(), P.ravel(), 1)[0]:.2f} · 셋 다 2배 {int((np.abs(E) <= L2).all(1).sum())}/64")
    say(f"  {'':<20s} 그중 시험 12: {s(E[te])}   · 학습 52(하나씩 빼기): {s(E[tr])}")
c = np.repeat(LT[tr].mean(0)[None], N, 0) - LT; say(f"  {'대조 상수':<20s} 64 조건: {s(c)}")
say("\n② 성분별 · 참 농도 수준별 판독 중앙값 µM (64 조건, 위 방식) — 채택 식 → 500–1800")
for j, sub in enumerate(SUB):
    say(f"  {sub}: " + " · ".join(f"{lv} → {10 ** np.median(PLOO['채택 식 (500–2500)'][T[:, j] == lv, j]):.1f} → {10 ** np.median(PLOO['500–1800'][T[:, j] == lv, j]):.1f}" for lv in (3, 6, 12, 24))
        + f"   | 2배 이내 {100 * (np.abs(PLOO['채택 식 (500–2500)'][:, j] - LT[:, j]) <= L2).mean():.1f} → {100 * (np.abs(PLOO['500–1800'][:, j] - LT[:, j]) <= L2).mean():.1f} %")
say("  수준별 2배 이내 %(세 성분 합쳐): " + " · ".join(
    f"{lv} µM {100 * (np.abs(PLOO['채택 식 (500–2500)'] - LT)[T == lv] <= L2).mean():.0f} → {100 * (np.abs(PLOO['500–1800'] - LT)[T == lv] <= L2).mean():.0f}" for lv in (3, 6, 12, 24)))
say("\n③ 다른 조제에서도 같은 방향인가 — 08-08 격자(9–72 µM, 65 맵 + 재조제 12 맵), 조건 하나씩 빼기. 논문 수치 아님")
z5 = np.load(f"{HERE}/e5_cache.npz", allow_pickle=True); T5, X5, rep5 = z5["T"], z5["X"], z5["rep"]; L5 = np.log10(T5)
c5 = np.array(["-".join("%g" % v for v in t) for t in T5])
assert X5.shape[1] == len(ax)
for n, mk in (("500–2500", ax <= 2500), ("500–2000", ax <= 2000), ("500–1800", ax <= 1800)):
    P = np.zeros_like(L5)
    for cc in np.unique(c5):
        b = np.where(c5 == cc)[0]; a = np.where(c5 != cc)[0]; P[b] = f8(X5[a][:, mk], L5[a], X5[b][:, mk])
    E = P - L5; m0 = rep5 == 0
    ia = [np.where(m0 & (c5 == c5[i]))[0][0] for i in np.where(~m0)[0]]; ib = list(np.where(~m0)[0])
    say(f"  {n:<10s} 격자 65 맵: {s(E[m0])} · 재조제 쌍 한 맵: RMSE {np.sqrt((np.r_[E[ia], E[ib]] ** 2).mean()):.3f} · 두 맵 평균: RMSE {np.sqrt((((E[ia] + E[ib]) / 2) ** 2).mean()):.3f}"
        f" · 2배 {100 * (np.abs((E[ia] + E[ib]) / 2) <= L2).mean():.1f} %")
