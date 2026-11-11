"""농도 단계를 신경망 → 혼합물 검량식으로 바꾸면 결과가 어떻게 달라지는가 (2026-11-10).

같은 맵 · 같은 입력(head_inputs.npz, 모델 기록과 같은 조건) · 조건 하나씩 빼기.
  현재      : 신경망(128-32), 82맵 학습 — head_cv_preds.npz 의 loo_net (모델 파일 기록 80.8 % 와 같은 값)
  검량식 A  : log10 C_j = a·log10(밴드_j) + b·몫_j + c                    (성분마다 계수 3개)
  검량식 B  : log10 C_j = Σ a_k·log10(밴드_k) + b1·몫_DQ + b2·몫_TBZ + c   (성분마다 계수 6개)
검량식은 3–24 µM 격자 맵만으로 맞춘다. 격자 밖 맵은 격자 전체로 맞춘 식을 그대로 적용(외삽).
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(f"{HERE}/head_inputs.npz", allow_pickle=True)
F, T, COND = z["F"], z["T"], z["cond"].astype(object)
NET = np.load(f"{HERE}/head_cv_preds.npz", allow_pickle=True)["loo_net"]
S = ["DQ", "TBZ", "THI"]
GRID = np.isin(T, [3, 6, 12, 24]).all(1)
RAT = F[:, 3:6]; LB = np.log10(np.expm1(F[:, 6:9]) + 1.0); LT = np.log10(np.clip(T, 0.05, None))
OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


def fit(X, y):
    return np.linalg.lstsq(np.c_[X, np.ones(len(X))], y, rcond=None)[0]


def apply(w, X):
    return 10 ** (np.c_[X, np.ones(len(X))] @ w)


XA = lambda j: np.c_[LB[:, j], RAT[:, j]]
XB = lambda j: np.c_[LB, RAT[:, :2]]
XS = lambda j: RAT[:, [j]]


def loo_grid(X_of):
    """격자 맵: 그 조건을 뺀 나머지 격자 맵으로 맞춤. 격자 밖 맵: 격자 전체로 맞춘 식."""
    P = np.zeros_like(T)
    g = np.where(GRID)[0]
    for j in range(3):
        X = X_of(j)
        for c in dict.fromkeys(COND[g].tolist()):
            te = g[COND[g] == c]; tr = g[COND[g] != c]
            P[te, j] = apply(fit(X[tr], LT[tr, j]), X[te])
        P[~GRID, j] = apply(fit(X[g], LT[g, j]), X[~GRID])
    return P


R = {"현재 신경망": NET, "검량식 A (자기 밴드 + 몫)": loo_grid(XA), "검량식 B (세 밴드 + 몫)": loo_grid(XB),
     "대조: 몫만": loo_grid(XS), "대조: 상수 8.5 µM": np.full_like(T, 10 ** LT[GRID].mean())}


def fold(p, t):
    return np.maximum(p / t, t / p)


say(f"[1] 3–24 µM 격자 {GRID.sum()}맵 · 성분 {GRID.sum() * 3}개 — 조건 하나씩 빼기 (조성 입력은 전체로 학습한 PLS)")
for n, P in R.items():
    q = fold(P[GRID], T[GRID]); pc = 100 * (q <= 2).mean(0)
    say(f"  {n:22s} 1.5배 {100 * (q <= 1.5).mean():5.1f} % · 2배 {100 * (q <= 2).mean():5.1f} % (DQ {pc[0]:5.1f} · TBZ {pc[1]:5.1f} · THI {pc[2]:5.1f})"
        f" · 3배 {100 * (q <= 3).mean():5.1f} % · 중앙 배수 {np.median(q):.2f}× · MAE {np.abs(P[GRID] - T[GRID]).mean():.2f} µM")

say()
say("[2] 참 농도별 — 2배 이내 % / 판독 중앙값 µM (격자 맵, 세 성분 합침)")
for n, P in R.items():
    cells = []
    for v in (3, 6, 12, 24):
        m = GRID[:, None] & (T == v)
        cells.append(f"{v:2d} µM: {100 * (fold(P[m], T[m]) <= 2).mean():5.1f} % / {np.median(P[m]):5.1f}")
    say(f"  {n:22s} " + " · ".join(cells))

say()
say("[3] 성분별 판독 범위 — 참 3 µM 과 24 µM 의 판독 중앙값 (좁을수록 농도를 구분 못 함)")
for n, P in R.items():
    say(f"  {n:22s} " + " · ".join(
        f"{s} {np.median(P[GRID & (T[:, j] == 3), j]):4.1f} → {np.median(P[GRID & (T[:, j] == 24), j]):4.1f}" for j, s in enumerate(S)))

say()
say("[4] 같은 조건을 두 방법이 얼마나 다르게 읽는가 (격자 맵)")
for n in ("검량식 A (자기 밴드 + 몫)", "검량식 B (세 밴드 + 몫)"):
    d = fold(R[n][GRID], NET[GRID])
    b_ok = fold(R[n][GRID], T[GRID]) <= 2; n_ok = fold(NET[GRID], T[GRID]) <= 2
    say(f"  {n}: 신경망 판독과의 배수 차 중앙 {np.median(d):.2f}× · 둘 다 2배 이내 {int((b_ok & n_ok).sum())}"
        f" · 검량식만 {int((b_ok & ~n_ok).sum())} · 신경망만 {int((~b_ok & n_ok).sum())} · 둘 다 밖 {int((~b_ok & ~n_ok).sum())} (성분 {b_ok.size}개)")

say()
off = ~GRID
say(f"[5] 격자 밖 {off.sum()}맵 (구간 위·이원) — 있는 성분의 2배 이내 %, 참 농도별")
lv = sorted(set(T[off][T[off] > 0].tolist()))
for n in ("현재 신경망", "검량식 A (자기 밴드 + 몫)", "검량식 B (세 밴드 + 몫)"):
    P = R[n]; cells = []
    for v in lv:
        m = off[:, None] & (T == v)
        cells.append(f"{v:g}: {100 * (fold(P[m], T[m]) <= 2).mean():.0f} % (n={int(m.sum())})")
    say(f"  {n:22s} " + " · ".join(cells))
say("  없는 성분(참 0)을 읽는 값, 중앙 / 최대 µM")
m0 = off[:, None] & (T == 0)
for n in ("현재 신경망", "검량식 A (자기 밴드 + 몫)", "검량식 B (세 밴드 + 몫)"):
    say(f"  {n:22s} {np.median(R[n][m0]):.1f} / {R[n][m0].max():.1f} µM (n={int(m0.sum())})")

say()
say("[6] 검량식 계수 (격자 52맵 전체로 맞춤) — log10 C = …")
g = np.where(GRID)[0]
for j, s in enumerate(S):
    a, b, c = fit(XA(j)[g], LT[g, j])
    say(f"  A {s:3s}: {a:+.2f}·log10(밴드) {b:+.2f}·몫 {c:+.2f}")
for j, s in enumerate(S):
    w = fit(XB(j)[g], LT[g, j])
    say(f"  B {s:3s}: {w[0]:+.2f}·log I_DQ {w[1]:+.2f}·log I_TBZ {w[2]:+.2f}·log I_THI {w[3]:+.2f}·몫_DQ {w[4]:+.2f}·몫_TBZ {w[5]:+.2f}")
open(f"{HERE}/replace_preview.txt", "w", encoding="utf-8").write("\n".join(OUT) + "\n")
np.savez(f"{HERE}/replace_preview_preds.npz", T=T, GRID=GRID, **{f"p{i}": P for i, P in enumerate(R.values())}, names=np.array(list(R)))
