"""농도 선형회귀의 입력을 하나씩 빼 본다 — 모양(PLS 몫)과 세기(밴드) 가운데 무엇이 일하는가 (2026-11-10).

head_inputs.npz(논문 모델 학습과 같은 호출에서 꺼낸 맵 단위 입력). 3–24 µM 격자 52맵, 조건 하나씩 빼기.
모든 식은 log10 C_j = w·x + c 꼴의 최소제곱, 성분마다 따로.

  실행:  .venv/bin/python3 documentation/results/conc_head_2026-11-10/input_ablation.py
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(f"{HERE}/head_inputs.npz", allow_pickle=True)
F, T, COND = z["F"], z["T"], z["cond"].astype(object)
G = np.where(np.isin(T, [3, 6, 12, 24]).all(1))[0]
SH = F[:, 3:6]                                   # PLS 몫 (맵 평균)
LB = np.log10(np.expm1(F[:, 6:9]) + 1.0)         # 특성 밴드 세기 3개
LTOT = F[:, [10]] / np.log(10)                   # 전체 스펙트럼 세기(픽셀 합의 맵 중앙값)
LT = np.log10(np.clip(T, 1e-9, None))
OUT = []


def loo(X):
    P = np.zeros((len(G), 3))
    for a, i in enumerate(G):
        tr = G[COND[G] != COND[i]]
        W = np.linalg.lstsq(np.c_[X[tr], np.ones(len(tr))], LT[tr], rcond=None)[0]
        P[a] = np.r_[X[i], 1] @ W
    return 10 ** np.abs(P - LT[G])


def row(name, f):
    s = f"  {name:34s} | {100 * np.mean(f <= 2):5.1f} | {100 * np.mean(f <= 1.5):5.1f} | {np.median(f):.2f}x"
    print(s); OUT.append(s)


OUT.append(f"농도 선형회귀 입력 비교 — 격자 {len(G)}맵 · 성분 {3 * len(G)}개, 조건 하나씩 빼기")
OUT.append("  입력                               | 2배 이내 % | 1.5배 이내 % | 중앙 배수")
print("\n".join(OUT))
const = 10 ** LT[G].mean()
row(f"대조: 상수 {const:.1f} µM", np.maximum(const / T[G], T[G] / const))
row("밴드 세기 3개만 (모양 없음)", loo(LB))
row("PLS 몫만 (세기 없음)", loo(SH[:, :2]))
row("전체 세기 1개 + PLS 몫", loo(np.c_[LTOT, SH[:, :2]]))
row("밴드 세기 3개 + PLS 몫 (채택, B)", loo(np.c_[LB, SH[:, :2]]))
open(f"{HERE}/input_ablation.txt", "w", encoding="utf-8").write("\n".join(OUT) + "\n")
