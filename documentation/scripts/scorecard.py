"""6조건 x 3반복 = 18맵.  조건 하나를 통째로 빼고(LOO) 검량 -> 그 조건 3반복을 예측.
   조성(비)과 농도(uM) 둘 다."""
import sys, math
import numpy as np

sys.path.insert(0, "/private/tmp/claude-501/-Users-seungki2-Documents-GitHub-Mixture-Classifier--claude-worktrees-ink-check-6maps-run-ad8cec/61a536fa-e12d-4dee-8d68-95377dbe80ce/scratchpad")
import silent as S

R, COMP, SUB = S.R, S.COMP, S.SUB
maps = sorted(COMP)
reps = {n: [r for r in (1, 2, 3) if (n, r) in R] for n in maps}


def fit(train):
    Sg = np.array([np.median([R[(n, r)]["S"] for r in reps[n]], axis=0) for n in train])
    Cg = np.array([COMP[n] for n in train], float)
    L = np.log10(Sg) - np.log10(Cg)
    rf = L.mean(axis=0) - L.mean()          # 상대 응답계수
    g = L.mean()                            # 평균 게인
    b = np.array([np.mean([math.log10(R[(n, r)]["band_ratio"]) for r in reps[n]])
                  for n in train])
    x = np.log10(Cg.sum(axis=1) / 3)
    sl, ic = np.polyfit(x, b, 1)            # 잉크 검량
    return rf, g, sl, ic


rows = []
for n in maps:
    tr = [q for q in maps if q != n]
    rf, g, sl, ic = fit(tr)
    Ct = np.array(COMP[n], float)
    for r in reps[n]:
        Sv = R[(n, r)]["S"]
        comp = Sv / 10 ** rf
        comp = comp / comp.sum()                                   # 조성
        tot = 10 ** ((math.log10(R[(n, r)]["band_ratio"]) - ic) / sl) * 3
        est_c = comp * tot                                         # 결합 경로
        est_a = Sv / 10 ** (g + rf)                                # 분석물 단독
        rows.append(dict(n=n, r=r, true=Ct, comp=comp, tot=tot,
                         est_c=est_c, est_a=est_a))

W = 96
print("=" * W)
print("표 1.  조성 (몰 %)  —  분석물 NNLS 비, 게인 무관, 잉크 불필요")
print("=" * W)
print(f"{'조건':>14} {'rep':>4} | {'참값 DQ/TBZ/THI':>22} | {'예측 DQ/TBZ/THI':>22} | {'오차':>6}")
print("-" * W)
errs = {}
for d in rows:
    t = d["true"] / d["true"].sum() * 100
    p = d["comp"] * 100
    e = np.abs(p - t).sum() / 2
    errs.setdefault(d["n"], []).append(e)
    lab = "/".join(str(int(v)) for v in d["true"]) if d["r"] == reps[d["n"]][0] else ""
    print(f"  #{d['n']} {lab:>9} {d['r']:>4} | " +
          " / ".join(f"{v:6.1f}" for v in t) + " | " +
          " / ".join(f"{v:6.1f}" for v in p) + f" | {e:5.1f}%")
allerr = [e for v in errs.values() for e in v]
print("-" * W)
print(f"{'':>20}조건별 평균: " + "  ".join(f"#{n} {np.mean(v):.0f}%" for n, v in errs.items()) +
      f"    ||  전체 평균 {np.mean(allerr):.1f}%  중앙 {np.median(allerr):.1f}%")

print("\n" + "=" * W)
print("표 2.  농도 (µM)  —  ① 결합: 총농도(잉크 2137밴드) x 조성   ② 분석물 단독")
print("=" * W)
print(f"{'조건':>14} {'rep':>4} | {'참값':>17} | {'① 결합 예측':>19} {'배수':>16} | {'② 단독 배수':>16}")
print("-" * W)
fc, fa = [], []
for d in rows:
    t = d["true"]
    a = np.maximum(d["est_c"] / t, t / d["est_c"])
    b = np.maximum(d["est_a"] / t, t / d["est_a"])
    fc.append(a); fa.append(b)
    lab = "/".join(str(int(v)) for v in t) if d["r"] == reps[d["n"]][0] else ""
    print(f"  #{d['n']} {lab:>9} {d['r']:>4} | " +
          "/".join(f"{v:5.1f}" for v in t) + " | " +
          "/".join(f"{v:5.1f}" for v in d["est_c"]) + "  " +
          "/".join(f"{v:4.2f}" for v in a) + " | " +
          "/".join(f"{v:4.2f}" for v in b))
fc, fa = np.array(fc), np.array(fa)
print("-" * W)
print(f"{'성분별 중앙 배수':>22} | " + " " * 19 +
      "  " + "/".join(f"{np.median(fc[:, j]):4.2f}" for j in range(3)) + " | " +
      "/".join(f"{np.median(fa[:, j]):4.2f}" for j in range(3)))
print(f"{'전체':>22} | 결합  중앙 {np.median(fc):.2f}배 · 2배이내 {100*(fc<2).mean():3.0f}% · 최악 {fc.max():.2f}배"
      f"   ||  단독  중앙 {np.median(fa):.2f}배 · 2배이내 {100*(fa<2).mean():3.0f}% · 최악 {fa.max():.2f}배")

print("\n" + "=" * W)
print("표 3.  총농도만 (잉크 2137밴드 단독)")
print("=" * W)
print(f"{'조건':>14} {'참값':>7} | {'3반복 예측 (µM)':>28} | {'중앙 배수':>9}")
print("-" * W)
ft = []
for n in maps:
    t = sum(COMP[n])
    v = [d["tot"] for d in rows if d["n"] == n]
    f = max(np.median(v) / t, t / np.median(v))
    ft.append(f)
    print(f"  #{n} {'/'.join(str(x) for x in COMP[n]):>9} {t:6d} | " +
          "  ".join(f"{x:7.1f}" for x in v) + f" | {f:8.2f}배")
print("-" * W)
print(f"{'':>24}중앙 {np.median(ft):.2f}배 · 2배이내 {100*(np.array(ft)<2).mean():.0f}% · 최악 {max(ft):.2f}배")
