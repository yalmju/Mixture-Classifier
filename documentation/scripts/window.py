"""정량 창이 왜 좁은가 — 위 끝과 아래 끝을 따로 뜯어본다.

아래 끝(LOD): 광자 노이즈인가, 템플릿 누출인가?  -> VIP 마커밴드로 재면 좁아지나?
위  끝(매몰): 자기 포화인가, 경쟁자에 밀려서인가?
"""
import os as _os, sys as _sys                 # paths 부트스트랩 — 기계마다 마운트가 다르다
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import sys, math
import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, "/private/tmp/claude-501/-Users-seungki2-Documents-GitHub-Mixture-Classifier--claude-worktrees-ink-check-6maps-run-ad8cec/61a536fa-e12d-4dee-8d68-95377dbe80ce/scratchpad")
import ink6, unmix
from real_data import load_map
from sers_mixture import als_baseline

names, wn_t, P = ink6.templates()
SUB = ["DQ", "TBZ", "THI"]
SI = [names.index(s) for s in SUB]
DB = paths.PURE
COMP = {1: (6, 6, 24), 2: (12, 12, 12), 3: (24, 6, 6),
        4: (6, 6, 6), 5: (24, 24, 24), 6: (12, 12, 48)}
maps = sorted(COMP)

# ---- VIP marker bands (cross-talk 최소 밴드) --------------------------------
ref = P[SI]                       # 이미 baseline+L2 처리된 순물질 템플릿
bands = unmix.vip_bands(wn_t, ref, SUB, k=2)
print("VIP 마커밴드 (교차간섭 최소):")
for s in SUB:
    print(f"  {s:>4}: " + ", ".join(f"{b:.0f} cm-1" for b in bands[s]))


def peak_h(y, wn, centre, half=8.0):
    """밴드 최대높이 - 양옆 로컬 바닥 (좁은 창)"""
    m = np.abs(wn - centre) <= half
    if m.sum() == 0:
        return 0.0
    side = (np.abs(wn - centre) > half) & (np.abs(wn - centre) <= 3 * half)
    base = np.median(y[side]) if side.sum() else 0.0
    return max(float(y[m].max() - base), 0.0)


def read_map(path):
    wn, cube, _m, _c = load_map(path)
    wn = np.asarray(wn, float)
    m = (wn >= ink6.LO) & (wn <= ink6.HI)
    wn_m, cube = wn[m], np.asarray(cube, float)[:, m]
    Xb = np.stack([np.clip(v - als_baseline(v), 0.0, None) for v in cube])
    if len(wn_m) != len(wn_t) or not np.allclose(wn_m, wn_t):
        Xb = np.stack([np.interp(wn_t, wn_m, y) for y in Xb])
    A = np.array([nnls(P.T, y)[0] for y in Xb])
    H = np.array([[max(peak_h(y, wn_t, b) for b in bands[s]) for s in SUB] for y in Xb])
    return np.median(A[:, SI], axis=0), np.median(H, axis=0)


blanks = [f"{DB}/INK-{i}.csv" for i in (1, 2, 3)] + [f"{DB}/BLK-{i}.csv" for i in (1, 2, 3, 4)]
Bn, Bh = [], []
for p in blanks:
    a, h = read_map(p)
    Bn.append(a); Bh.append(h)
Bn, Bh = np.array(Bn), np.array(Bh)

Sn, Sh = [], []
for n in maps:
    a, h = read_map(f"/Volumes/Seungki/260805/tert-new/{n}-1.csv")
    Sn.append(a); Sh.append(h)
Sn, Sh = np.array(Sn), np.array(Sh)
C = np.array([COMP[n] for n in maps], float)

print("\n=== 아래 끝: LOD를 무엇이 정하는가 ===")
print(f"{'':>6} {'':>10} {'blank sigma':>12} {'기울기(최악)':>13} {'LOD(최악)':>11}")
for meth, B, S in (("NNLS 전스펙트럼", Bn, Sn), ("VIP 마커밴드", Bh, Sh)):
    print(f"  --- {meth} ---")
    for j, s in enumerate(SUB):
        sd = B[:, j].std(ddof=1)
        sl = (S[:, j] / C[:, j]).min()
        print(f"  {s:>4} {'':>10} {sd:12.1f} {sl:13.0f} {3.3*sd/sl:10.2f}uM")

print("\n  blank에서의 허위 신호 (누출) — 0이어야 정상:")
for meth, B in (("NNLS", Bn), ("VIP ", Bh)):
    for j, s in enumerate(SUB):
        print(f"    {meth} {s:>4}: 평균 {B[:,j].mean():8.1f}  최대 {B[:,j].max():8.1f}")

print("\n=== 위 끝: 매몰이 자기 포화인가 경쟁 때문인가 ===")
print("  1:1:1 희석 계열 (모든 성분이 함께 오른다):")
for tag, c in (("003mM", 12.3), ("01mM", 37.0), ("03mM", 111.0), ("1mM", 333.0)):
    row = []
    for s in (1, 2):
        a, _ = ink6.ink_of(f"/Volumes/Seungki/260805/tert-{s}/tert_{tag}_{s}.csv",
                           names, wn_t, P)
        row.append(a)
    row = np.array(row)
    print(f"    {c:6.1f} uM/성분 : " +
          "  ".join(f"{SUB[j]}={row[:,SI[j]].mean():8.0f}" for j in range(3)))

print("\n  같은 DQ 농도에서 경쟁자만 바꿨을 때 (새 6맵):")
for j, s in enumerate(SUB):
    print(f"    {s}:")
    for k, n in enumerate(maps):
        others = C[k].sum() - C[k, j]
        print(f"      #{n} C_{s}={C[k,j]:4.0f}  경쟁자합={others:4.0f}  신호={Sn[k,j]:8.0f}"
              f"   uM당 {Sn[k,j]/C[k,j]:7.0f}")
