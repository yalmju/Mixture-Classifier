"""1:1:1 계열로 농도별 보정곡선을 만들고, 편중 조성에 적용해 검증."""
import sys, math, os
import numpy as np
from scipy.optimize import nnls

REPO = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/github/Mixture Classifier"
sys.path.insert(0, REPO)
sys.path.insert(0, "/private/tmp/claude-501/-Users-seungki2-Documents-GitHub-Mixture-Classifier--claude-worktrees-ink-check-6maps-run-ad8cec/61a536fa-e12d-4dee-8d68-95377dbe80ce/scratchpad")
import ink6
from real_data import load_map

names, wn_t, P = ink6.templates()
SUB = ["DQ", "TBZ", "THI"]
SI = [names.index(s) for s in SUB]
DB = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB"
NEW = f"{DB}/tert-new-baseline"
OLD = f"{DB}/Ratio/260805"
COMP = {1: (6, 6, 24), 2: (12, 12, 12), 3: (24, 6, 6),
        4: (6, 6, 6), 5: (24, 24, 24), 6: (12, 12, 48)}


def frac(path, already_baselined=True):
    wn, cube, _m, _c = load_map(path)
    wn = np.asarray(wn, float)
    m = (wn >= ink6.LO) & (wn <= ink6.HI)
    X = np.asarray(cube, float)[:, m]
    if not already_baselined:
        from sers_mixture import als_baseline
        X = np.stack([v - als_baseline(v) for v in X])
    y = np.interp(wn_t, wn[m], np.clip(X, 0, None).mean(axis=0))
    a, _ = nnls(P.T, y)
    s = a[SI]
    return s / max(s.sum(), 1e-9)


# ---- 1:1:1 계열 수집 ----
series = []                                  # (C per component, fractions)
for n, c in ((4, 6.0), (2, 12.0), (5, 24.0)):
    for r in (1, 2, 3):
        p = f"{NEW}/{n}-{r}_corrected.csv"
        if os.path.exists(p):
            series.append((c, frac(p)))
for tag, c in (("003mM", 12.3), ("01mM", 37.0), ("03mM", 111.0), ("1mM", 333.0)):
    for s in (1, 2):
        p = f"{OLD}/tert-{s}/tert_{tag}_{s}_corrected.csv"
        if os.path.exists(p):
            series.append((c, frac(p)))

print("=== 1:1:1 계열 — 참값은 항상 33.3 / 33.3 / 33.3 ===")
print(f"{'C/성분 µM':>10} {'n':>3} | {'관측 조성 % (중앙)':>26} | {'왜곡 배수 (관측/33.3)':>26}")
lv = sorted(set(c for c, _ in series))
curve = {}
for c in lv:
    F = np.array([f for cc, f in series if cc == c])
    med = np.median(F, axis=0)
    curve[c] = med
    print(f"  {c:8.1f} {len(F):3d} | " + " / ".join(f"{v*100:7.1f}" for v in med) +
          " | " + " / ".join(f"{v/(1/3):7.2f}" for v in med))

xs = np.log10(lv)
R = np.array([curve[c] for c in lv])          # (n_lv, 3) 관측 분율


def correct(fobs, C_tot_per_comp):
    """농도에 맞는 왜곡을 로그보간으로 찾아 나눠줌"""
    x = math.log10(max(C_tot_per_comp, lv[0]))
    r = np.array([np.interp(x, xs, np.log10(np.maximum(R[:, j], 1e-6))) for j in range(3)])
    r = 10 ** r
    out = fobs / r
    return out / out.sum()


print("\n=== 편중 조성에 적용 (보정곡선은 1:1:1에서만 만들어짐) ===")
print(f"{'조건':>14} {'rep':>3} | {'참 %':>18} | {'보정 전 %':>20} {'오차':>6} | {'보정 후 %':>20} {'오차':>6}")
b4, af = [], []
for n in (1, 3, 6):
    C = np.array(COMP[n], float)
    t = C / C.sum() * 100
    for r in (1, 2, 3):
        p = f"{NEW}/{n}-{r}_corrected.csv"
        if not os.path.exists(p):
            continue
        f = frac(p)
        e0 = np.abs(f * 100 - t).sum() / 2
        fc = correct(f, C.sum() / 3)          # 참 총농도를 안다고 가정(상한 성능)
        e1 = np.abs(fc * 100 - t).sum() / 2
        b4.append(e0); af.append(e1)
        lab = "/".join(str(int(v)) for v in C) if r == 1 else ""
        print(f"  #{n} {lab:>9} {r:>3} | " + "/".join(f"{v:5.1f}" for v in t) +
              " | " + "/".join(f"{v:5.1f}" for v in f * 100) + f" {e0:5.1f}%" +
              " | " + "/".join(f"{v:5.1f}" for v in fc * 100) + f" {e1:5.1f}%")
print(f"\n  편중 조성 평균 오차   보정 전 {np.mean(b4):5.1f}%  ->  보정 후 {np.mean(af):5.1f}%")

print("\n=== 참고: 1:1:1 자기 자신에 적용하면 (자명하게 좋아야 함) ===")
e0, e1 = [], []
for n, c in ((4, 6.0), (2, 12.0), (5, 24.0)):
    for r in (1, 2, 3):
        p = f"{NEW}/{n}-{r}_corrected.csv"
        if not os.path.exists(p):
            continue
        f = frac(p)
        e0.append(np.abs(f * 100 - 100 / 3).sum() / 2)
        e1.append(np.abs(correct(f, c) * 100 - 100 / 3).sum() / 2)
print(f"  보정 전 {np.mean(e0):5.1f}%  ->  보정 후 {np.mean(e1):5.1f}%")
