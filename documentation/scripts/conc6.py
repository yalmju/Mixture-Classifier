"""Can the 6-map set return concentration from the ANALYTE signal alone (no ink)?

Model tested:  s_im = g_m * r_i * C_im      (per-map gain g, per-component response r)
  - composition  = s_i / r_i normalised  -> gain-free, testable by leave-one-map-out
  - absolute C   = needs g_m, which is exactly what the hot-spot scatter destroys
"""
import sys, math
import numpy as np

sys.path.insert(0, "/private/tmp/claude-501/-Users-seungki2-Documents-GitHub-Mixture-Classifier--claude-worktrees-ink-check-6maps-run-ad8cec/61a536fa-e12d-4dee-8d68-95377dbe80ce/scratchpad")
import ink6
from real_data import load_map
from sers_mixture import als_baseline
from scipy.optimize import nnls

names, wn_t, P = ink6.templates()
SUB = ["DQ", "TBZ", "THI"]
SI = [names.index(s) for s in SUB]
base = "/Volumes/Seungki/260805/tert-new"
comp = {1: (6, 6, 24), 2: (12, 12, 12), 3: (24, 6, 6),
        4: (6, 6, 6), 5: (24, 24, 24), 6: (12, 12, 48)}


def pixels(path):
    wn, cube, _m, _c = load_map(path)
    wn = np.asarray(wn, float)
    m = (wn >= ink6.LO) & (wn <= ink6.HI)
    wn_m, cube = wn[m], np.asarray(cube, float)[:, m]
    out = []
    for v in cube:
        y = np.clip(v - als_baseline(v), 0.0, None)
        if len(wn_m) != len(wn_t) or not np.allclose(wn_m, wn_t):
            y = np.interp(wn_t, wn_m, y)
        a, _ = nnls(P.T, y)
        out.append(a)
    return np.asarray(out)


A = {n: pixels(f"{base}/{n}-1.csv") for n in comp}
S = np.array([[np.median(A[n][:, i]) for i in SI] for n in sorted(comp)])   # (6,3)
C = np.array([comp[n] for n in sorted(comp)], float)
maps = sorted(comp)

print("=== measured substance coefficients (pixel median) vs true uM ===")
print(f"{'map':>18} " + "  ".join(f"{s:>18}" for s in SUB))
for k, n in enumerate(maps):
    print(f"  #{n} {'/'.join(f'{int(c):2d}' for c in C[k]):>12} " +
          "  ".join(f"{S[k,j]:8.0f} @{C[k,j]:4.0f}uM" for j in range(3)))

print("\n=== is each component detected? (signal per uM, relative to its own best map) ===")
for j, s in enumerate(SUB):
    per = S[:, j] / C[:, j]
    print(f"  {s:>4}: signal/uM = " + "  ".join(f"#{maps[k]}:{per[k]:7.0f}" for k in range(6)) +
          f"   spread {per.max()/per.min():5.1f}x")

# ---- two-way fit  log s = log g_m + log r_i + log C  ------------------------
Y = np.log10(S) - np.log10(C)          # = log g_m + log r_i
gm = Y.mean(axis=1)                    # per-map gain (log)
ri = Y.mean(axis=0)                    # per-component response (log)
ri = ri - ri.mean()
gm = Y.mean(axis=1) + (Y.mean(axis=0) - ri).mean() * 0
pred = gm[:, None] + ri[None, :]
res = Y - pred
print("\n=== two-way (gain x response) fit residuals — how well does s ~ g*r*C hold? ===")
print("  per-map gain g (x, rel. to mean):", "  ".join(f"#{maps[k]}:{10**(gm[k]-gm.mean()):5.2f}" for k in range(6)))
print("  response r (x, rel. to DQ)      :", "  ".join(f"{SUB[j]}:{10**(ri[j]-ri[0]):6.1f}" for j in range(3)))
print(f"  residual: typical {10**np.sqrt((res**2).mean()):.2f}x   worst {10**np.abs(res).max():.2f}x")
for k in range(6):
    print(f"    #{maps[k]}: " + "  ".join(f"{SUB[j]}:{10**res[k,j]:5.2f}x" for j in range(3)))

# ---- leave-one-map-out: predict composition (gain-free) ---------------------
print("\n=== LOO: predict COMPOSITION (mole fractions) from s_i / r_i ===")
errs = []
for k in range(6):
    tr = [q for q in range(6) if q != k]
    r = (np.log10(S[tr]) - np.log10(C[tr])).mean(axis=0)
    r = r - r.mean()
    est = S[k] / 10 ** r
    est = est / est.sum()
    true = C[k] / C[k].sum()
    e = np.abs(est - true).sum() / 2 * 100          # total-variation error, %
    errs.append(e)
    print(f"  #{maps[k]} true " + "/".join(f"{v*100:4.1f}" for v in true) +
          "   est " + "/".join(f"{v*100:4.1f}" for v in est) + f"   err {e:4.1f}%")
print(f"  mean composition error = {np.mean(errs):.1f}%   (MLP on 34 maps: 19.1%, NNLS 27.2%)")

# ---- LOO: predict ABSOLUTE concentration -----------------------------------
print("\n=== LOO: predict ABSOLUTE uM  (needs the per-map gain — use the training mean) ===")
folds = []
for k in range(6):
    tr = [q for q in range(6) if q != k]
    L = np.log10(S[tr]) - np.log10(C[tr])
    g_tr = L.mean()                                  # single global gain from training
    r = L.mean(axis=0) - L.mean()
    est = S[k] / 10 ** (g_tr + r)
    fold = np.maximum(est / C[k], C[k] / est)
    folds.append(fold)
    print(f"  #{maps[k]} true " + "/".join(f"{v:5.1f}" for v in C[k]) +
          " uM   est " + "/".join(f"{v:5.1f}" for v in est) +
          "    fold err " + "/".join(f"{v:4.1f}x" for v in fold))
folds = np.array(folds)
print(f"  median fold error {np.median(folds):.2f}x | within 2x: {100*(folds<2).mean():.0f}% | worst {folds.max():.1f}x")

# ---- total concentration ----------------------------------------------------
print("\n=== LOO: predict TOTAL uM ===")
for k in range(6):
    tr = [q for q in range(6) if q != k]
    L = np.log10(S[tr]) - np.log10(C[tr])
    g_tr, r = L.mean(), L.mean(axis=0) - L.mean()
    est = (S[k] / 10 ** (g_tr + r)).sum()
    tot = C[k].sum()
    print(f"  #{maps[k]} true {tot:5.1f} uM   est {est:6.1f} uM   {max(est/tot, tot/est):4.2f}x")
