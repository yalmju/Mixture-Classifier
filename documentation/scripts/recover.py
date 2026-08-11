"""Per-component recovery in the regime where THI does NOT dominate.

Q1  Does the Langmuir ratio law hold?  s_i/s_j should equal (R_i C_i)/(R_j C_j)
    with the competition term cancelled.  If R_i/R_j drifts, competition is
    stronger than Langmuir and needs its own term.
Q2  How well is COMPOSITION recovered (gain-free), per component?
Q3  Does adding an exclusion term  log(s_i/s_THI) ~ log(C_i/C_THI) + log(sum C)
    fix it?
"""
import sys, math, itertools
import numpy as np

sys.path.insert(0, "/private/tmp/claude-501/-Users-seungki2-Documents-GitHub-Mixture-Classifier--claude-worktrees-ink-check-6maps-run-ad8cec/61a536fa-e12d-4dee-8d68-95377dbe80ce/scratchpad")
import ink6
from real_data import load_map
from sers_mixture import als_baseline
from scipy.optimize import nnls

names, wn_t, P = ink6.templates()
SUB = ["DQ", "TBZ", "THI"]
SI = [names.index(s) for s in SUB]
IK = names.index("INK")
comp = {1: (6, 6, 24), 2: (12, 12, 12), 3: (24, 6, 6),
        4: (6, 6, 6), 5: (24, 24, 24), 6: (12, 12, 48)}
maps = sorted(comp)


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


A = {n: pixels(f"/Volumes/Seungki/260805/tert-new/{n}-1.csv") for n in maps}
S = np.array([[np.median(A[n][:, i]) for i in SI] for n in maps])
INK = np.array([np.median(A[n][:, IK]) for n in maps])
C = np.array([comp[n] for n in maps], float)
TOT = C.sum(axis=1)

print("=== Q1  Langmuir ratio law:  (s_i/s_THI) / (C_i/C_THI)  must be CONSTANT ===")
print(f"{'map':>16} {'tot':>5} {'R_DQ/R_THI':>11} {'R_TBZ/R_THI':>12}")
rr = []
for k, n in enumerate(maps):
    r_dq = (S[k, 0] / S[k, 2]) / (C[k, 0] / C[k, 2])
    r_tb = (S[k, 1] / S[k, 2]) / (C[k, 1] / C[k, 2])
    rr.append((r_dq, r_tb))
    print(f"  #{n} {'/'.join(str(int(v)) for v in C[k]):>11} {TOT[k]:5.0f} {r_dq:11.3f} {r_tb:12.3f}")
rr = np.array(rr)
print(f"  spread: DQ {rr[:,0].max()/rr[:,0].min():5.1f}x    TBZ {rr[:,1].max()/rr[:,1].min():5.1f}x"
      "     (Langmuir predicts 1.0x for both)")

print("\n  --- does the drift track total concentration? ---")
for j, s in enumerate(("DQ", "TBZ")):
    sl, ic = np.polyfit(np.log10(TOT), np.log10(rr[:, j]), 1)
    pr = sl * np.log10(TOT) + ic
    r2 = 1 - ((np.log10(rr[:, j]) - pr) ** 2).sum() / ((np.log10(rr[:, j]) - np.log10(rr[:, j]).mean()) ** 2).sum()
    print(f"    R_{s}/R_THI  ~ total^{sl:+.2f}   R2={r2:.3f}")

# ---------------- Q2 composition, per component ----------------------------
def loo_composition(use_exclusion):
    out = []
    for k in range(6):
        tr = [q for q in range(6) if q != k]
        if not use_exclusion:
            r = (np.log10(S[tr]) - np.log10(C[tr])).mean(axis=0)
            est = S[k] / 10 ** r
        else:
            # fit log(s_i/s_THI) = a_i + log(C_i/C_THI) + q_i*log(totC) on the 5,
            # then solve the held-out map's composition with its total unknown ->
            # iterate: start from the no-exclusion estimate, refine the total.
            coef = {}
            for j in (0, 1):
                y = np.log10(S[tr, j] / S[tr, 2]) - np.log10(C[tr, j] / C[tr, 2])
                Xd = np.column_stack([np.ones(len(tr)), np.log10(C[tr].sum(axis=1))])
                coef[j], *_ = np.linalg.lstsq(Xd, y, rcond=None)
            est = S[k] / np.array([1.0, 1.0, 1.0])
            est = est / est.sum() * TOT[k]          # seed with the TRUE total (upper bound)
            for _ in range(50):
                t = est.sum()
                ratio = np.ones(3)
                for j in (0, 1):
                    logR = coef[j][0] + coef[j][1] * math.log10(t)
                    ratio[j] = (S[k, j] / S[k, 2]) / 10 ** logR
                est_new = np.array([ratio[0], ratio[1], 1.0])
                est_new = est_new / est_new.sum() * t
                est = 0.5 * est + 0.5 * est_new
        est = est / est.sum()
        out.append((est, C[k] / C[k].sum()))
    return out


for lbl, flag in (("Langmuir ratio (no exclusion term)", False),
                  ("+ exclusion term  R_i/R_THI ~ total^q", True)):
    res = loo_composition(flag)
    tv = [np.abs(e - t).sum() / 2 * 100 for e, t in res]
    per = np.array([np.abs(e - t) * 100 for e, t in res])
    print(f"\n=== Q2/Q3  LOO composition — {lbl} ===")
    for k, (e, t) in enumerate(res):
        print(f"  #{maps[k]} true " + "/".join(f"{v*100:4.1f}" for v in t) +
              "   est " + "/".join(f"{v*100:4.1f}" for v in e) + f"   err {tv[k]:4.1f}%")
    print(f"  mean total-variation error {np.mean(tv):5.1f}%   "
          f"per-component mean |Δ|: " + "  ".join(f"{SUB[j]}:{per[:,j].mean():4.1f}%p" for j in range(3)))

# ---------------- how many gain-free numbers do we have? -------------------
print("\n=== degrees of freedom ===")
print("  measured, gain-free  : coefficient DIRECTION = 3 numbers when INK is included")
print("                         (DQ:TBZ:THI:INK ratios), only 2 without INK")
print("  wanted               : 3 concentrations")
print("  -> composition needs no scale reference; absolute uM needs exactly one.")
print("\n  ink fraction per map (the scale reference):")
for k, n in enumerate(maps):
    tot_sig = S[k].sum() + INK[k]
    print(f"    #{n} tot={TOT[k]:3.0f} uM   ink frac {INK[k]/tot_sig:.3f}")
