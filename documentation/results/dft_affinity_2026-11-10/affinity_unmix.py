"""Does putting binding affinities into the unmixing improve the composition read?

NNLS gives the share of each component in the surface signal, s_j.  If the three
molecules bind with different strength, the solution share is x_j ∝ s_j / w_j
(w_j = relative affinity).  Tested on the 64 held-out grid conditions (3–24 µM):
  - w from DFT adsorption energies as Boltzmann weights (298 K)
  - w = exp(-beta * E_DFT) with one fitted scale beta (leave-one-condition-out)
  - w free, two parameters (leave-one-condition-out)
  - w free + one exponent (leave-one-condition-out)
against PLS trained on mixtures (no affinity input).
Error = mean over conditions of the largest absolute share error (paper definition).
"""
import csv, os
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "fig4_paper_pls_2026-11-10", "heldout_maps.csv")
S3 = ("DQ", "TBZ", "THI")
KT = 0.02568  # eV, 298 K
# E_ads,aq (eV) from DFT_results_summary.md §3, order DQ, TBZ, THI
DFT = {"266 A2": (-3.27, +0.63, -3.36), "118 A2": (-3.22, -0.33, -2.55), "66 A2": (-2.33, -0.63, -3.28)}

rows = [r for r in csv.DictReader(open(SRC)) if r["grid64"] == "1"]
T = np.array([[float(r[k + "_true_uM"]) for k in S3] for r in rows])
X = T / T.sum(1, keepdims=True)

def shares(tag):
    P = np.array([[float(r[k + "_" + tag]) for k in S3] for r in rows])
    return P / P.sum(1, keepdims=True)

def err(P):
    return 100 * np.abs(P - X).max(1).mean()

def correct(S, logw, gamma=1.0):
    Z = np.power(np.clip(S, 1e-12, None), gamma) * 10.0 ** (-np.asarray(logw))
    Z = np.where(S > 0, Z, 0.0)
    return Z / Z.sum(1, keepdims=True)

def loo(S, pack, p0):
    """pack(p) -> (logw, gamma). Fit on 63 conditions, read the one left out."""
    out = np.zeros_like(S); ps = []
    for i in range(len(S)):
        m = np.ones(len(S), bool); m[i] = False
        f = lambda p: ((correct(S[m], *pack(p)) - X[m]) ** 2).mean()
        r = minimize(f, p0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-9, "maxiter": 4000})
        ps.append(r.x); out[i] = correct(S[i:i + 1], *pack(r.x))[0]
    return out, np.array(ps)

L = []
def say(s=""):
    print(s); L.append(s)

say("64 held-out grid conditions, composition error = mean of largest share error (%p)")
say("PLS (mixture-trained, no affinity)      %5.1f" % err(shares("PLS")))
say("uniform 1/3                             %5.1f" % err(np.full_like(X, 1 / 3)))
for tag, name in (("NNLSb", "NNLS (BLK)"), ("NNLS", "NNLS (pipeline)")):
    S = shares(tag)
    say("")
    say("%s as is                         %5.1f   bias DQ/TBZ/THI %s" % (name.ljust(15), err(S), np.round(100 * (S - X).mean(0), 1)))
    for key, E in DFT.items():
        E = np.array(E); logw = -(E - E.mean()) / (KT * np.log(10))
        P = correct(S, logw)
        say("  + DFT %s Boltzmann weights        %5.1f   log10 w DQ/TBZ/THI %s   read mean share %s" % (
            key, err(P), np.round(logw, 1), np.round(100 * P.mean(0), 0)))
    for key, E in DFT.items():
        E = np.array(E); e0 = E - E.mean()
        P, ps = loo(S, lambda p: (-p[0] * e0, 1.0), [0.0])
        say("  + DFT %s, one fitted scale        %5.1f   beta %.3f /eV (1/kT ln10 = %.1f)  -> log10 w %s" % (
            key, err(P), ps[:, 0].mean(), 1 / (KT * np.log(10)), np.round(-ps[:, 0].mean() * e0, 2)))
    P, ps = loo(S, lambda p: (np.array([p[0], p[1], -p[0] - p[1]]), 1.0), [0.0, 0.0])
    w = ps.mean(0); say("  + free weights (2 parameters)         %5.1f   log10 w DQ/TBZ/THI %s   bias %s" % (
        err(P), np.round([w[0], w[1], -w[0] - w[1]], 2), np.round(100 * (P - X).mean(0), 1)))
    P, ps = loo(S, lambda p: (np.array([p[0], p[1], -p[0] - p[1]]), p[2]), [0.0, 0.0, 1.0])
    w = ps.mean(0); say("  + free weights + exponent (3)         %5.1f   log10 w %s  exponent %.2f" % (
        err(P), np.round([w[0], w[1], -w[0] - w[1]], 2), w[2]))

open(os.path.join(HERE, "affinity_unmix.txt"), "w").write("\n".join(L) + "\n")
