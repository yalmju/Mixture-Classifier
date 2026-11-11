"""Fixed occupancy ratios (THI over-represented k-fold at equimolar) put into the unmixing.
x_j ∝ s_j / w_j with w fixed in advance (nothing fitted). 64 held-out grid conditions."""
import csv, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
rows = [r for r in csv.DictReader(open(os.path.join(HERE, "..", "fig4_paper_pls_2026-11-10", "heldout_maps.csv"))) if r["grid64"] == "1"]
S3 = ("DQ", "TBZ", "THI")
T = np.array([[float(r[k + "_true_uM"]) for k in S3] for r in rows]); X = T / T.sum(1, keepdims=True)
def shares(tag):
    P = np.array([[float(r[k + "_" + tag]) for k in S3] for r in rows]); return P / P.sum(1, keepdims=True)
err = lambda P: 100 * np.abs(P - X).max(1).mean()
def corr(S, w):
    Z = S / np.asarray(w, float); return Z / Z.sum(1, keepdims=True)
L = []
def say(s=""):
    print(s); L.append(s)
eq = np.ptp(T, 1) == 0
say("surface-signal share at the 4 equimolar conditions (true 33/33/33), DQ/TBZ/THI %")
for tag in ("NNLSb", "NNLS", "PLS"):
    S = shares(tag)
    say("  %-6s " % tag + "  ".join("%g µM: %s" % (T[i, 0], np.round(100 * S[i]).astype(int)) for i in np.where(eq)[0][np.argsort(T[eq, 0])]))
    say("         mean THI/DQ %.2f  THI/TBZ %.2f   (all 64: geometric mean of (s_THI/s_DQ)/(x_THI/x_DQ) %.2f, THI/TBZ %.2f)" % (
        S[eq, 2].mean() / S[eq, 0].mean(), S[eq, 2].mean() / S[eq, 1].mean(),
        np.exp(np.nanmean(np.log(np.clip(S[:, 2], 1e-4, None) / np.clip(S[:, 0], 1e-4, None) / (X[:, 2] / X[:, 0])))),
        np.exp(np.nanmean(np.log(np.clip(S[:, 2], 1e-4, None) / np.clip(S[:, 1], 1e-4, None) / (X[:, 2] / X[:, 1]))))))
say("")
W = [("as is", (1, 1, 1)), ("THI x2", (1, 1, 2)), ("THI x3", (1, 1, 3)),
     ("THI x2 vs DQ only (DFT per-area)", (1, 2, 2)), ("THI x2.4 vs DQ, x1.2 vs TBZ", (1, 2, 2.4)),
     ("binary response: THI 16x DQ, 8x TBZ", (1, 2, 16)), ("binary x0: THI 11x DQ, 5x TBZ", (1, 2.2, 11.2))]
say("composition error (%%p), PLS %.1f" % err(shares("PLS")))
say("%-38s %10s %10s" % ("weights w (DQ, TBZ, THI)", "NNLS BLK", "NNLS pipe"))
for name, w in W:
    say("%-38s %10.1f %10.1f   bias BLK %s  pipe %s" % (name + " " + str(w), err(corr(shares("NNLSb"), w)), err(corr(shares("NNLS"), w)),
        np.round(100 * (corr(shares("NNLSb"), w) - X).mean(0), 1), np.round(100 * (corr(shares("NNLS"), w) - X).mean(0), 1)))
say("")
say("over-representation in the surface share, (s_a/s_b)/(x_a/x_b), geometric mean by total concentration")
tot = T.sum(1)
for tag in ("NNLSb", "NNLS", "PLS"):
    S = np.clip(shares(tag), 1e-3, None)
    for lo, hi in ((9, 18), (21, 30), (33, 42), (45, 72)):
        m = (tot >= lo) & (tot <= hi)
        g = lambda a, b: np.exp(np.mean(np.log(S[m, a] / S[m, b] / (X[m, a] / X[m, b]))))
        say("  %-6s total %2d-%2d µM (n=%2d): THI/DQ %.2f  THI/TBZ %.2f  DQ/TBZ %.2f" % (tag, lo, hi, m.sum(), g(2, 0), g(2, 1), g(0, 1)))
open(os.path.join(HERE, "fixed_ratio.txt"), "w").write("\n".join(L) + "\n")
