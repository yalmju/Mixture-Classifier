"""Final dispensed-concentration estimate with ERROR BARS.

Point estimate: one shared 2 µL droplet, area anchored on DQ+THI (they demand
the same geometry independently); TBZ is read, not forced.

Uncertainty, two components combined in quadrature on the log scale:
  - sampling: scan-line cluster bootstrap (1000x) of sum(C_app) — resampling
    whole scan lines preserves the within-line spatial correlation, matching
    how the app bootstraps its median CI
  - geometry: the anchor is the geometric mean of DQ's and THI's required
    areas; their ratio (x1.40) is the honest spread of that anchor, entering
    every substance as the SAME multiplicative band (does not move ratios)
"""
import csv
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "out"
FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)
COL = {"DQ": "#1a73e8", "TBZ": "#27A679", "THI": "#e8467c"}
TRUTH = 12.0
A_PX = 0.01

r = pickle.load(open(OUT / "res_cal.pkl", "rb"))
nb = [r.comps[i] for i in r.nonbg]
hit = r.hit
um = np.where(np.isfinite(r.conc) & (r.conc > 0), r.conc * 1e6, 0.0)
um[~hit] = 0.0

# ---- point estimate ----
S = um.sum(axis=0)                                     # Σ C_app per substance
A_req = {nm: A_PX * S[k] / TRUTH for k, nm in enumerate(nb)}
A_std = float(np.sqrt(A_req["DQ"] * A_req["THI"]))     # DQ+THI anchor
C = {nm: A_PX * S[k] / A_std for k, nm in enumerate(nb)}

# ---- sampling error: scan-line cluster bootstrap of the sums ----
y = r.coords[:, 1]
lines = [np.where(y == v)[0] for v in np.unique(y)]
rng = np.random.default_rng(0)
n_lines = len(lines)
boot = np.zeros((1000, len(nb)))
for b in range(len(boot)):
    idx = np.concatenate([lines[j] for j in rng.integers(0, n_lines, n_lines)])
    boot[b] = um[idx].sum(axis=0)
rel_boot = boot.std(axis=0, ddof=1) / boot.mean(axis=0)   # relative SE of Σ

# ---- geometry error: the DQ-vs-THI anchor spread, same for every substance ----
g = np.sqrt(A_req["THI"] / A_req["DQ"])                # x1.40 → ±log(g)/...
rel_geom = (g - 1.0) / np.sqrt(2)                      # half-spread as 1σ-ish

err = {}
for k, nm in enumerate(nb):
    rel = np.sqrt(rel_boot[k] ** 2 + rel_geom ** 2)
    err[nm] = C[nm] * rel
    print(f"{nm}: {C[nm]:.1f} ± {err[nm]:.1f} uM  "
          f"(sampling ±{100 * rel_boot[k]:.0f}%, geometry ±{100 * rel_geom:.0f}%)")
print(f"shared droplet {A_std:.2f} mm^2 (dia {2 * np.sqrt(A_std / np.pi):.2f} mm)")

with open(FIG / "conc_final.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["substance", "dispensed_estimate_uM", "err_uM", "truth_uM",
                "recovery", "assumption"])
    for nm in nb:
        w.writerow([nm, f"{C[nm]:.2f}", f"{err[nm]:.2f}", TRUTH,
                    f"{C[nm] / TRUTH:.2f}",
                    f"shared 2uL droplet {A_std:.2f} mm2"])

fig, ax = plt.subplots(figsize=(5.2, 4.2), facecolor="white")
ax.bar(nb, [C[n] for n in nb], color=[COL[n] for n in nb], width=0.6,
       yerr=[err[n] for n in nb], capsize=5,
       error_kw=dict(ecolor="#333", lw=1.4))
ax.axhline(TRUTH, color="#444", ls="--", lw=1.2)
ax.text(2.45, TRUTH + 0.35, "dispensed 12 µM", fontsize=10, ha="right",
        color="#444")
for i, n in enumerate(nb):
    ax.text(i, C[n] + err[n] + 0.35, f"{C[n]:.1f}±{err[n]:.1f}", ha="center",
            fontsize=10)
ax.set_ylabel("dispensed-equivalent concentration (µM)", fontsize=11)
ax.set_title("Mass-balance readout, shared 2 µL droplet", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "Fig_concentration.png", dpi=150, facecolor="white")
print("saved", FIG / "Fig_concentration.png")
