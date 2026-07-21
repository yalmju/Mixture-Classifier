"""
demo_absolute.py
================
Absolute concentration (µM) read-out from SERS -- why it needs an INTERNAL
STANDARD, and how that also beats substrate irreproducibility AND saturation.

Physics
-------
Observed:  Y = g * [ A_a θ_a P_a  +  A_is θ_is P_is ]      (+ noise)
           θ_i = K_i C_i / (1 + Σ K_j C_j)     (competitive Langmuir)
g = unknown per-measurement substrate gain (varies wildly batch to batch).

Raw analyte signal  B_a = g A_a θ_a  depends on g and saturates -> you CANNOT
turn it into µM reliably.

Internal-standard ratio:
    r = B_a / B_is = (A_a K_a C_a) / (A_is K_is C_is)
-> g cancels, the competition term (1+ΣKC) cancels, saturation cancels.
With C_is fixed, r is LINEAR in C_a.  One calibration series fixes the slope;
then r -> C_a in µM.  LOD from the blank's 3σ.

Honest ceiling: if the analyte crowds the standard off the surface (θ_is sinks
under the noise), r becomes unmeasurable -> upper limit of the usable range.
"""
import numpy as np
from scipy.optimize import nnls
from numpy.polynomial import polynomial as Pnp
from synthetic import make_components
from sers_mixture import preprocess

rng = np.random.default_rng(3)
axis = np.linspace(400, 1800, 500)
# component 0 = analyte, component 1 = internal standard (distinct peaks)
prof, _ = make_components(2, axis, rng)
P = preprocess(prof)                      # unit-norm templates [analyte, IS]
P_a, P_is = P[0], P[1]

A = np.array([1.0, 1.1])                  # brightness [analyte, IS]
K = np.array([0.02, 0.03])               # affinity (1/µM)
C_is = 50.0                               # fixed internal-standard conc (µM)


def measure(C_a, gain, noise=0.03, with_is=True):
    C = np.array([C_a, C_is if with_is else 0.0])
    S = (K * C).sum()
    theta = K * C / (1 + S)
    y = gain * (A * theta) @ P
    y = y + rng.normal(0, noise * (y.max() + 1e-9), y.shape)
    return np.clip(y, 0, None)


def fit_B(Y):
    B, _ = nnls(np.vstack([P_a, P_is]).T, Y)
    return B                               # [B_a, B_is]


# ---------------------------------------------------------------
# 1. Calibration series (analyte 1..1000 µM), random gain each shot
# ---------------------------------------------------------------
conc = np.array([1, 3, 10, 30, 100, 300, 1000.0])
r_cal, braw_cal = [], []
for c in conc:
    g = rng.uniform(3, 12)                 # substrate gain varies 4x
    B = fit_B(measure(c, g))
    r_cal.append(B[0] / (B[1] + 1e-12))
    braw_cal.append(B[0])                  # raw (no IS) for comparison
r_cal = np.array(r_cal)

# linear calibration r = slope * C_a  (through ~origin)
slope = np.linalg.lstsq(conc[:, None], r_cal, rcond=None)[0][0]

# LOD: 20 blanks -> 3σ of r, converted to concentration
r_blank = []
for _ in range(20):
    g = rng.uniform(3, 12)
    B = fit_B(measure(0.0, g))
    r_blank.append(B[0] / (B[1] + 1e-12))
LOD = 3 * np.std(r_blank) / slope

print("=" * 60)
print("Absolute concentration via internal standard")
print("=" * 60)
print(f"calibration slope  r = {slope:.4e} * C(µM)")
print(f"LOD (3σ blank)     ≈ {LOD:.2f} µM")
print("-" * 60)

# ---------------------------------------------------------------
# 2. Recover unknowns -- WITH vs WITHOUT internal standard,
#    across varying substrate gain
# ---------------------------------------------------------------
# without-IS calibration (raw signal vs conc, at one 'nominal' gain)
slope_raw = np.linalg.lstsq(conc[:, None], np.array(braw_cal), rcond=None)[0][0]

print(f"{'true µM':>8} | {'IS recovered':>14} | {'no-IS recovered':>16}")
for Ctrue in [5, 20, 80, 200.0]:
    errs_is, errs_raw = [], []
    for _ in range(30):
        g = rng.uniform(3, 12)             # unknown gain per measurement
        B = fit_B(measure(Ctrue, g))
        c_is = (B[0] / (B[1] + 1e-12)) / slope          # IS-normalized
        c_raw = B[0] / slope_raw                        # raw signal
        errs_is.append(c_is); errs_raw.append(c_raw)
    m_is, s_is = np.mean(errs_is), np.std(errs_is)
    m_raw, s_raw = np.mean(errs_raw), np.std(errs_raw)
    print(f"{Ctrue:8.0f} | {m_is:6.1f} ± {s_is:4.1f}   | "
          f"{m_raw:7.1f} ± {s_raw:5.1f}")

print("-" * 60)
print("With IS: accurate & gain-robust. Without IS: mean off and huge")
print("scatter because substrate gain (3–12x) rides straight through.")
print("=" * 60)
