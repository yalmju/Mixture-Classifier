"""
competitive.py
==============
Recover CONCENTRATION RATIOS of known compounds from a SERS mixture spectrum
under competitive (Langmuir) adsorption -- even when one compound dominates
and buries the others.

Key result
----------
Under competitive Langmuir adsorption the observed spectrum is

    Y = g * sum_i  A_i * theta_i * P_i ,     theta_i = K_i C_i / (1 + sum_j K_j C_j)

where g = unknown substrate gain (varies batch to batch), A_i = intrinsic SERS
brightness of compound i, K_i = surface affinity, P_i = unit-norm pure
fingerprint.  Fit Y with non-negative least squares onto the pure templates:

    Y ~= sum_i  B_i * P_i ,   B_i >= 0        (B_i = g A_i theta_i)

Then the concentration RATIO is

    C_i : C_j  =  (B_i / R_i) : (B_j / R_j) ,   R_i := A_i * K_i

and the unknown gain g AND the competition term (1 + sum K C) cancel in the
ratio.  So absolute-intensity irreproducibility does NOT break the ratio.

You only need one number per compound, R_i (a "response factor").  Get it
from a SINGLE calibration mixture of known composition:  R_i ∝ B_i^cal / C_i^cal.
No DFT/affinity needed if you can make one standard.  (If you cannot make
standards, plug DFT/MLIP affinities for K_i and a brightness estimate for A_i.)

The honest limit: when one compound dominates ~100x, the minor compounds'
B_i sink under the noise floor -> their recovered ratio gets noisy or drops
below a quantification limit.  This module flags that instead of lying.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import nnls


# --------------------------------------------------------------------------
# Forward model (for simulation / what-if)
# --------------------------------------------------------------------------
def coverages(C, K):
    """Competitive Langmuir surface coverage theta_i for concentrations C."""
    C = np.asarray(C, float)
    K = np.asarray(K, float)
    num = K * C
    return num / (1.0 + num.sum())


def forward_spectrum(C, K, A, P, gain=1.0):
    """Noise-free observed spectrum for concentrations C."""
    th = coverages(C, K)
    return gain * (A * th) @ P            # (n,)·(n,feat) -> feat


# --------------------------------------------------------------------------
# Fit: spectrum -> B_i (brightness * coverage, up to gain)
# --------------------------------------------------------------------------
def fit_B(Y, P):
    """Non-negative least squares of Y onto pure templates P (n, feat).
    Returns B (n,) >= 0 and the relative residual."""
    Y = np.asarray(Y, float)
    B, rnorm = nnls(P.T, Y)
    rel = rnorm / (np.linalg.norm(Y) + 1e-12)
    return B, rel


# --------------------------------------------------------------------------
# Calibration: one known mixture -> response factors R_i
# --------------------------------------------------------------------------
def calibrate_response(Y_cal, C_cal, P):
    """From a calibration mixture of KNOWN concentrations C_cal, recover the
    per-compound response factor R_i (defined up to a global constant, which
    is irrelevant because we only ever use ratios).

    R_i ∝ B_i^cal / C_cal_i.
    Compounds with C_cal_i == 0 are left as NaN (not calibrated)."""
    B, _ = fit_B(Y_cal, P)
    C_cal = np.asarray(C_cal, float)
    R = np.full_like(C_cal, np.nan)
    nz = C_cal > 0
    R[nz] = B[nz] / C_cal[nz]
    R[nz] /= np.nanmedian(R[nz])          # normalize (ratios only)
    return R


# --------------------------------------------------------------------------
# Recover concentration ratios of an unknown mixture
# --------------------------------------------------------------------------
def recover_ratios(Y, P, R, names=None,
                   quant_floor=0.02, n_boot=200, noise_frac=0.03, seed=0):
    """Estimate the concentration ratio of the known compounds in spectrum Y.

    R          : response factors from calibrate_response (or A*K if modeled)
    quant_floor: components whose recovered fraction < this are flagged as
                 'below quantification limit' (dominated / buried).
    n_boot     : bootstrap resamples (add synthetic noise) for uncertainty.

    Returns dict:
       ratio      : np.ndarray, concentration fractions (sum=1)
       ratio_std  : bootstrap std of each fraction
       below_LOQ  : bool mask of components under the quant floor
       residual   : NNLS relative residual (fit quality)
       B          : raw fitted coefficients
    """
    R = np.asarray(R, float)
    B, rel = fit_B(Y, P)
    C_est = B / R
    C_est = np.clip(C_est, 0, None)
    ratio = C_est / (C_est.sum() + 1e-12)

    # bootstrap uncertainty: perturb Y with noise, refit
    rng = np.random.default_rng(seed)
    scale = noise_frac * (np.max(Y) if np.max(Y) > 0 else 1.0)
    boots = []
    for _ in range(n_boot):
        Yb = Y + rng.normal(0, scale, size=Y.shape)
        Yb = np.clip(Yb, 0, None)
        Bb, _ = fit_B(Yb, P)
        Cb = np.clip(Bb / R, 0, None)
        boots.append(Cb / (Cb.sum() + 1e-12))
    boots = np.array(boots)
    ratio_std = boots.std(0)

    below = ratio < quant_floor
    out = {
        "ratio": ratio,
        "ratio_std": ratio_std,
        "below_LOQ": below,
        "residual": rel,
        "B": B,
    }
    if names is not None:
        out["pretty"] = {
            names[i]: (f"{ratio[i]*100:4.1f}% ± {ratio_std[i]*100:.1f}"
                       + ("  [<LOQ]" if below[i] else ""))
            for i in range(len(names))
        }
    return out
