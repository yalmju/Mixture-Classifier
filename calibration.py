"""calibration.py
===============
Ratio → absolute **M** quantification and **Langmuir competition** judgment for
SERS mixtures. UI-agnostic (numpy / scipy only).

Why a dilution series is needed
-------------------------------
From a single mixture the NNLS fit gives, per compound,

    B_i = g · A_i · θ_i ,      θ_i = K_i C_i / (1 + Σ_j K_j C_j)          (1)

with g = substrate gain, A_i = SERS brightness, K_i = surface affinity. A single
standard (see ``competitive.calibrate_response``) only pins the product
R_i = A_i·K_i, so it recovers a concentration *ratio* — never absolute M, and it
cannot separate coverage from concentration (so it cannot judge competition).

A **dilution series** of each pure compound bends the isotherm:

    B_i(C) = (g A_i) · K_i C / (1 + K_i C)                                (2)

Fitting (2) recovers **K_i** and the product **gA_i** separately. Then for an
unknown mixture measured at the *same gain g*:

    θ_i = B_i / (g A_i)                         (coverage, from calibration)
    Σθ  known  ⇒  C_i = θ_i / ( K_i (1 − Σθ) )   (invert competitive Langmuir) (3)

which is **absolute molarity**. The gain g cancels because it is baked into the
same gA_i used for calibration and measurement. (If g drifts between the two,
use an internal standard; this module flags the assumption.)

Competition judgment
--------------------
The surface over-represents high-affinity compounds. Selectivity K_i/K_j tells
you how much compound j is *buried* per unit concentration; when the surface
"dominant" compound (max θ) differs from the solution "dominant" (max C),
competitive adsorption has flipped the apparent composition.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from competitive import fit_B, coverages, forward_spectrum


# --------------------------------------------------------------------------
# Isotherm model + fit
# --------------------------------------------------------------------------
def _langmuir_B(C, gA, K):
    """Single-compound response isotherm, eq. (2)."""
    C = np.asarray(C, float)
    return gA * K * C / (1.0 + K * C)


def fit_isotherm(C_series, B_series):
    """Fit B(C) = gA·K·C/(1+K·C) to one compound's dilution series.

    Returns (gA, K). Robust-ish initial guess from the low/high-C limits."""
    C = np.asarray(C_series, float)
    B = np.asarray(B_series, float)
    order = np.argsort(C)
    C, B = C[order], B[order]
    gA0 = max(B.max() * 1.3, 1e-9)              # plateau ~ gA
    # low-C slope B ≈ gA·K·C  ->  K ≈ slope / gA
    lo = C < np.median(C)
    slope = (np.polyfit(C[lo], B[lo], 1)[0] if lo.sum() >= 2 else B.max() / C.max())
    K0 = max(slope / gA0, 1.0 / max(C.max(), 1e-12))
    try:
        popt, _ = curve_fit(_langmuir_B, C, B, p0=[gA0, K0],
                            bounds=([0, 0], [np.inf, np.inf]), maxfev=20000)
        gA, K = float(popt[0]), float(popt[1])
    except Exception:
        gA, K = gA0, K0
    return gA, K


@dataclass
class Calibration:
    names: list[str]
    K: np.ndarray        # per-compound surface affinity  (1/M)
    gA: np.ndarray       # per-compound gain·brightness    (signal / coverage)
    # kept for plotting the fitted isotherms
    C_series: np.ndarray = None      # (K_comp, n_points)
    B_series: np.ndarray = None

    @property
    def n(self):
        return len(self.names)


def calibrate(dilutions, P, names):
    """Build a Calibration from per-compound dilution series.

    dilutions : list over compounds; each item = (C_grid, spectra) where
                spectra is (n_points, n_feat) measured for that pure compound.
    P         : (n_comp, n_feat) unit-norm pure templates (for the NNLS fit).
    """
    n = len(names)
    K = np.zeros(n)
    gA = np.zeros(n)
    Cs, Bs = [], []
    for i, (C_grid, spectra) in enumerate(dilutions):
        C_grid = np.asarray(C_grid, float)
        B_i = np.array([fit_B(y, P)[0][i] for y in spectra])   # this compound's B
        gA[i], K[i] = fit_isotherm(C_grid, B_i)
        Cs.append(C_grid); Bs.append(B_i)
    return Calibration(names=names, K=K, gA=gA,
                       C_series=np.array(Cs), B_series=np.array(Bs))


# --------------------------------------------------------------------------
# Quantify an unknown mixture: ratio → coverage → absolute M
# --------------------------------------------------------------------------
def concentration_from_coverage(theta, K):
    """Invert competitive Langmuir, eq. (3):  C_i = θ_i / (K_i (1 − Σθ))."""
    theta = np.clip(np.asarray(theta, float), 0, None)
    tot = theta.sum()
    tot = min(tot, 0.999)                      # guard against noise ≥ 1
    denom = K * (1.0 - tot)
    C = np.where(denom > 0, theta / denom, np.nan)
    return C


def quantify(Y, P, calib: Calibration, present=None):
    """Full read-out for one mixture spectrum Y.

    present : optional index list of compounds detected in Y; others are
              forced to zero so absent compounds don't get phantom M.
    Returns a dict (all arrays aligned to calib.names)."""
    B, residual = fit_B(Y, P)
    if present is not None:
        mask = np.zeros(calib.n, bool); mask[list(present)] = True
        B = np.where(mask, B, 0.0)

    theta = np.where(calib.gA > 0, B / calib.gA, 0.0)   # coverage per compound
    theta = np.clip(theta, 0, None)
    C = concentration_from_coverage(theta, calib.K)     # absolute M
    C = np.where(np.isfinite(C), C, 0.0)

    conc_ratio = C / (C.sum() + 1e-30)
    cov_ratio = theta / (theta.sum() + 1e-30)
    report = competition_report(theta, C, calib.K, calib.names)
    return {
        "names": calib.names,
        "B": B,
        "theta": theta,                 # surface coverage
        "theta_total": float(theta.sum()),
        "C": C,                         # absolute concentration (M)
        "conc_ratio": conc_ratio,       # solution ratio (sums to 1)
        "cov_ratio": cov_ratio,         # what a naive signal ratio would report
        "residual": float(residual),
        "competition": report,
    }


# --------------------------------------------------------------------------
# Langmuir competition judgment
# --------------------------------------------------------------------------
def competition_report(theta, C, K, names, sig_selectivity=2.0):
    """Compare surface coverage vs solution concentration.

    A high-affinity compound is over-represented at the surface, burying the
    others. Reports the apparent (surface) vs true (solution) dominant compound,
    and a per-compound suppression factor = K_max / K_i (how much compound i is
    under-counted per unit concentration relative to the strongest binder)."""
    theta = np.asarray(theta, float)
    C = np.asarray(C, float)
    K = np.asarray(K, float)
    active = C > 0
    if active.sum() == 0:
        return {"significant": False, "note": "no quantified compounds"}

    Kmax = K[active].max()
    suppression = {names[i]: float(Kmax / K[i]) for i in range(len(names))
                   if active[i] and K[i] > 0}
    surf_dom = names[int(np.argmax(np.where(active, theta, -np.inf)))]
    soln_dom = names[int(np.argmax(np.where(active, C, -np.inf)))]
    selectivity = float(Kmax / K[active].min()) if K[active].min() > 0 else np.inf
    significant = bool(active.sum() >= 2 and selectivity >= sig_selectivity)

    # the most-buried compound (largest suppression among the minor ones)
    buried = None
    if suppression:
        buried = max(suppression, key=suppression.get)
        if suppression[buried] <= 1.0 + 1e-6:
            buried = None

    return {
        "significant": significant,
        "surface_dominant": surf_dom,
        "solution_dominant": soln_dom,
        "flipped": bool(surf_dom != soln_dom),
        "selectivity": selectivity,          # K_max / K_min
        "suppression": suppression,          # name -> K_max/K_i (>=1)
        "buried": buried,
    }


# --------------------------------------------------------------------------
# Synthetic "lab": generate calibration standards + validation mixtures with
# a fully known ground truth, so the pipeline can be validated end-to-end.
# --------------------------------------------------------------------------
def build_synthetic_lab(n_components=3, n_feat=500, seed=0,
                        n_dilution=7, n_validation=6, gain=1.0):
    """Returns everything the Quantify page needs, with known K / A / C."""
    from synthetic import make_components
    rng = np.random.default_rng(seed)
    axis = np.linspace(400, 1800, n_feat)
    profiles, _aff = make_components(n_components, axis, rng)
    # unit-norm templates
    P = profiles / (np.linalg.norm(profiles, axis=1, keepdims=True) + 1e-12)
    names = [f"C{i+1}" for i in range(n_components)]

    # ground-truth physics
    K_true = rng.uniform(0.5e5, 8e5, n_components)        # 1/M, spread affinities
    A_true = rng.uniform(0.6, 1.6, n_components)          # brightness
    C_grid = np.geomspace(1e-7, 5e-4, n_dilution)         # M

    # per-compound dilution series (only that compound present)
    dilutions = []
    for i in range(n_components):
        specs = []
        for c in C_grid:
            C = np.zeros(n_components); C[i] = c
            y = forward_spectrum(C, K_true, A_true, P, gain=gain)
            y = y + rng.normal(0, 0.01 * (y.max() or 1.0), n_feat)
            specs.append(np.clip(y, 0, None))
        dilutions.append((C_grid, np.array(specs)))

    # validation mixtures with known concentrations
    val_specs, val_true = [], []
    for _ in range(n_validation):
        k = rng.integers(2, n_components + 1)
        idx = rng.choice(n_components, size=k, replace=False)
        C = np.zeros(n_components)
        C[idx] = 10 ** rng.uniform(-6, -3.3, size=k)      # 1e-6 .. 5e-4 M
        y = forward_spectrum(C, K_true, A_true, P, gain=gain)
        y = y + rng.normal(0, 0.01 * (y.max() or 1.0), n_feat)
        val_specs.append(np.clip(y, 0, None)); val_true.append(C.copy())

    return {
        "axis": axis, "names": names, "P": P,
        "K_true": K_true, "A_true": A_true, "gain": gain,
        "dilutions": dilutions,
        "val_specs": np.array(val_specs), "val_true": np.array(val_true),
    }


if __name__ == "__main__":
    lab = build_synthetic_lab(n_components=3, seed=1)
    calib = calibrate(lab["dilutions"], lab["P"], lab["names"])
    print("K_true :", np.array2string(lab["K_true"], precision=1))
    print("K_fit  :", np.array2string(calib.K, precision=1))
    print()
    for j in range(len(lab["val_specs"])):
        q = quantify(lab["val_specs"][j], lab["P"], calib)
        Ct = lab["val_true"][j]
        print(f"mix {j}: comp = {q['competition']['surface_dominant']}(surf)/"
              f"{q['competition']['solution_dominant']}(soln) "
              f"flip={q['competition']['flipped']} sel={q['competition']['selectivity']:.1f}")
        for i, nm in enumerate(q["names"]):
            if Ct[i] > 0 or q["C"][i] > 1e-8:
                print(f"   {nm}: true {Ct[i]:.2e} M  |  est {q['C'][i]:.2e} M "
                      f"| θ={q['theta'][i]:.3f}")
