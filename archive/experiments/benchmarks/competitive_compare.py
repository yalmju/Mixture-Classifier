"""
competitive_compare.py
======================
Explain competitive adsorption by comparing a PURE-trained additive baseline
to MEASURED mixtures.  Two independent signatures:

  (1) SHAPE non-additivity  -- fit the measured mixture as a linear combo of
      pure templates (NNLS).  The residual = anything a linear mix of pures
      cannot explain: peak shifts, new surface-complex bands, orientation
      change.  A large residual = chemical non-additivity.

  (2) INTENSITY competition -- titrate one analyte with the others fixed.
      Under additivity its band grows LINEARLY with concentration.  Under
      competitive Langmuir adsorption it SATURATES, and the fixed partners
      get DISPLACED (their bands drop as the titrant rises).  Fitting the
      titration curve to Langmuir recovers the relative affinity K.

This is the "measure a few mixtures, compare to the additive prediction, and
the deviation IS the competitive-adsorption story" workflow.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import nnls, curve_fit


# ---- (1) shape non-additivity of a single measured mixture ----
def additive_residual(Y, P):
    """Fit Y ~ Σ B_i P_i (NNLS) and return B, reconstruction, relative residual."""
    Y = np.asarray(Y, float)
    B, _ = nnls(P.T, Y)
    Yhat = B @ P
    res = np.linalg.norm(Y - Yhat) / (np.linalg.norm(Y) + 1e-12)
    return B, Yhat, res


# ---- (2) titration: Langmuir vs linear ----
def _langmuir(C, Bmax, K):
    return Bmax * K * C / (1.0 + K * C)


def fit_titration(C_series, B_series):
    """Fit an analyte's band intensity vs its concentration.
    Returns dict with Langmuir (Bmax, K) and linear fits + their R²."""
    C = np.asarray(C_series, float)
    B = np.asarray(B_series, float)

    # linear (additive) through origin
    slope = np.sum(C * B) / np.sum(C * C)
    ss = lambda pred: 1 - np.sum((B - pred) ** 2) / np.sum((B - B.mean()) ** 2)
    r2_lin = ss(slope * C)

    # Langmuir (saturating, competition)
    try:
        p0 = [B.max() * 1.2, 1.0 / np.median(C)]
        popt, _ = curve_fit(_langmuir, C, B, p0=p0, maxfev=10000,
                            bounds=([0, 0], [np.inf, np.inf]))
        r2_lang = ss(_langmuir(C, *popt))
        Bmax, K = float(popt[0]), float(popt[1])
    except Exception:
        Bmax, K, r2_lang = np.nan, np.nan, np.nan

    return {"linear_slope": float(slope), "r2_linear": float(r2_lin),
            "Bmax": Bmax, "K": K, "r2_langmuir": float(r2_lang)}


def displacement(C_titrant, B_fixed_partner):
    """Correlation of a FIXED partner's band vs the titrant concentration.
    Negative slope = the titrant is displacing it (competition)."""
    C = np.asarray(C_titrant, float)
    B = np.asarray(B_fixed_partner, float)
    A = np.vstack([C, np.ones_like(C)]).T
    slope, intercept = np.linalg.lstsq(A, B, rcond=None)[0]
    return float(slope), float(intercept)
