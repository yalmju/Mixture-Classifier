"""
sers_mixture.py
===============
Component-wise (multi-label) classifier for SERS mixtures, trained on
PURE-SUBSTANCE spectra only.  Given a measured spectrum of an unknown
mixture of up to 3 compounds, it reports which components are present.

Approach (follows "component evidence learning + two-stage inference",
Molecules 2025, 10.3390/molecules31091412):

  Stage 0  Preprocess   : common wavenumber axis -> ALS baseline removal
                          -> L2 normalization.
  Stage 0b Augment      : from each pure spectrum synthesize many noisy /
                          baseline-shifted / peak-shifted / intensity-scaled
                          copies so the model generalizes to real conditions.
  Model               : one independent binary "evidence" head per component
                          (One-vs-Rest).  Default = RandomForest (no deep-
                          learning deps).  Optional 1D-CNN in torch_model.py
                          style is described in the README.
  Stage 1  Evidence     : per-component presence probability -> threshold ->
                          candidate set.
  Stage 2  Verify       : reconstruct the test spectrum as a NON-NEGATIVE
                          combination of the candidate pure templates (NNLS).
                          Drop components whose fitted weight is negligible,
                          keep at most `max_components` (=3) by weight.
                          This kills false positives from peak overlap.

Everything is numpy / scipy / scikit-learn.  Swap in your own data by
loading pure + (optional) test spectra as described in the README.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.optimize import nnls
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier


# --------------------------------------------------------------------------
# 0. Preprocessing
# --------------------------------------------------------------------------
_ALS_D_CACHE = {}


def _als_penalty(L, lam):
    """The smoothness penalty λ·Dᵀ·D for length L — the same for every spectrum of
    that length, so it is built once and cached (its construction, not the solve,
    is the real per-pixel cost on large maps)."""
    key = (L, lam)
    D = _ALS_D_CACHE.get(key)
    if D is None:
        d = diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2), dtype=float)
        D = csc_matrix(lam * (d @ d.T))
        _ALS_D_CACHE[key] = D
    return D


def als_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01,
                 n_iter: int = 10) -> np.ndarray:
    """Asymmetric Least Squares baseline (Eilers & Boelens).

    Returns the estimated smooth baseline; subtract it from ``y``.
    ``lam`` controls smoothness, ``p`` the asymmetry (0<p<1).
    """
    L = len(y)
    D = _als_penalty(L, lam)
    w = np.ones(L)
    z = y
    for _ in range(n_iter):
        z = spsolve(csc_matrix(diags(w, 0, dtype=float) + D), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def preprocess(spectra: np.ndarray, do_baseline: bool = True) -> np.ndarray:
    """ALS baseline removal + L2 normalization. ``spectra`` is (n, n_features)."""
    spectra = np.asarray(spectra, dtype=float)
    out = np.empty_like(spectra)
    for i, y in enumerate(spectra):
        if do_baseline:
            y = y - als_baseline(y)
        y = np.clip(y, 0, None)                 # SERS intensities are non-negative
        n = np.linalg.norm(y)
        out[i] = y / n if n > 0 else y
    return out


def resample_to_axis(x_src: np.ndarray, y_src: np.ndarray,
                     x_dst: np.ndarray) -> np.ndarray:
    """Linear-interpolate one spectrum onto a common wavenumber axis."""
    return np.interp(x_dst, x_src, y_src, left=0.0, right=0.0)


# --------------------------------------------------------------------------
# 0b. Augmentation of pure spectra  ->  training set
# --------------------------------------------------------------------------
@dataclass
class AugmentConfig:
    n_per_pure: int = 200          # augmented copies per pure spectrum
    noise_frac: float = 0.03       # gaussian noise, fraction of max intensity
    shift_max: int = 2             # peak shift, in bins (instrument drift)
    intensity_lo: float = 0.5      # random global intensity scaling range
    intensity_hi: float = 1.5
    baseline_amp: float = 0.05     # amplitude of added smooth baseline drift
    dropout_frac: float = 0.0      # randomly zero a fraction of channels
    seed: int = 0


def _random_baseline(n: int, amp: float, rng) -> np.ndarray:
    """Low-order smooth curve to mimic residual baseline drift."""
    xs = np.linspace(0, 1, n)
    a, b, c = rng.normal(0, 1, 3)
    curve = a + b * xs + c * xs ** 2
    curve -= curve.min()
    if curve.max() > 0:
        curve /= curve.max()
    return amp * curve


def augment_pure(pure: np.ndarray, labels: np.ndarray,
                 cfg: AugmentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Expand pure spectra into a noisy training set.

    ``pure``   : (n_components, n_features) preprocessed pure spectra
    ``labels`` : (n_components,) integer component ids
    Returns (X_aug, y_aug) with per-component one-vs-rest labels handled
    downstream. Labels here stay single-component integer ids.
    """
    rng = np.random.default_rng(cfg.seed)
    n_feat = pure.shape[1]
    X, y = [], []
    for spec, lab in zip(pure, labels):
        for _ in range(cfg.n_per_pure):
            s = spec.copy()
            # peak shift (roll)
            if cfg.shift_max > 0:
                s = np.roll(s, int(rng.integers(-cfg.shift_max, cfg.shift_max + 1)))
            # intensity scaling
            s = s * rng.uniform(cfg.intensity_lo, cfg.intensity_hi)
            # baseline drift
            if cfg.baseline_amp > 0:
                s = s + _random_baseline(n_feat, cfg.baseline_amp, rng)
            # additive noise
            if cfg.noise_frac > 0:
                s = s + rng.normal(0, cfg.noise_frac * s.max(), n_feat)
            # channel dropout
            if cfg.dropout_frac > 0:
                mask = rng.random(n_feat) > cfg.dropout_frac
                s = s * mask
            s = np.clip(s, 0, None)
            nrm = np.linalg.norm(s)
            if nrm > 0:
                s /= nrm
            X.append(s)
            y.append(lab)
    return np.asarray(X), np.asarray(y)


# --------------------------------------------------------------------------
# The classifier
# --------------------------------------------------------------------------
@dataclass
class SERSMixtureClassifier:
    """Multi-label SERS component detector trained on pure spectra only."""
    component_names: list[str]
    prob_threshold: float = 0.30       # stage-1 evidence threshold
    max_components: int = 3            # stage-2 cap (ternary)
    nnls_rel_threshold: float = 0.10   # drop components below this fraction of
                                        # the largest fitted weight
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    random_state: int = 0

    # filled by fit()
    _pure_templates: np.ndarray = None   # (n_comp, n_feat) preprocessed pures
    _clf: OneVsRestClassifier = None

    # ---- training -------------------------------------------------------
    def fit(self, pure_spectra: np.ndarray):
        """``pure_spectra`` : (n_components, n_features), already preprocessed,
        row i corresponds to ``component_names[i]``."""
        pure_spectra = np.asarray(pure_spectra, dtype=float)
        self._pure_templates = pure_spectra
        n_comp = len(self.component_names)
        labels = np.arange(n_comp)

        X, y = augment_pure(pure_spectra, labels, self.augment)
        # one-hot -> one-vs-rest multilabel targets
        Y = np.zeros((len(y), n_comp), dtype=int)
        Y[np.arange(len(y)), y] = 1

        base = RandomForestClassifier(
            n_estimators=200, max_depth=None,
            n_jobs=-1, random_state=self.random_state,
        )
        self._clf = OneVsRestClassifier(base)
        self._clf.fit(X, Y)
        return self

    # ---- inference ------------------------------------------------------
    def predict_proba(self, spectra: np.ndarray) -> np.ndarray:
        """Per-component presence probability, shape (n_samples, n_comp)."""
        spectra = np.atleast_2d(spectra)
        return np.asarray(self._clf.predict_proba(spectra))

    def _nnls_verify(self, spec: np.ndarray, candidates: list[int]):
        """Stage 2: fit spec ~= sum_k w_k * template_k, w_k >= 0.
        Returns (kept_components, weights_dict)."""
        if not candidates:
            return [], {}
        A = self._pure_templates[candidates].T          # (n_feat, n_cand)
        w, _ = nnls(A, spec)
        if w.max() <= 0:
            return [], {}
        rel = w / w.max()
        # keep components with meaningful weight
        keep = [c for c, r in zip(candidates, rel) if r >= self.nnls_rel_threshold]
        kw = {c: float(wt) for c, wt, r in zip(candidates, w, rel)
              if r >= self.nnls_rel_threshold}
        # cap to max_components by weight
        if len(keep) > self.max_components:
            keep = sorted(keep, key=lambda c: -kw[c])[: self.max_components]
            kw = {c: kw[c] for c in keep}
        return keep, kw

    def predict(self, spectra: np.ndarray, return_details: bool = False):
        """Two-stage prediction.

        Returns a list (one entry per input spectrum). Each entry is a dict:
          components : list[str]  detected component names
          indices    : list[int]
          proportions: dict[name -> relative weight]  (from NNLS, sums~1)
          proba      : dict[name -> stage-1 probability]
        If ``return_details`` is False, returns just the list of name-lists.
        """
        spectra = np.atleast_2d(np.asarray(spectra, dtype=float))
        probs = self.predict_proba(spectra)
        results = []
        for spec, p in zip(spectra, probs):
            cand = [i for i, pi in enumerate(p) if pi >= self.prob_threshold]
            # fall back to the single strongest if nothing crosses threshold
            if not cand:
                cand = [int(np.argmax(p))]
            keep, kw = self._nnls_verify(spec, cand)
            if not keep:                     # NNLS rejected all -> strongest evidence
                keep = [int(np.argmax(p))]
                kw = {keep[0]: 1.0}
            tot = sum(kw.values()) or 1.0
            entry = {
                "indices": keep,
                "components": [self.component_names[i] for i in keep],
                "proportions": {self.component_names[i]: kw[i] / tot for i in keep},
                "proba": {self.component_names[i]: float(p[i]) for i in keep},
            }
            results.append(entry)
        if return_details:
            return results
        return [r["components"] for r in results]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """y_true / y_pred : (n_samples, n_components) binary indicator matrices."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    exact = float(np.mean(np.all(y_true == y_pred, axis=1)))
    hamming = float(np.mean(y_true != y_pred))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "exact_match_ratio": exact,
        "hamming_loss": hamming,
        "micro_precision": float(precision),
        "micro_recall": float(recall),
        "micro_f1": float(f1),
    }


def names_to_indicator(list_of_namelists: list[list[str]],
                       component_names: list[str]) -> np.ndarray:
    idx = {n: i for i, n in enumerate(component_names)}
    M = np.zeros((len(list_of_namelists), len(component_names)), dtype=int)
    for r, names in enumerate(list_of_namelists):
        for nm in names:
            M[r, idx[nm]] = 1
    return M
