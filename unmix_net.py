"""
unmix_net.py
============
DL **abundance regressor** (a spectral-unmixing network) for SERS mixtures.

Where ``resnet1d.ResNet1DDetector`` answers *which* compounds are present — a
one-hot / BCE **classifier** — this answers *in what proportion*: it predicts a
continuous, non-negative, sum-to-one **composition vector** directly from an
arbitrary mixture spectrum. It is the deep counterpart of the per-pixel NNLS in
``unmix.py``: same job (recover the ratio), learned end-to-end.

Same pure-only philosophy as the rest of UNMIXR — it **never needs measured
mixtures**. Training data is synthesised from the pure templates: Dirichlet-
weighted mixtures over the components (``make_mixture_training_set``), then
corrupted with the SAME augmentation as the detector (peak shift, baseline
drift, intensity scale, noise).

Why a net instead of plain NNLS
-------------------------------
NNLS fits *fixed* templates, so instrument drift that shifts or tilts the
spectrum (peak roll, residual baseline) smears its recovered weights — the model
has no way to know the peak moved. The net is *trained to be invariant* to
exactly those corruptions, so it recovers the composition where NNLS degrades.
On perfectly linear, drift-free spectra NNLS is already optimal and the net only
ties it; the net earns its keep under realistic, non-ideal spectra.

Run the head-to-head::

    python unmix_net.py

Interface mirrors the detector (``fit(pure_spectra)`` / ``predict_abundance``)
so the pages adopt it as an unmixing backend. UI-agnostic; torch is only needed
for this module (as for ``resnet1d.py``).

Where it plugs into the app
---------------------------
Both nets enter through the drop-in factories at the bottom of this file, so the
pages stay one-liners:

* **Real-data tab** (``page_real`` → ``unmix.unmix_map``) — ``method='dl'`` swaps
  the per-pixel NNLS composition for ``ResNet1DUnmixer.predict_abundance``
  (``unmixer_from_templates``).
* **Real-data + Quantify tabs** — the "DL spectrum->B" toggle routes the single
  Y->B step through ``ResNet1DQuantifier.predict_B_one`` via
  ``calibration.quantify(..., B_predictor=...)`` (``quantifier_from_calibration``).
  The calibration fit and the θ->M inversion are untouched — NNLS always.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls

import torch
import torch.nn as nn

from resnet1d import ResNet1D
from sers_mixture import AugmentConfig, _random_baseline
from competitive import coverages, fit_B
from calibration import concentration_from_coverage, competition_report


# --------------------------------------------------------------------------
# Synthetic mixture training data — built from the pure templates only
# --------------------------------------------------------------------------
def sample_compositions(n_comp: int, n: int, rng, frac_pure: float = 0.25,
                        max_k: int = 3, alpha: float = 1.0) -> np.ndarray:
    """Random composition vectors over ``n_comp`` components (rows sum to 1).

    A fraction ``frac_pure`` are pure (one-hot); the rest are k-component
    Dirichlet mixtures (k = 2..``max_k``) placed on random component subsets —
    so the training set covers the pure vertices, the binary edges and the
    interior of the composition simplex.
    """
    max_k = min(max_k, n_comp)
    W = np.zeros((n, n_comp))
    for i in range(n):
        if rng.random() < frac_pure or max_k < 2:
            W[i, rng.integers(n_comp)] = 1.0
        else:
            k = int(rng.integers(2, max_k + 1))
            idx = rng.choice(n_comp, size=k, replace=False)
            W[i, idx] = rng.dirichlet(alpha * np.ones(k))
    return W


def corrupt_spectrum(x: np.ndarray, cfg: AugmentConfig, rng) -> np.ndarray:
    """Apply the detector's augmentation to an (already-mixed) spectrum and
    L2-normalise — the mixture-space analogue of ``augment_pure``. Keeping the
    transforms identical means the net trains on the same nuisance variation the
    detector already handles."""
    n = x.shape[0]
    if cfg.shift_max > 0:
        x = np.roll(x, int(rng.integers(-cfg.shift_max, cfg.shift_max + 1)))
    x = x * rng.uniform(cfg.intensity_lo, cfg.intensity_hi)
    if cfg.baseline_amp > 0:
        x = x + _random_baseline(n, cfg.baseline_amp, rng)
    if cfg.noise_frac > 0 and x.max() > 0:
        x = x + rng.normal(0, cfg.noise_frac * x.max(), n)
    x = np.clip(x, 0.0, None)
    nrm = np.linalg.norm(x)
    return x / nrm if nrm > 0 else x


def make_mixture_training_set(templates: np.ndarray, n: int, cfg: AugmentConfig,
                              rng, frac_pure: float = 0.25, max_k: int = 3,
                              alpha: float = 1.0):
    """Return ``(X, W)`` — corrupted mixture spectra and their true
    compositions. Mixtures are additive (``W @ templates``); the realism comes
    from ``corrupt_spectrum``. ``templates`` is ``(n_comp, n_feat)`` preprocessed
    pure spectra (row i == ``component_names[i]``)."""
    templates = np.asarray(templates, float)
    n_comp = templates.shape[0]
    W = sample_compositions(n_comp, n, rng, frac_pure, max_k, alpha)
    X = W @ templates
    X = np.stack([corrupt_spectrum(x, cfg, rng) for x in X])
    return X.astype(np.float32), W.astype(np.float32)


# --------------------------------------------------------------------------
# The abundance regressor
# --------------------------------------------------------------------------
@dataclass
class ResNet1DUnmixer:
    """ResNet1D backbone + softmax composition head, trained on synthetic
    Dirichlet mixtures of the pure templates.

    ``predict_abundance`` returns ``(n_samples, n_comp)`` rows that are
    non-negative and sum to 1 (softmax enforces both constraints natively —
    the ASC/ANC of hyperspectral unmixing). Drop-in ``fit(pure_spectra)`` /
    ``predict_abundance`` interface, parallel to ``ResNet1DDetector``.
    """
    component_names: list
    epochs: int = 40
    batch_size: int = 128
    lr: float = 1e-3
    base: int = 16
    n_train: int = 6000
    frac_pure: float = 0.25
    max_k: int = 3
    alpha: float = 1.0
    augment: AugmentConfig = field(
        default_factory=lambda: AugmentConfig(
            noise_frac=0.04, shift_max=3, baseline_amp=0.08))
    device: str = "cpu"
    seed: int = 0
    _net: ResNet1D = None
    _templates: np.ndarray = None

    def fit(self, pure_spectra: np.ndarray, verbose: bool = True):
        """``pure_spectra`` : ``(n_components, n_features)``, preprocessed
        (ALS baseline removed + L2), row i == ``component_names[i]``."""
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self._templates = np.asarray(pure_spectra, float)
        n_comp = len(self.component_names)
        X, W = make_mixture_training_set(
            self._templates, self.n_train, self.augment, rng,
            self.frac_pure, self.max_k, self.alpha)
        ds = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(W))
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size,
                                         shuffle=True)
        self._net = ResNet1D(n_comp, base=self.base).to(self.device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        self._net.train()
        for ep in range(self.epochs):
            tot = 0.0
            for xb, wb in dl:
                xb, wb = xb.to(self.device), wb.to(self.device)
                opt.zero_grad()
                pred = torch.softmax(self._net(xb), dim=1)
                loss = ((pred - wb) ** 2).mean()        # composition MSE
                loss.backward()
                opt.step()
                tot += loss.item() * len(xb)
            if verbose and (ep % 10 == 0 or ep == self.epochs - 1):
                print(f"    epoch {ep:3d}  comp-mse={tot/len(ds):.5f}")
        return self

    @torch.no_grad()
    def predict_abundance(self, spectra: np.ndarray) -> np.ndarray:
        """Composition per spectrum, shape ``(n_samples, n_comp)``; rows >= 0
        and sum to 1."""
        self._net.eval()
        x = torch.tensor(np.atleast_2d(np.asarray(spectra, float)),
                         dtype=torch.float32, device=self.device)
        return torch.softmax(self._net(x), dim=1).cpu().numpy()


# --------------------------------------------------------------------------
# NNLS baseline + comparison metrics
# --------------------------------------------------------------------------
def nnls_abundance(spectra: np.ndarray, templates: np.ndarray) -> np.ndarray:
    """Per-spectrum NNLS composition against fixed templates (the classic
    baseline this net is measured against). Rows sum to 1."""
    A = np.asarray(templates, float).T                  # (n_feat, n_comp)
    S = np.atleast_2d(np.asarray(spectra, float))
    out = np.zeros((len(S), A.shape[1]))
    for i, s in enumerate(S):
        w, _ = nnls(A, s)
        tot = w.sum()
        out[i] = w / tot if tot > 0 else w
    return out


def composition_errors(pred: np.ndarray, true: np.ndarray):
    """(MAE, RMSE) between predicted and true composition matrices."""
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    mae = float(np.abs(pred - true).mean())
    rmse = float(np.sqrt(((pred - true) ** 2).mean()))
    return mae, rmse


# ==========================================================================
# CONCENTRATION:  DL learns  spectrum -> B (= gA·θ),  a drift-robust
# replacement for the NNLS fit_B step. Absolute M then comes from YOUR
# calibration curves via calibration.concentration_from_coverage — unchanged.
# ==========================================================================
def corrupt_abs(x: np.ndarray, cfg: AugmentConfig, rng) -> np.ndarray:
    """Instrument-drift corruption that PRESERVES absolute magnitude (no L2
    norm — concentration lives in the overall scale). Peak shift + a smooth
    baseline (scaled to the signal) + proportional noise. Deliberately no
    global intensity jitter: absolute B is only identifiable at a fixed gain,
    exactly the assumption calibration.py makes."""
    n = x.shape[0]
    peak = x.max() if x.max() > 0 else 1.0
    if cfg.shift_max > 0:
        x = np.roll(x, int(rng.integers(-cfg.shift_max, cfg.shift_max + 1)))
    if cfg.baseline_amp > 0:
        x = x + _random_baseline(n, cfg.baseline_amp, rng) * peak
    if cfg.noise_frac > 0:
        x = x + rng.normal(0, cfg.noise_frac * peak, n)
    return np.clip(x, 0.0, None)


def make_quant_training_set(templates, K, gA, n, cfg, rng,
                            C_lo=1e-7, C_hi=5e-4, max_k=3):
    """(Y, B): corrupted absolute-intensity mixture spectra and their fitted
    coefficients B = gA·θ. Concentrations are log-uniform over [C_lo, C_hi] on
    random component subsets; coverage θ and B come from the SAME competitive
    Langmuir model your calibration (K, gA) defines — so no true physics is
    needed, only the calibration you fitted."""
    templates = np.asarray(templates, float)
    K = np.asarray(K, float)
    gA = np.asarray(gA, float)
    n_comp = templates.shape[0]
    max_k = min(max_k, n_comp)
    Ys, Bs = [], []
    for _ in range(n):
        C = np.zeros(n_comp)
        k = int(rng.integers(1, max_k + 1))
        idx = rng.choice(n_comp, size=k, replace=False)
        C[idx] = 10.0 ** rng.uniform(np.log10(C_lo), np.log10(C_hi), size=k)
        B = gA * coverages(C, K)
        Y = corrupt_abs(B @ templates, cfg, rng)
        Ys.append(Y); Bs.append(B)
    return np.asarray(Ys, np.float32), np.asarray(Bs, np.float32)


@dataclass
class ResNet1DQuantifier:
    """Learned ``spectrum -> B`` map (softplus head, B >= 0): a drift-robust
    stand-in for NNLS ``fit_B``. Trained on synthetic mixtures generated from
    your fitted calibration ``(K, gA)``; ``quantify`` then reuses
    ``calibration.concentration_from_coverage`` to reach absolute M. Same
    ``fit`` / ``predict_B`` shape as the rest of the module."""
    component_names: list
    K: np.ndarray
    gA: np.ndarray
    epochs: int = 60
    batch_size: int = 128
    lr: float = 1e-3
    base: int = 16
    n_train: int = 8000
    C_lo: float = 1e-7
    C_hi: float = 5e-4
    augment: AugmentConfig = field(
        default_factory=lambda: AugmentConfig(
            noise_frac=0.03, shift_max=3, baseline_amp=0.06,
            intensity_lo=1.0, intensity_hi=1.0))     # gain fixed
    device: str = "cpu"
    seed: int = 0
    _net: ResNet1D = None
    _templates: np.ndarray = None
    _bscale: float = 1.0

    def fit(self, pure_spectra: np.ndarray, verbose: bool = True):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self._templates = np.asarray(pure_spectra, float)
        n_comp = len(self.component_names)
        Y, B = make_quant_training_set(
            self._templates, self.K, self.gA, self.n_train, self.augment, rng,
            self.C_lo, self.C_hi)
        self._bscale = float(np.median(B.sum(axis=1))) + 1e-12   # typical total B
        Xt = torch.tensor(Y / self._bscale)
        Bt = torch.tensor(B / self._bscale)
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xt, Bt),
            batch_size=self.batch_size, shuffle=True)
        self._net = ResNet1D(n_comp, base=self.base).to(self.device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        self._net.train()
        for ep in range(self.epochs):
            tot = 0.0
            for xb, bb in dl:
                xb, bb = xb.to(self.device), bb.to(self.device)
                opt.zero_grad()
                pred = torch.nn.functional.softplus(self._net(xb))
                loss = ((pred - bb) ** 2).mean()
                loss.backward()
                opt.step()
                tot += loss.item() * len(xb)
            if verbose and (ep % 15 == 0 or ep == self.epochs - 1):
                print(f"    epoch {ep:3d}  B-mse={tot/len(Xt):.6f}")
        return self

    @torch.no_grad()
    def predict_B(self, spectra: np.ndarray) -> np.ndarray:
        """Non-negative coefficients B = gA·θ per spectrum, (n_samples, n_comp)."""
        self._net.eval()
        x = torch.tensor(np.atleast_2d(np.asarray(spectra, float)) / self._bscale,
                         dtype=torch.float32, device=self.device)
        B = torch.nn.functional.softplus(self._net(x)).cpu().numpy()
        return B * self._bscale

    def predict_B_one(self, y: np.ndarray) -> np.ndarray:
        """B (n_comp,) for a SINGLE spectrum — the exact shape ``competitive.fit_B``
        returns, so this backs ``calibration.quantify``'s ``B_predictor`` and makes
        the DL step a true drop-in for the NNLS Y->B fit (used by the Real-data and
        Quantify tabs)."""
        return self.predict_B(np.atleast_2d(y))[0]

    def quantify(self, spectra: np.ndarray):
        """spectrum -> absolute concentration (M) per component, via the same
        Langmuir inversion the Quantify page uses. Returns (C, theta, B)."""
        B = self.predict_B(spectra)
        theta = np.where(self.gA > 0, B / self.gA, 0.0)
        theta = np.clip(theta, 0, None)
        C = np.stack([concentration_from_coverage(t, self.K) for t in theta])
        C = np.where(np.isfinite(C), C, 0.0)
        return C, theta, B


# --------------------------------------------------------------------------
# App-facing factories — build a trained net straight from what the pages
# already hold, so the wiring in unmix.py / page_real / page_quantify stays a
# one-liner (mirrors how classify.py hands the detector a folder of references).
# --------------------------------------------------------------------------
def unmixer_from_templates(names, templates, **kw):
    """Trained ``ResNet1DUnmixer`` over ``templates`` (row i == ``names[i]``,
    preprocessed + L2). Its ``predict_abundance`` is the DL stand-in for the
    per-pixel NNLS composition in ``unmix.unmix_map`` (method='dl')."""
    return ResNet1DUnmixer(list(names), **kw).fit(np.asarray(templates, float),
                                                  verbose=False)


def quantifier_from_calibration(calib, pure_spectra, **kw):
    """Trained ``ResNet1DQuantifier`` built from a fitted ``calibration.Calibration``
    (its ``K`` / ``gA``) and the pure templates ``P``. Its ``predict_B_one`` is the
    drift-robust drop-in for ``fit_B`` behind ``calibration.quantify``'s
    ``B_predictor``. Calibration itself always stays NNLS — this only touches the
    later spectrum->B read-out."""
    q = ResNet1DQuantifier(list(calib.names), K=np.asarray(calib.K, float),
                           gA=np.asarray(calib.gA, float), **kw)
    return q.fit(np.asarray(pure_spectra, float), verbose=False)


def _log_mae(C_pred, C_true, floor=1e-8):
    """Mean |Δlog10 C| over components that are actually present (C_true>0)."""
    C_pred = np.clip(np.asarray(C_pred, float), floor, None)
    C_true = np.asarray(C_true, float)
    m = C_true > 0
    if not m.any():
        return np.nan
    return float(np.abs(np.log10(C_pred[m]) - np.log10(np.clip(C_true[m], floor, None))).mean())


# --------------------------------------------------------------------------
# Head-to-head benchmark:  DL unmixer  vs  NNLS  under instrument drift
# --------------------------------------------------------------------------
def _demo():
    from synthetic import make_components, measure_pure
    from sers_mixture import preprocess

    rng = np.random.default_rng(0)
    n_comp, n_feat = 4, 400
    axis = np.linspace(400, 1800, n_feat)
    profiles, _aff = make_components(n_comp, axis, rng)
    names = [f"C{i+1}" for i in range(n_comp)]

    # Templates = one measured pure per component, preprocessed exactly as the
    # app does (ALS baseline removed + L2). Both methods share these templates.
    pure_raw = np.array([measure_pure(profiles[c], axis, rng)
                         for c in range(n_comp)])
    templates = preprocess(pure_raw, do_baseline=True)

    print(f"[unmix_net] training ResNet1D abundance regressor on {n_comp} pure "
          f"templates (synthetic mixtures, no measured mixtures)...")
    net = ResNet1DUnmixer(names, epochs=40, n_train=6000, seed=0)
    net.fit(templates, verbose=True)

    # Held-out test mixtures built from the same templates, then corrupted with
    # increasing peak-shift / baseline drift — the regime where fixed-template
    # NNLS breaks down.
    n_test = 400
    Wtrue = sample_compositions(n_comp, n_test, np.random.default_rng(99),
                                frac_pure=0.2, max_k=3, alpha=1.0)
    clean = Wtrue @ templates

    print("\n  corruption (shift bins / baseline)     NNLS MAE   DL MAE   "
          "NNLS RMSE  DL RMSE   winner")
    print("  " + "-" * 82)
    levels = [
        ("none (ideal linear)",      AugmentConfig(shift_max=0, baseline_amp=0.0,  noise_frac=0.01)),
        ("shift 1, base 0.03",       AugmentConfig(shift_max=1, baseline_amp=0.03, noise_frac=0.02)),
        ("shift 2, base 0.05",       AugmentConfig(shift_max=2, baseline_amp=0.05, noise_frac=0.03)),
        ("shift 4, base 0.08",       AugmentConfig(shift_max=4, baseline_amp=0.08, noise_frac=0.04)),
        ("shift 6, base 0.10",       AugmentConfig(shift_max=6, baseline_amp=0.10, noise_frac=0.05)),
    ]
    for label, cfg in levels:
        rng_t = np.random.default_rng(7)
        test = np.stack([corrupt_spectrum(x.copy(), cfg, rng_t) for x in clean])
        nnls_pred = nnls_abundance(test, templates)
        dl_pred = net.predict_abundance(test)
        n_mae, n_rmse = composition_errors(nnls_pred, Wtrue)
        d_mae, d_rmse = composition_errors(dl_pred, Wtrue)
        win = "DL" if d_rmse < n_rmse else "NNLS"
        print(f"  {label:34s}  {n_mae:7.3f}  {d_mae:7.3f}   "
              f"{n_rmse:8.3f}  {d_rmse:7.3f}   {win}")
    print("\n  Read: on ideal linear spectra NNLS ties or wins (it is optimal "
          "there);\n  as peak-shift / baseline drift grow, the augmentation-"
          "trained net pulls ahead.")


def _demo_quant():
    """Concentration head-to-head, end to end on the synthetic lab (fully known
    K / A / C). Calibrate from dilution curves, then recover absolute M of the
    validation mixtures two ways: NNLS fit_B vs the DL quantifier — clean and
    under peak-shift / baseline drift."""
    from calibration import calibrate, build_synthetic_lab

    print("\n" + "=" * 78)
    print("[unmix_net] CONCENTRATION: DL quantifier vs NNLS  (synthetic lab, "
          "true K/A/C known)")
    print("=" * 78)
    lab = build_synthetic_lab(n_components=3, n_feat=400, seed=1,
                              n_dilution=7, n_validation=12)
    P, names = lab["P"], lab["names"]

    # Your calibration curves -> K, gA (this is what load_calibration_csv feeds).
    calib = calibrate(lab["dilutions"], P, names)
    print(f"  calibrated from dilution series:  K_fit={np.array2string(calib.K, precision=1)}")

    # DL quantifier trained ONLY from the fitted calibration (no true physics).
    print("  training DL quantifier (spectrum -> B) on calibration-derived "
          "mixtures...")
    q = ResNet1DQuantifier(names, K=calib.K, gA=calib.gA, epochs=60,
                           n_train=8000, seed=0).fit(P, verbose=True)

    def concentrations_nnls(Y):
        B, _ = fit_B(Y, P)
        theta = np.where(calib.gA > 0, B / calib.gA, 0.0)
        return np.where(np.isfinite(c := concentration_from_coverage(theta, calib.K)), c, 0.0)

    val, Ctrue = lab["val_specs"], lab["val_true"]
    print("\n  corruption                       NNLS logMAE   DL logMAE   winner")
    print("  " + "-" * 66)
    for label, cfg in [
        ("none (ideal)",     AugmentConfig(shift_max=0, baseline_amp=0.0, noise_frac=0.01)),
        ("shift 2, base 0.05", AugmentConfig(shift_max=2, baseline_amp=0.05, noise_frac=0.02)),
        ("shift 5, base 0.10", AugmentConfig(shift_max=5, baseline_amp=0.10, noise_frac=0.03)),
    ]:
        rng_t = np.random.default_rng(3)
        Yc = np.stack([corrupt_abs(y.copy(), cfg, rng_t) for y in val])
        n_err = np.nanmean([_log_mae(concentrations_nnls(y), Ctrue[j])
                            for j, y in enumerate(Yc)])
        C_dl, _, _ = q.quantify(Yc)
        d_err = np.nanmean([_log_mae(C_dl[j], Ctrue[j]) for j in range(len(Yc))])
        win = "DL" if d_err < n_err else "NNLS"
        print(f"  {label:32s}  {n_err:9.3f}   {d_err:9.3f}   {win}")
    print("\n  logMAE = mean |Δlog10 C| over present compounds (0.3 ≈ 2x off).")
    print("  Absolute M rides on the calibration curves either way; the DL step "
          "just\n  makes the spectrum->B fit robust to the same drift that "
          "smears NNLS.")


if __name__ == "__main__":
    _demo()
    _demo_quant()
