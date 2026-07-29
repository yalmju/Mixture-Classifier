"""dl_quantify.py — physics-informed DEEP quantification (track A).

The classifier is strong on pure substances but weak on mixtures, and the NNLS +
Langmuir pipeline handles mixtures only under its own assumptions (linear mixing,
single competitive isotherm, gain cancels in ratios). This module trains a neural
regressor **spectrum → concentration (µM)** on mixtures the physics forward model
generates, so it can learn the non-idealities the closed-form inverse misses while
staying grounded in the same physics (competitive Langmuir + templates + gain).

Why synthetic training is sound here
------------------------------------
Real known-ratio mixtures are few (~dozens) — far too few to fit a spectrum→conc
regressor without overfitting. But ``competitive.forward_spectrum`` is an exact
physical generator: given per-compound affinity K, brightness A and unit templates P,
it produces the noise-free mixture spectrum for ANY concentration vector. Sampling C
(and gain / noise / baseline nuisance) across the whole operating range yields an
unlimited, ground-truth-labelled training set. K and A come straight from a fitted
calibration, so calibration is *baked into* the simulator — "Calib까지" for free.

The honest limit: absolute concentration needs the signal magnitude, and substrate
gain drifts batch to batch. We train across a gain band so the model is robust rather
than brittle, and report validation error truthfully; an internal-standard / ratio
head (and the residual-correction model, track B) build on top of this.

UI-agnostic: numpy only at import; torch is imported lazily inside the train/predict
paths (mirrors model_training.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from competitive import forward_spectrum


# --------------------------------------------------------------------------
# 1. Physics-informed synthetic mixture generator
# --------------------------------------------------------------------------
def simulate_mixtures(P, K, A, n, rng, *, c_lo=1e-7, c_hi=1e-3, p_present=0.6,
                      gain_lo=0.7, gain_hi=1.4, noise=0.01, baseline=0.0,
                      min_present=1):
    """Generate ``n`` physics-forward mixture spectra with known concentrations.

    P (n_comp, n_feat) unit templates, K (1/M), A (brightness) from a calibration.
    Each sample: a random subset of compounds is present (each w.p. ``p_present``,
    at least ``min_present``); present concentrations are log-uniform in [c_lo, c_hi];
    a random gain in [gain_lo, gain_hi] models substrate drift; Gaussian ``noise``
    (fraction of the spectrum max) and a smooth ``baseline`` ramp are added.

    Returns (X (n, n_feat) float32 spectra, C (n, n_comp) concentrations in M)."""
    P = np.asarray(P, float); K = np.asarray(K, float); A = np.asarray(A, float)
    n_comp, n_feat = P.shape
    X = np.zeros((n, n_feat), np.float32)
    C = np.zeros((n, n_comp), np.float32)
    axis = np.linspace(0.0, 1.0, n_feat)
    for i in range(n):
        present = rng.random(n_comp) < p_present
        if present.sum() < min_present:                 # force at least min_present
            present[rng.integers(n_comp)] = True
        c = np.zeros(n_comp)
        c[present] = 10 ** rng.uniform(np.log10(c_lo), np.log10(c_hi), present.sum())
        gain = rng.uniform(gain_lo, gain_hi)
        y = forward_spectrum(c, K, A, P, gain=gain)
        if baseline > 0:                                # random smooth ramp/curve
            b = baseline * (y.max() or 1.0)
            y = y + b * (rng.uniform(-1, 1) * axis + rng.uniform(-1, 1) * axis ** 2)
        if noise > 0:
            y = y + rng.normal(0, noise * (y.max() or 1.0), n_feat)
        X[i] = np.clip(y, 0, None); C[i] = c
    return X, C


def _to_logC(C, floor=1e-9):
    """Concentration (M) → log10 target; absent compounds map to log10(floor)."""
    return np.log10(np.clip(np.asarray(C, float), floor, None)).astype(np.float32)


def _from_logC(logC, floor=1e-9, thresh_decades=0.5):
    """Inverse of _to_logC; values within ``thresh_decades`` of the floor → 0 (absent)."""
    C = 10.0 ** np.asarray(logC, float)
    C[np.asarray(logC) <= np.log10(floor) + thresh_decades] = 0.0
    return C


# --------------------------------------------------------------------------
# 2. Model + input scaling
# --------------------------------------------------------------------------
@dataclass
class DLQuantifier:
    """A trained regressor plus everything predict() needs (numpy-portable)."""
    state: dict                 # torch state_dict (tensors)
    x_mean: np.ndarray          # per-feature input standardisation
    x_std: np.ndarray
    names: list
    n_feat: int
    hidden: tuple
    floor: float = 1e-9

    @property
    def n_comp(self):
        return len(self.names)


def _build_net(n_feat, n_comp, hidden):
    import torch.nn as nn
    layers, d = [], n_feat
    for h in hidden:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)]
        d = h
    layers += [nn.Linear(d, n_comp)]                    # → log10 C per compound
    return nn.Sequential(*layers)


def train_quantifier(P, K, A, names, *, n_train=20000, n_val=4000, hidden=(256, 128),
                     epochs=60, batch=256, lr=1e-3, seed=0, sim=None, progress=None):
    """Train the spectrum→log10C regressor on physics-simulated mixtures.

    ``sim`` overrides simulate_mixtures kwargs (dict). Returns (DLQuantifier, history)
    where history has per-epoch train/val loss and val log-MAE."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    rng = np.random.default_rng(seed)
    skw = dict(sim or {})
    Xtr, Ctr = simulate_mixtures(P, K, A, n_train, rng, **skw)
    Xva, Cva = simulate_mixtures(P, K, A, n_val, rng, **skw)
    x_mean = Xtr.mean(0); x_std = Xtr.std(0) + 1e-8
    Xtr_s = (Xtr - x_mean) / x_std
    Xva_s = (Xva - x_mean) / x_std
    Ytr = _to_logC(Ctr); Yva = _to_logC(Cva)

    torch.manual_seed(seed)
    net = _build_net(P.shape[1], len(names), hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = torch.nn.SmoothL1Loss()                     # robust to the absent-floor mass
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr_s), torch.from_numpy(Ytr)),
                    batch_size=batch, shuffle=True)
    Xva_t = torch.from_numpy(Xva_s); Yva_t = torch.from_numpy(Yva)
    hist = {"train": [], "val": [], "val_mae": []}
    for ep in range(epochs):
        net.train(); tot = 0.0
        for xb, yb in dl:
            opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        net.eval()
        with torch.no_grad():
            pv = net(Xva_t)
            vl = float(lossf(pv, Yva_t))
            vmae = float((pv - Yva_t).abs().mean())
        hist["train"].append(tot / len(Xtr_s)); hist["val"].append(vl)
        hist["val_mae"].append(vmae)
        if progress and (ep % 5 == 0 or ep == epochs - 1):
            progress(f"epoch {ep + 1}/{epochs}  val log-MAE {vmae:.3f} decades")
    state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    return (DLQuantifier(state=state, x_mean=x_mean, x_std=x_std, names=list(names),
                         n_feat=P.shape[1], hidden=tuple(hidden)), hist)


def predict_concentration(model: DLQuantifier, X):
    """Spectra (m, n_feat) → concentrations (m, n_comp) in M (absent → 0)."""
    import torch
    net = _build_net(model.n_feat, model.n_comp, model.hidden)
    net.load_state_dict(model.state); net.eval()
    Xs = (np.asarray(X, np.float32) - model.x_mean) / model.x_std
    with torch.no_grad():
        logC = net(torch.from_numpy(Xs.astype(np.float32))).numpy()
    return _from_logC(logC, floor=model.floor)


# --------------------------------------------------------------------------
# 2b. Track B — residual correction: NNLS surface composition → solution composition
#
# Track A learns to invert the SAME forward model NNLS already inverts, so on real data
# it only matches NNLS. The real known-ratio mixtures encode the model's *errors*
# (competitive THI over-representation, etc.). Track B learns that residual directly:
# it maps the observed surface composition to the true solution composition. We pretrain
# the map on physics-simulated (surface, solution) pairs, then fine-tune on the few real
# mixtures — transfer learning so ~dozens of labelled mixtures don't overfit.
# --------------------------------------------------------------------------
def _ratio(v):
    v = np.clip(np.asarray(v, float), 0, None); s = v.sum()
    return v / s if s > 0 else v


def surface_composition(spectra, P):
    """NNLS surface-composition ratio per spectrum: (m, n_feat) → (m, n_comp)."""
    from competitive import fit_B
    P = np.asarray(P, float)
    out = []
    for y in np.atleast_2d(np.asarray(spectra, float)):
        B, _ = fit_B(y / (np.linalg.norm(y) + 1e-12), P)
        out.append(_ratio(B))
    return np.array(out)


def simulate_surface_solution(P, K, A, n, rng, **sim):
    """Physics-simulated (surface composition, solution composition) pairs — the pretrain
    set for the residual corrector."""
    X, C = simulate_mixtures(P, K, A, n, rng, **sim)
    return surface_composition(X, P), np.array([_ratio(c) for c in C])


def fit_response_factors(surf, sol, ref=0):
    """Linear baseline: per-substance response factor r (observed_i ∝ r_i·true_i),
    anchored so r[ref]=1. Correct with ``apply_response_factors``."""
    surf = np.asarray(surf, float); sol = np.asarray(sol, float)
    n = surf.shape[1]; logr = np.zeros(n)
    for i in range(n):
        d = []
        for o, t in zip(surf, sol):
            if o[i] > 0 and o[ref] > 0 and t[i] > 0 and t[ref] > 0:
                d.append(np.log(o[i] / o[ref]) - np.log(t[i] / t[ref]))
        logr[i] = np.mean(d) if d else 0.0
    r = np.exp(logr - logr[ref])
    return r


def apply_response_factors(surf, r):
    """Solution composition from surface composition via response factors: true ∝ obs/r."""
    surf = np.atleast_2d(np.asarray(surf, float))
    return np.array([_ratio(np.divide(o, r, out=np.zeros_like(o), where=r > 0)) for o in surf])


def _residual_net(n_comp, hidden):
    import torch.nn as nn
    layers, d = [], n_comp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, n_comp)]                    # logits → softmax composition
    return nn.Sequential(*layers)


def train_residual(surf, sol, *, pretrain=None, hidden=(16,), epochs=500, lr=5e-3,
                   weight_decay=1e-3, seed=0):
    """Fit surface→solution composition. ``pretrain=(surf0, sol0)`` warms up on the
    physics-simulated pairs first, then this fine-tunes on (surf, sol). Returns a dict
    with the torch state + shapes (portable to ``predict_residual``)."""
    import torch
    n_comp = np.asarray(surf).shape[1]
    torch.manual_seed(seed)
    net = _residual_net(n_comp, hidden)
    logsm = torch.nn.LogSoftmax(dim=1)

    def _fit(S, Y, ep, lr_):
        opt = torch.optim.Adam(net.parameters(), lr=lr_, weight_decay=weight_decay)
        St = torch.tensor(np.asarray(S, np.float32)); Yt = torch.tensor(np.asarray(Y, np.float32))
        for _ in range(ep):
            net.train(); opt.zero_grad()
            pred = logsm(net(St)).exp()
            loss = (pred - Yt).abs().sum(1).mean()      # L1 on composition
            loss.backward(); opt.step()

    if pretrain is not None:
        _fit(pretrain[0], pretrain[1], epochs, lr)
    if len(surf):
        _fit(surf, sol, epochs, lr * 0.5)               # gentler fine-tune on real
    net.eval()
    return dict(state={k: v.clone() for k, v in net.state_dict().items()},
                n_comp=n_comp, hidden=tuple(hidden))


def predict_residual(model, surf):
    """Surface composition (m, n_comp) → corrected solution composition (m, n_comp)."""
    import torch
    net = _residual_net(model["n_comp"], model["hidden"])
    net.load_state_dict(model["state"]); net.eval()
    with torch.no_grad():
        pred = torch.nn.functional.softmax(
            net(torch.tensor(np.atleast_2d(np.asarray(surf, np.float32)))), dim=1)
    return pred.numpy()


# --------------------------------------------------------------------------
# 3. Self-test: train on a synthetic lab, report recovery on a held-out set
# --------------------------------------------------------------------------
def _recovery_report(C_true, C_pred, names):
    """Per-compound: median recovery % (pred/true) over samples where the compound is
    truly present, and detection precision/recall for presence."""
    C_true = np.asarray(C_true); C_pred = np.asarray(C_pred)
    out = {}
    for i, nm in enumerate(names):
        t = C_true[:, i]; p = C_pred[:, i]
        present = t > 0
        rec = (p[present] / t[present]) * 100 if present.any() else np.array([])
        tp = int(((p > 0) & present).sum()); fp = int(((p > 0) & ~present).sum())
        fn = int(((p == 0) & present).sum())
        out[nm] = dict(
            median_recovery=float(np.median(rec)) if rec.size else float("nan"),
            n_present=int(present.sum()),
            precision=tp / (tp + fp) if tp + fp else float("nan"),
            recall=tp / (tp + fn) if tp + fn else float("nan"))
    return out


if __name__ == "__main__":
    from calibration import build_synthetic_lab, calibrate

    lab = build_synthetic_lab(n_components=3, n_feat=400, seed=1)
    calib = calibrate(lab["dilutions"], lab["P"], lab["names"])
    print("fitted K :", np.array2string(calib.K, precision=1))
    print("fitted gA:", np.array2string(calib.gA, precision=1))
    # brightness A implied by the calibration (gA / gain); K from the fit
    A_est = calib.gA / lab["gain"]

    model, hist = train_quantifier(
        lab["P"], calib.K, A_est, lab["names"],
        n_train=12000, n_val=3000, epochs=40,
        sim=dict(noise=0.01, baseline=0.02), progress=print)

    rng = np.random.default_rng(99)
    Xte, Cte = simulate_mixtures(lab["P"], calib.K, A_est, 3000, rng,
                                 noise=0.01, baseline=0.02)
    Cpred = predict_concentration(model, Xte)
    rep = _recovery_report(Cte, Cpred, lab["names"])
    print("\nheld-out recovery (physics-simulated):")
    for nm, r in rep.items():
        print(f"  {nm}: median recovery {r['median_recovery']:.0f}%  "
              f"precision {r['precision']:.2f}  recall {r['recall']:.2f}  "
              f"(n={r['n_present']})")
    print(f"\nfinal val log-MAE: {hist['val_mae'][-1]:.3f} decades "
          f"(≈ ×{10 ** hist['val_mae'][-1]:.2f} concentration factor)")
