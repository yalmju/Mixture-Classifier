"""
synthetic.py
============
Generate realistic synthetic SERS spectra so the pipeline runs with zero
real data.  Each compound has a few Lorentzian peaks.  MIXTURES are built
with a *competitive-adsorption* model: components compete for the metal
surface, so a high-affinity compound suppresses the others.  This makes
mixture spectra NON-additive (A+B != A + B), which is the hard, realistic
case -- and the model never sees a single mixture during training.
"""

from __future__ import annotations
import numpy as np


def lorentzian(x, center, width, amp):
    return amp * (width ** 2) / ((x - center) ** 2 + width ** 2)


def make_components(n_components: int, axis: np.ndarray, rng):
    """Random but fixed peak set per component. Returns (n_comp, n_feat),
    plus a per-component surface 'affinity' used for competitive mixing."""
    comps = np.zeros((n_components, len(axis)))
    affinities = rng.uniform(0.5, 2.0, n_components)
    for c in range(n_components):
        n_peaks = rng.integers(3, 7)
        centers = rng.uniform(axis.min() + 50, axis.max() - 50, n_peaks)
        widths = rng.uniform(6, 18, n_peaks)
        amps = rng.uniform(0.4, 1.0, n_peaks)
        for ctr, w, a in zip(centers, widths, amps):
            comps[c] += lorentzian(axis, ctr, w, a)
    return comps, affinities


def measure_pure(comp_profile, axis, rng, noise=0.02, baseline=0.05):
    """One noisy measured realization of a pure spectrum."""
    s = comp_profile.copy()
    s = s * rng.uniform(0.7, 1.3)                       # intensity variation
    xs = np.linspace(0, 1, len(axis))
    s = s + baseline * (rng.uniform(0, 1) + rng.uniform(-1, 1) * xs) * s.max()
    s = s + rng.normal(0, noise * s.max(), len(axis))
    return np.clip(s, 0, None)


def measure_mixture(comps, affinities, present, axis, rng,
                    noise=0.02, baseline=0.05):
    """Competitive-adsorption mixture of the components in ``present``.

    Surface coverage of each component ~ affinity * concentration, normalized
    so total coverage = 1  (Langmuir-style competition).  The observed
    spectrum is the coverage-weighted sum -> non-additive suppression.
    """
    present = list(present)
    conc = rng.uniform(0.3, 1.0, len(present))          # random amounts
    aff = affinities[present]
    coverage = aff * conc
    coverage = coverage / coverage.sum()                # competition
    s = np.zeros(len(axis))
    for w, c in zip(coverage, present):
        s += w * comps[c]
    s = s * rng.uniform(0.7, 1.3)
    xs = np.linspace(0, 1, len(axis))
    s = s + baseline * (rng.uniform(0, 1) + rng.uniform(-1, 1) * xs) * s.max()
    s = s + rng.normal(0, noise * s.max(), len(axis))
    return np.clip(s, 0, None), dict(zip(present, coverage))


def build_dataset(n_components=6, n_feat=500, seed=0,
                  n_test_per_size=40):
    """Returns everything needed for a demo run.

    pure_raw   : (n_comp, n_feat) one clean measured pure spectrum each
    test_specs : (N, n_feat) mixture spectra (sizes 1,2,3)
    test_true  : list[list[int]] true component indices per test spectrum
    axis, names
    """
    rng = np.random.default_rng(seed)
    axis = np.linspace(400, 1800, n_feat)               # cm^-1
    profiles, affinities = make_components(n_components, axis, rng)
    names = [f"C{i+1}" for i in range(n_components)]

    # one measured pure spectrum per component (training input)
    pure_raw = np.array([measure_pure(profiles[c], axis, rng)
                         for c in range(n_components)])

    # test set: mixtures of size 1, 2, 3
    test_specs, test_true = [], []
    for size in (1, 2, 3):
        for _ in range(n_test_per_size):
            present = rng.choice(n_components, size=size, replace=False)
            spec, _cov = measure_mixture(profiles, affinities, present,
                                         axis, rng)
            test_specs.append(spec)
            test_true.append(sorted(int(i) for i in present))
    return {
        "axis": axis,
        "names": names,
        "profiles": profiles,
        "pure_raw": pure_raw,
        "test_specs": np.array(test_specs),
        "test_true": test_true,
    }
