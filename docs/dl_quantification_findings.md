# Physics-informed DL for mixture quantification — findings

**Goal.** A general tool that predicts the **composition and concentration of an
arbitrary mixture** from three ingredients that are cheap to measure — the **pure**
component spectra, a **calibration** (dilution series) per component, and a **limited**
set of real mixtures — *without* having to measure every composition/concentration
combination experimentally. Pesticides (DQ / TBZ / THI) are the first test system; the
method is meant to transfer to other mixture systems.

**Why a physics-informed model.** A neural net trained on real mixtures alone would need
thousands of labelled mixtures we cannot collect. Instead the calibration fixes the
physics (`competitive.forward_spectrum`: competitive-Langmuir coverage × brightness ×
templates × gain), which can generate an unlimited, ground-truth-labelled training set
spanning the whole composition × concentration space. The few real mixtures then correct
the simulation-to-reality gap. So generalization comes from the physics; the real data
only calibrates and corrects.

---

## What was built (`dl_quantify.py`)

| Track | Idea | Functions |
|---|---|---|
| **A** | learn the inverse of the physics: **spectrum → concentration (µM)** | `simulate_mixtures`, `train_quantifier`, `predict_concentration` |
| **B** | learn the **residual** the analytic inverse misses: **surface composition → solution composition**, pretrained on physics, fine-tuned on real | `surface_composition`, `simulate_surface_solution`, `train_residual`, `predict_residual`, `fit_response_factors` (linear baseline) |

Both are UI-agnostic (numpy + lazy torch, mirroring `model_training.py`).

## How it was evaluated

- References (`Pure`) → unit templates **P**; calibration (`Ratio/results/calibration_spectra.csv`)
  → per-compound **K, gA** via `calibration.calibrate`.
- 28 real binary mixtures (`Ratio/Binary`), true ratio from the filename.
- Metric: composition error = ½·Σ|predicted − true| (0 = perfect, gain-invariant ratio).
- **Leave-one-MAP-out** cross-validation — an entire mixture (all its pixels) is held out,
  so pixels never leak between train and test.

## Results

![benchmark](dl_findings.png)

| | mean spectrum (28) | per-pixel (3,360) |
|---|---|---|
| NNLS (no correction) | 27.5% | 28.0% |
| linear response factor | **21.3%** | 23.2% |
| DL residual (this work) | 21.8% | 26.8% |

- **Track A** (spectrum→µM, physics-simulated): on real mixtures it **matches NNLS**
  (both ≈ 27.5% ratio error). Expected — A learns to invert the *same* analytic model
  NNLS already inverts. Under a *stable gain* it recovers absolute µM at
  **91–102% median recovery** on held-out synthetic data; the accuracy ceiling is set by
  substrate-gain drift (absolute magnitude ⇒ concentration is unidentifiable without an
  internal standard), not by the model.
- **Track B** (residual correction): correction **works** — 27.5% → ~21%. But the
  **DL residual does not beat the 3-parameter linear response-factor model.**
- **Per-pixel does not help** (it makes DL *worse*). A map's hundreds of pixels share one
  true ratio and one competition state; pixel-to-pixel variation is hotspot **noise**, not
  new composition/concentration information. The number of *independent* surface→solution
  examples is the number of **distinct compositions (28)**, not the pixel count. Extra
  pixels just add noise the DL overfits — and averaging (mean spectrum) beats per-pixel for
  every method.

## Interpretation (for the general goal)

The binding constraint here is **not model capacity or pixel count — it is the number of
distinct compositions/concentrations sampled.** With 3 compounds and 28 mixtures on a
simple, analytically-invertible Langmuir system, the linear response-factor model is
already the right-sized estimator, so a flexible DL cannot add value *on this system*.

This is exactly the case the physics-informed framework is built to grow beyond. DL's
advantage should appear when:

1. **The physics is not analytically invertible** — non-additive mixing, peak shifts,
   matrix effects, saturation coupling. NNLS/linear-RF have no closed form there; a DL
   trained on a richer simulator does.
2. **More components** — linear response factors scale poorly as competition couples many
   species; a learned inverse handles the joint structure.
3. **Absolute concentration across the whole space** — the calibration-grounded simulator
   gives µM for compositions/concentrations never measured, which is the actual product.

## What unlocks the DL advantage (next measurements)

- Sample **more distinct compositions**, and add a **concentration axis** (same ratio,
  different total concentration): competition is concentration-dependent, so this is where a
  learned model can exceed a concentration-independent response factor.
- Enrich the **simulator** with the measured non-idealities (feed recovery's response
  factors / competition into A, add realistic noise & baseline) to close the sim-to-real gap.
- Add an **internal standard / ratio head** so absolute µM survives gain drift.

Until then, the operating correction is the **linear response factor** (already in the
Recovery tab): ~21% composition error, principled, no overfitting.

---

*Reproduce:* `python dl_quantify.py` (synthetic self-test). Real-data validation scripts
live in the session scratchpad (`dl_validate_real.py`, `dl_validate_B.py`,
`dl_validate_B_pixel.py`).
