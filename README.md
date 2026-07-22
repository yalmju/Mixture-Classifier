# Mixture Classifier — UNMIXR

Detects **which compounds are present** in a SERS spectrum of an unknown mixture
of up to 3 components while training **only on pure-substance spectra**, then
recovers their concentration ratio / absolute M under competitive adsorption.
The detection idea ("component evidence learning + two-stage inference") is from
*SERS Mixture Recognition from Pure-Substance Spectra* (Molecules, 2025,
doi:10.3390/molecules31091412), implemented from scratch in
numpy / scipy / scikit-learn so you can read and extend every line.

## UNMIXR — the app (PyQt6)

A single native PyQt6 window; every tool is a tab (the earlier customtkinter
tools are retired — their jobs are covered natively here):

```bash
pip install -r requirements.txt
python UNMIXR.py
```

| Tab | What it does |
|-----|--------------|
| **Samples** | **Step 1 — prepare the data.** Group your reference maps into substance classes (repeat measurements of one substance are *batches* of one class: `THI`, `THI_2` → one "THI", **not** separate classes), set each map's Role (train / test / **exclude**), and choose the **preprocessing here once** (ALS baseline · Savitzky-Golay derivative · L2/SNV norm · wavenumber trim). Saved to `samples.csv` + `preprocess.json`; **Model / Predict / Real data all reuse the same preprocessing** — you set it once and never re-enter it. Core `dataset.py`. |
| **Model** | **Step 2 — train.** Pick the **algorithm** (RandomForest / ResNet1D / SVM / k-NN / Logistic Reg. / Gradient Boosting / **PLS-DA**) and the **split** (spatial = honest, random = leaky, batch / batch-CV = leave-one-batch-out, manual = Samples roles); read a live learning curve, a row-normalised confusion matrix, per-class F1, PCA and the **discriminative bands** (ANOVA F, or PLS-DA **VIP**). Export saves every plot as PNG + CSV and the fitted model for reuse. Preprocessing comes from Samples. Core `model_training.py`. |
| **Predict** | Load ONE unknown sample map → its component ratio from **per-pixel NNLS** averaged over the map (recovers minor components a mean spectrum buries), plus a per-pixel dominant-component map. Core `predict.py`. |
| **Quantify** | **Ratio → absolute M + Langmuir competition**. Fits each compound's Langmuir isotherm from a dilution series (`K_i` and `gA_i` separately), inverts competitive adsorption to absolute **molarity**, and judges competition — surface- vs solution-dominant, selectivity `K_max/K_min`, which compound is buried. Core `calibration.py`. |
| **Real data** | Unmix ONE test map against your references by **NNLS** (fixed templates) or **MCR-ALS** (refines the component spectra), and read it four ways: per-substance **intensity maps** (where each substance is), a per-pixel **reliability** map (reconstruction R²), a measured-vs-reconstructed **validation** plot (+ spectral-angle), and the overall **composition pie**. Generalised to any substances — the DQ / THI / TBZ pesticides are just the shipped example. Core `unmix.py`. |

Every page has **Load… / Data folder…** (feed your own maps) and **Export…**
(results CSV + figures as PNG). CSV formats are documented in `io_utils.py`; drop
a PNG at `assets/icon.png` for the app icon. Rename the app via `APP_NAME` in
`UNMIXR.py`.

## Why this design

You cannot measure every A+B / A+B+C combination — the count explodes and SERS
mixtures are **non-additive** (competitive surface adsorption means a
high-affinity compound suppresses the others). So the model learns each pure
compound's fingerprint, then:

1. **Evidence (stage 1)** — an independent binary classifier per compound gives a
   presence probability; threshold it to get candidate components.
2. **Verify (stage 2)** — reconstruct the spectrum as a non-negative combination
   (NNLS) of the candidate pure templates; components with negligible fitted
   weight are dropped (at most 3 kept), removing false positives from overlapping
   peaks.

The real-data lesson (see the **Real data** page): on the DQ/THI/TBZ maps,
THI's ~13× SERS response buries DQ/TBZ in the **mean** spectrum, so a mean-spectrum
classifier misses them. **Per-pixel NNLS voting** recovers them (F1 0.73→0.92) by
using the spatial variation the mean discards — the bottleneck was information, not
model capacity.

## Files

**App:**
- `UNMIXR.py` — the PyQt6 app (entry point); every tab is native with embedded matplotlib.

**UI-agnostic cores** (numpy / scipy / sklearn — reusable from any front-end):
- `dataset.py` — turn a data folder into a dataset spec: discover reference maps, group batches into classes, read/write `samples.csv` and the mixture manifest (Samples page).
- `model_training.py` — train a classifier on the reference maps with honest spatial / batch / random splits (Model page).
- `predict.py` — apply the references to an unknown sample; per-pixel NNLS composition + dominant-component map (Predict page).
- `calibration.py` — Langmuir isotherm fit, coverage→M inversion, competition judgment (Quantify page). `build_synthetic_lab()` gives a fully-known ground truth — `python calibration.py`.
- `real_data.py` — load the maps, run the detection strategies + response-factor calibration (Real data page). `python real_data.py` prints the strategy table.

**Analysis engine:**
- `sers_mixture.py` — component-DETECTION pipeline (which compounds present).
- `competitive.py` — concentration-RATIO recovery under competitive Langmuir adsorption.
- `competitive_compare.py` — explain competitive adsorption from measured mixtures (additive residual + Langmuir-vs-linear + partner displacement).
- `synthetic.py` — synthetic SERS generator (competitive adsorption) so everything runs with zero real data.
- `resnet1d.py` — ResNet1D multi-label detector (PyTorch), the Molecules-2025 architecture; the "ResNet1D" Model backend.

**Folders:** `examples/` sample CSVs · `docs/` project status / reference pages · `assets/` the app icon.

## Concentration ratios under competitive adsorption (`competitive.py`)

The observed spectrum is `Y = g · Σ A_i θ_i P_i` with unknown gain `g`, brightness
`A_i`, competitive coverage `θ_i = K_i C_i / (1 + Σ K_j C_j)`. Fitting `Y` to the
pure templates by NNLS gives `B_i = g A_i θ_i`, and

    C_i : C_j  =  (B_i / R_i) : (B_j / R_j),   R_i = A_i K_i

so **gain and the competition term cancel in the ratio** — substrate
irreproducibility does not break it. You get `R_i` from **one calibration mixture
of known composition** (`R_i ∝ B_i^cal / C_i^cal`); no DFT needed if you can make
one standard. Two honest limits:

1. **Surface saturation** — if the dominant saturates the surface (`Σθ → 1`), its
   SERS signal plateaus and its concentration is only a lower bound. Fix: **dilute**
   until `Σθ < ~0.5`, then quantify.
2. **Noise floor** — a minor whose `K·C` is ~1000× below the dominant's falls under
   the noise; no algorithm recovers what isn't in the signal. Fixes: raise SNR
   (average M spectra ⇒ noise/√M, or a better substrate), use a clean marker band,
   or **per-pixel voting / standard addition**.

## Absolute concentration in µM

Ratios cancel the unknown substrate gain, so they need no anchor; absolute µM does.
Two routes, both in `calibration.py` / the **Quantify** page:

- **Dilution-series isotherm** — fit `B_i(C) = gA_i·K_iC/(1+K_iC)` per compound to
  recover `K_i` and `gA_i` separately, then invert competitive Langmuir for a mixture:
  `C_i = θ_i / (K_i(1−Σθ))`. Assumes a stable gain between calibration and measurement.
- **Internal standard** — spike a fixed known reference; the ratio
  `r = B_analyte / B_internal` is linear in `C_a` and cancels gain, competition, and
  saturation. One calibration series fixes the slope; LOD from the blank's 3σ.

## Knobs that matter

- `prob_threshold` — lower = more sensitive (higher recall, more false positives).
- `nnls_rel_threshold` — the stage-2 cutoff; lower keeps weaker components.
- `AugmentConfig` (`noise_frac`, `shift_max`, `baseline_amp`) — match to your
  instrument's real batch-to-batch variability. **The single biggest lever** for
  how well the model transfers to real data.

## Where to take it next

- **Marker-band / hierarchical detection** (JACS 2025, 147, 6654 style): detect each
  compound by its discriminative bands (DQ ~1176/1572, THI ~1366, TBZ ~1010 cm⁻¹) with
  an SNR test, then quantify by band ratios — targets the buried minor bands directly.
- **Concentration calibration** on a real dilution series (100 µM base, wide ratios)
  to feed the Quantify page with real M curves.
- Swap the RandomForest heads for the 1D-CNN (`resnet1d.py`) once real mixture
  training data is available — deep models pay off with more data.
