# Mixture Classifier

![dashboard preview](preview_dashboard.png)

## SERS Mixture Component Detector (trained on pure spectra only)

Detects **which compounds are present** in a SERS spectrum of an unknown
mixture of up to 3 components — while training **only on pure-substance
spectra**. This is the "component evidence learning + two-stage inference"
idea from *SERS Mixture Recognition from Pure-Substance Spectra* (Molecules,
2025, doi:10.3390/molecules31091412), implemented from scratch in
numpy / scipy / scikit-learn so you can read and extend every line.

## UNMIXR — SERS mixture analysis suite (PyQt6)

The suite front-end. A light PyQt6 app, structured as a three-stage pipeline —
**Train → Calibrate → Analyze** — over a top pill navigation (no sidebar):

```bash
python unmixr.py
```

| Stage | Page | What it does |
|-------|------|--------------|
| **1. Train** | **Model** | Train the mixture classifier on synthetic pure spectra and read its metrics **live** — KPI tiles (micro F1 / precision / recall / exact-match) over a 2×2 plot grid: **PCA scatter**, **confusion matrix**, per-component **precision / recall / F1**, reference templates. Core `model_metrics.py`. |
| **2. Calibrate** | **Quantify** | **Ratio → absolute M + Langmuir competition** (real). Fits each compound's Langmuir isotherm from a dilution series (`K_i` and `gA_i` separately), inverts competitive adsorption to absolute **molarity**, and judges competition — surface- vs solution-dominant, selectivity `K_max/K_min`, which compound is buried. Core `calibration.py`. |
| **3. Analyze** | **Discriminator** | The real DQ / THI / TBZ maps in one screen: **single-component 4-class confusion** (per pixel, ~100%), a **detection-strategy comparison** (RF-on-mean vs **per-pixel NNLS voting** vs matched-filter — per-pixel wins: F1 0.73→0.92, exact 20%→80% by using the spatial info the mean discards), the per-pixel **detection grid**, **composition confusion**, and **response-factor correction**. `Data folder…` re-points to the data; **Mixture tool** / **Map tool** buttons launch the detailed customtkinter apps (`sers_app.py`, `sers_discriminator_ctk.py`) in their own window. Core `real_data.py`. |

All three pages are native PyQt6 (embedded matplotlib); the two detailed tools are
still customtkinter and launch as separate processes (Qt port is the next step).
Rename the app by editing `APP_NAME` in `unmixr.py`.

Two UI-agnostic cores (numpy / scipy / sklearn only) hold the science so any
front-end can reuse them: `model_metrics.py` (training + evaluation) and
`calibration.py` (Langmuir isotherm fit, coverage→M inversion, competition
judgment; `build_synthetic_lab()` provides a fully-known ground truth to
validate the recovery — `python calibration.py`).

### Legacy launcher — `SERS_SUITE.py` (customtkinter)

The earlier all-customtkinter shell (workflow rail + embedded tools + placeholder
per-pixel report) still runs with `python SERS_SUITE.py`; UNMIXR supersedes it.
Both `sers_app.py` and `sers_discriminator_ctk.py` also run standalone.

## Why this design

You cannot measure every A+B / A+B+C combination — the count explodes and
SERS mixtures are **non-additive** (competitive surface adsorption means a
high-affinity compound suppresses the others). So the model learns each
pure compound's fingerprint, then:

1. **Evidence (stage 1)** — an independent binary classifier per compound
   gives a presence probability. Threshold it to get candidate components.
2. **Verify (stage 2)** — reconstruct the spectrum as a non-negative
   combination (NNLS) of the candidate pure templates. Components with
   negligible fitted weight are dropped; at most 3 are kept. This removes
   false positives caused by overlapping peaks.

## Files

- `sers_mixture.py` — component-DETECTION pipeline (which compounds present).
- `competitive.py` — concentration-RATIO recovery under competitive Langmuir
  adsorption (the "one dominates and buries the others" problem).
- `synthetic.py` — realistic synthetic SERS generator with competitive
  adsorption, so the demos run with zero real data.
- `demo.py` — detection demo (multi-label, trained on pure spectra).
- `demo_quant.py` — quantification demo: shows where ratio recovery works and
  the two walls (surface saturation, noise floor) where it breaks — honestly.
- `resnet1d.py` — **ResNet1D** multi-label detector (PyTorch), the Molecules-2025
  "evidence learning" architecture; drop-in for the RF heads. `demo_resnet.py`
  trains it on pure spectra and compares to the RF baseline.
- `competitive_compare.py` — explain competitive adsorption from measured
  mixtures: additive-baseline shape residual + Langmuir-vs-linear titration fit
  + partner-displacement. `demo_compare.py` shows the full read-out.
- `run_on_my_data.py` — template to run detection on **your** CSV spectra.

### Explaining competitive adsorption from your own mixtures (`competitive_compare.py`)

Train the additive baseline on pure spectra, then measure a titration (vary one
analyte, hold the others fixed). Two signatures pop out: the titrated band
**saturates** (Langmuir fits, additive/linear fails) and the fixed partners get
**displaced** (their bands drop as the titrant rises). That deviation *is* the
competitive-adsorption story, quantified from a handful of measured mixtures —
no need to measure every combination. (Recovered K here is an *apparent*
affinity; absolute K needs an internal-standard scale.)

## Desktop GUI app (`sers_app.py`)

A customtkinter dashboard (sidebar + cards + embedded matplotlib + export) that
wraps the analysis core — same layout idiom as a typical report-node app.

```bash
pip install customtkinter matplotlib scikit-learn scipy numpy
python sers_app.py
```

Load `example_pure.csv` (references) and `example_unknown.csv` (spectra to
analyse), set the detection threshold, hit **Run**. You get: the measured
spectrum with its NNLS reconstruction, a composition pie, a per-component
table (present / ratio / probability / residual), and CSV/PNG export. `brand.py`
holds the colors. To use your real XY-map CSVs, swap `load_csv` for your map
loader — the rest of the pipeline is unchanged.

## Concentration ratios under competitive adsorption (`competitive.py`)

The observed spectrum is `Y = g · Σ A_i θ_i P_i` with unknown gain `g`,
brightness `A_i`, competitive coverage `θ_i = K_i C_i / (1 + Σ K_j C_j)`.
Fitting `Y` to the pure templates by NNLS gives `B_i = g A_i θ_i`, and

    C_i : C_j  =  (B_i / R_i) : (B_j / R_j),   R_i = A_i K_i

so **gain and the competition term cancel in the ratio** — substrate
irreproducibility does not break it. You get `R_i` from **one calibration
mixture of known composition** (`R_i ∝ B_i^cal / C_i^cal`); no DFT needed if
you can make one standard. (If you can't, plug DFT/MLIP affinities for `K_i`.)

Two honest limits, both shown in `demo_quant.py`:

1. **Surface saturation** — if the dominant saturates the surface
   (`Σθ → 1`), its SERS signal plateaus and its concentration is only a
   lower bound. Fix: **dilute** until `Σθ < ~0.5`, then quantify.
2. **Noise floor** — a minor whose `K·C` is ~1000× below the dominant's
   falls under the noise; no algorithm recovers what isn't in the signal.
   Fixes: raise SNR (average M spectra ⇒ noise/√M, or a better substrate),
   use a clean marker band, or **standard addition / deplete the dominant**.

When dominance is mostly in *concentration* (comparable affinities), a 2–5%
buried minor is recoverable even from a single spectrum — see Part 2(b).

## Absolute concentration in µM (`demo_absolute.py`)

Ratios cancel the unknown substrate gain, so they need no anchor. Absolute
µM does — the raw SERS intensity is not reproducible batch to batch. The fix
is an **internal standard**: spike every sample with a fixed, known
concentration of a reference molecule and use the ratio

    r = B_analyte / B_internal_standard  =  (A_a K_a C_a) / (A_is K_is C_is)

The gain `g`, the competition term `(1+ΣKC)`, and even surface saturation all
cancel, so `r` is **linear in C_a**. One calibration series fixes the slope;
then `r → µM`, with an LOD from the blank's 3σ. Demo result: with the internal
standard, recovery is accurate and gain-robust (5→5.7, 20→22, 80→90 µM);
without it, a 3–12× gain swing makes the same numbers scatter by 3–5×.

Ceiling: if the analyte crowds the standard off the surface (its signal sinks
under noise), `r` becomes unmeasurable — the top of the usable range. Use a
Langmuir (not linear) fit for the upper decade.

## Install & run the demo

```bash
pip install numpy scipy scikit-learn
python demo.py
```

The demo trains on 6 pure spectra and evaluates on 120 synthetic mixtures
(sizes 1–3). Expect ~perfect precision, strong recall on 1–2 component
mixtures, and lower recall on 3-component mixtures where competitive
suppression genuinely buries the weakest component below noise — an honest
picture of the hard case, not a cooked number.

## Use your own measured spectra

Put your data in two CSVs sharing one wavenumber axis:

```
pure.csv       wavenumber, CompoundA, CompoundB, CompoundC, ...
mixtures.csv   wavenumber, mix1, mix2, ...
```

Then:

```bash
python run_on_my_data.py
```

If pure and mixture files use different axes, set `RESAMPLE=True` in that
file. To get accuracy numbers, fill in `TRUE_LABELS`.

## Knobs that matter

- `prob_threshold` (default 0.18) — lower = more sensitive (higher recall,
  more false positives). Raise it if you see spurious components.
- `nnls_rel_threshold` (default 0.06) — the stage-2 cutoff; lower keeps
  weaker components.
- `AugmentConfig` — match `noise_frac`, `shift_max`, `baseline_amp` to your
  instrument's real variability. **This is the single biggest lever**: the
  more your augmentation resembles real batch-to-batch SERS variation, the
  better the model transfers.

## Where to take it next

- Swap the RandomForest heads for a **1D-CNN** (PyTorch): a shared conv
  encoder + one sigmoid output per compound. Same interface, usually better
  on subtle peaks. The `predict_proba` contract is all you need to keep.
- Add an **internal-standard / ratiometric normalization** step in
  `preprocess` to fight SERS intensity irreproducibility.
- Model the competition explicitly: replace NNLS with a Langmuir-weighted
  fit if you can estimate relative surface affinities.
```
