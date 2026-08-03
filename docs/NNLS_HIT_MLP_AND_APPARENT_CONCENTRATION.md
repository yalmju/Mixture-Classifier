# NNLS-hit MLP and apparent SERS-equivalent concentration

## Purpose

UNMIXR separates two questions that must not be conflated:

1. **Where is a usable SERS signal present?**
2. **Among those hit pixels, which compounds are observed and at what apparent
   SERS-equivalent concentration?**

NNLS answers the first question. The MLP remains responsible for per-pixel
composition and concentration inference after screening.

## Dynamic hit workflow

The hit fraction is not fixed to 44%. For every map, UNMIXR independently fits the
pure substance and background references by NNLS. A pixel is a hit when its NNLS
substance share passes the configured threshold. Consequently:

- `260723_mixture(bg).csv`: 916/2100 hit pixels = 43.619%;
- known binary/ternary Recovery maps: 99.5–100% hit pixels.

The MLP does not redefine this spatial mask. This preserves the observed 고려대 ink
pattern in the Real panel while allowing different maps to have different hit fractions.

The shared order of operations is:

```text
map -> preprocessing -> NNLS hit/background screening
    -> MLP on hit pixels -> apparent composition and component-wise concentration
```

## MLP inputs

The production MLP remains in use. The Model panel now enables `NNLS-screen ink first`
by default and samples 20 hit pixels per map in addition to the hit-region aggregate.
Splits remain grouped by map, so pixels from one map cannot leak across train and test.

Experiments showed that absolute intensity carries useful information:

| Representation | Independent-test mean composition error | 1:1:1 error |
|---|---:|---:|
| L2 spectrum only | 11.2% | 31.8% |
| log-raw spectrum | 8.7% | 28.8% |
| log-raw + L2 shape + intensity scale | 8.2% | 27.3% |

The hybrid representation improved performance but did not eliminate the reproducible
THI-dominant/TBZ-suppressed response.

## Surface response versus input concentration

Equal-volume mixing does not imply equal SERS response. A nominal 1:1:1 mixture
reproducibly produced approximately THI:DQ:TBZ = 59–62:30–33:7–8 in the learned
apparent composition. This is compatible with compound-dependent adsorption, ink
affinity and SERS enhancement.

The experimental inverse-competition benchmark learns:

```text
apparent response + log-raw spectrum + intensity scale + hit fraction
    -> input composition / component-wise concentration
```

For the independent 1:1:1 map, the ridge inverse head changed the apparent suppressed
TBZ response toward an input estimate of DQ:TBZ:THI = 25.0:27.9:47.1. Its direct
component-wise concentration estimate was DQ 35.4, TBZ 17.3 and THI 182.1 µM.

These values are not constrained to sum to the applied total concentration. They are
interpreted as **apparent SERS-equivalent concentrations at the observed region**, not
as a mass balance or proof of the prepared solution concentration.

The inverse-competition benchmark is currently experimental and is not silently
substituted for a loaded production `.dlm`/`.psm` in the Real panel.

## Equal-volume concentration labels

Mixture metadata stores source-solution concentrations. Samples were prepared with
equal component volumes, so concentration-head training and Recovery truth apply:

```text
final component concentration = source concentration / number of mixed components
```

Thus a binary mixture divides each source concentration by 2 and a ternary mixture by
3. Composition ratios are unchanged. The original `mixtures.json` values remain intact;
the dilution is applied at training/evaluation time.

## Real-panel concentration display

Each hit pixel has a component-wise concentration estimate. The concentration panel:

- plots one heatmap per compound;
- uses one shared µM colour axis across DQ, TBZ and THI;
- hides background, invalid and OOD pixels;
- does not normalise component concentrations to a fixed total.

The summary reports three distinct uncertainty descriptions:

- **median ± spatial-bootstrap SE**: representative hit-pixel value and standard error
  of the median;
- **P10–P90**: the central 80% of the observed hit-pixel distribution;
- **95% CI**: scan-line cluster-bootstrap interval for the median.

Scan-line resampling preserves within-line spatial correlation better than treating all
neighbouring pixels as independent replicates.

## Reproducible benchmarks

- `archive/experiments/benchmarks/compare_spectral_representations.py`
- `archive/experiments/benchmarks/inverse_competition_benchmark.py`
- `archive/experiments/benchmarks/validate_pixel_concentration_v2.py`

Generated JSON/model outputs are written under `output/` and are intentionally ignored
by Git.

## Interpretation limits

- Apparent concentration is semi-quantitative and component-specific.
- Component values need not sum to the prepared total concentration.
- A narrow pixel percentile range is not the same as model generalisation accuracy.
- Bootstrap intervals quantify sampling/model variation under the available maps; they
  do not remove batch, substrate or calibration bias.
- Absolute µM claims require independent mixture maps spanning the intended substrate,
  batch and concentration range.
