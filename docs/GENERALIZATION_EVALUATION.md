# Generalization evaluation

## Verdict

The current NNLS-hit MLP is **not yet demonstrated to be a general model for arbitrary
experimental conditions**.

All 34 labeled mixture maps currently come from the same `Ratio_mix` source. The saved
train/test roles provide six independent maps, but no labeled mixture batch from a
different measurement day, substrate lot, ink lot or instrument condition was available.
Consequently, genuine cross-condition generalization cannot be claimed from this dataset.

## Controlled stress test

The equal-volume-corrected MLP was trained on 28 maps and evaluated on the six Role=test
maps. NNLS hit screening was recalculated after each controlled spectral perturbation.

| Condition | Mean composition error | Mean present-component log10 µM MAE | Interpretation |
|---|---:|---:|---|
| Clean | 7.6% | 0.69 | baseline reference |
| Gain 0.5× | 7.6% | 0.75 | composition gain-invariant; concentration shifts |
| Gain 2× | 7.6% | 0.73 | composition gain-invariant; concentration shifts |
| Baseline drift | 60.7% | 3.57 | failure |
| 2% noise | 13.4% | 0.52 | degraded but usable for composition |
| +3-channel shift | 20.0% | 0.55 | material registration sensitivity |

The six Recovery/test maps were 100% hit under every condition, so this experiment does
not establish hit-gate robustness for sparse-ink maps. The clean 260723 Real map separately
retained its naturally calculated 916/2100 = 43.619% hit mask; that fraction is not fixed.

## Interpretation

- L2-normalized composition is robust to uniform gain changes.
- Component-wise apparent concentration is not gain invariant and remains
  semi-quantitative.
- Baseline mismatch is the largest observed transfer failure.
- Wavenumber registration changes also materially affect composition.
- Bootstrap SE/CI describes uncertainty under the available map population; it does not
  cover an unseen substrate or day bias.

## Required evidence before claiming a general model

Collect a labeled external batch that changes at least:

1. measurement day;
2. SERS substrate/ink lot;
3. operator or instrument alignment;
4. balanced binary/ternary concentrations, including 1:1:1;
5. sparse-ink maps with non-trivial hit fractions.

Keep that entire batch out of model selection. Report composition error, component-wise
median ± spatial-bootstrap SE, P10–P90, OOD rate, hit-mask agreement and concentration
log error. Until that evaluation passes, the UI should describe concentration as
`apparent SERS-equivalent`, `semi-quantitative`, and condition-specific.

## Reproduction

The benchmark is archived at:

`archive/experiments/benchmarks/generalization_stress.py`

Generated JSON is written to `output/generalization_stress.json` and is ignored by Git.
