# NNLS-hit MLP and apparent concentration

UNMIXR separates detection from estimation:

```text
map -> Samples preprocessing -> NNLS hit/background gate
    -> loaded MLP on hit pixels -> composition + component-wise apparent concentration
```

NNLS recalculates the hit mask for every map. The MLP cannot replace that mask, so sparse SERS-ink patterns remain intact. The threshold is configurable and saved in the model.

The MLP learns from known binary/ternary mixtures in Samples. Composition and concentration use the saved analyte order. Component concentrations are not forced to sum to a prepared total.

When Samples values are source-solution concentrations and equal volumes were mixed, enable `equal-volume mixtures`; training truth is divided by the number of non-zero component solutions. Disable it when Samples already contains final mixture concentrations. This interpretation is saved in the model.

Real reporting uses component µM maps plus median ± spatial-bootstrap SE, P10–P90 and a bootstrap interval. These are semi-quantitative apparent SERS-equivalent values for the observed hit region, not mass balance or cross-condition proof.

For the current dataset, `260723_mixture(bg).csv` produced 916/2100 hit pixels (43.619%), while dense Recovery maps were near 100% hit. These are measured outcomes, never fixed code values.