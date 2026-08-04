# NNLS-hit MLP and apparent concentration

UNMIXR separates detection from estimation:

```text
map -> Samples preprocessing -> NNLS hit/background gate
    -> MLP on each hit pixel -> composition and apparent concentration
    -> map summary: median, spatial-bootstrap SE, P10-P90, confidence interval
```

## Fixed workflow

- NNLS alone determines hit/background for every map. The MLP cannot alter this mask.
- Only NNLS-hit pixels enter the composition and concentration heads.
- Training and Real inference use the same spectral window and preprocessing saved by Samples.
- Row-wise L2 normalization is not used by the learned composition head, so absolute signal information is retained.
- Pixels from one map always remain in the same validation fold because they share one mixture label.
- Component concentrations are estimated separately and are never forced to sum to a prepared total.

## Preventing overfit

The epoch field is a maximum, not a forced training length. About 20% of complete maps are held out to select the best epoch. The model is then reinitialized and refit on all maps for that selected number of epochs. The selected composition and concentration epochs are stored in the model and shown in the Model-tab status.

The concentration head uses map-level weak supervision. Each hit pixel is encoded separately, but the loss compares the median prediction over a map with that map's known applied concentration. It does not pretend that every pixel has an independently measured concentration label.

## Concentration interpretation

When Samples contains source-solution concentrations and equal volumes were mixed, enable equal-volume mixtures. The target for each present component is divided by the number of non-zero component solutions. Disable it when Samples already contains final mixture concentrations. This interpretation is stored in the model.

Reported values are semi-quantitative apparent SERS-equivalent concentrations for the observed hit region. They are not mass balance and are not automatically transferable across substrates, acquisition settings, or experimental batches.

## Current-data verification (2026-08-04)

For `260723_mixture(bg).csv`, NNLS selected 916 of 2100 pixels (43.619%). This value is measured from the map and is not hard-coded.

With 34 mixture maps and a maximum of 350 epochs, map-group validation selected:

- composition: 68 epochs;
- concentration: 46 epochs.

The fixed 350-epoch composition model had shifted the Real prediction toward TBZ, demonstrating overfit. Early stopping reduced that drift, but the concentration validation loss remained poor. Independent leave-one-map-out checks on the current low-dimensional concentration observations produced approximately five-fold mean error with Random Forest and eight-fold error with Ridge; DQ and TBZ were substantially less identifiable than THI.

Therefore, early stopping is necessary but does not by itself validate exact Real-data micromolar values. Concentration claims require map-level or independent-batch log-error/recovery reporting, and additional experimental batches should be evaluated before publication.