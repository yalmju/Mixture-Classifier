# Reusable configuration

UNMIXR derives experimental identity from the selected Samples dataset and saved model, not pesticide-specific constants.

## Runtime contract

- Reference classes come from `samples.csv`/reference filenames; background aliases are excluded from analyte outputs.
- Composition MLP features use `log1p(raw intensity)`; row-wise L2 normalization is not used, so between-pixel intensity information is retained.
- With `pixels/map 400`, all 400 spatial spectra are training inputs; evaluation remains grouped by map.
- Class order is stored in `model["subs"]` and reused by Real, Recovery, plots and CSV exports.
- Spectral window and baseline come from Samples preprocessing. With no window, the full measured axis is used.
- NNLS hit screening is recalculated per map. `screen_min_frac` is user-configurable and stored; 44% is an observation, not a constant.
- Samples concentrations can be final-mixture values or equal-volume source solutions. The explicit `equal_volume_mix` choice and `concentration_basis` are stored.
- Ternary plots appear only for exactly three analytes. Other class counts retain parity, error, ROC, maps and tabular exports.

DQ, TBZ and THI are the current dataset, not required class names. A 1:1:1 ratio, 1 mM total, 500–1800 cm⁻¹ window, or fixed hit percentage is not inferred unless supplied in settings/metadata.

## Reuse checklist

1. Add pure references and assign class/batch/background roles in Samples.
2. Set baseline and spectral range for that protocol.
3. Add known mixtures and ratios/concentrations.
4. Choose final concentrations or equal-volume source solutions.
5. Set the NNLS hit threshold and train/save the model.
6. Apply it only when incoming spectra match the stored axis/preprocessing; otherwise retrain or report OOD.

This is configuration portability. It does not itself prove scientific transfer to a new substrate, day, instrument or ink batch.