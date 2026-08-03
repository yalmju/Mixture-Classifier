# Composition figures & metrics (in the Validate tab)

These figures live in the **Validate** tab (below the response-factor plots) — the
same known-ratio mixtures drive both the correction (response factors) and this
composition view, so nothing is loaded or unmixed twice from the user's side.
They reuse the app's own NNLS unmixing
(`unmix.unmix_map`: L2-normalised template + non-negative least squares, with the
`BLK` blank held out as a background class) to read out, per reference/mixture
map, the **composition** (as a colour blend), the **predicted → real drift**, and
the **apparent (intensity) recovery**. No retraining or separate model.

Colours are the shared per-substance palette (top-bar picker): **DQ / TBZ / THI**.

Maps to analyse come from **Add maps…** (an explicit multi-file selection; the
nominal ratio is parsed from each file name, e.g. `DQ1TH3` → DQ:1, THI:3), or —
if none are loaded — are auto-discovered from `Reference/Ratio/*.csv` plus the
pure `DQ-1` / `TBZ-1` corners of the folder set in **Samples**.

## The four figures

| figure | what it shows |
|--------|---------------|
| **Composition maps** | each map's per-pixel composition as an RGB blend `f_DQ·c_DQ + f_TBZ·c_TBZ + f_THI·c_THI`; non-hit (weak-signal) pixels fade to white. Tile caption = mean composition. |
| **Drift triangle** | ternary with DQ (top), THI (bottom-left), TBZ (bottom-right); each edge carries quarter ticks (0·¼·½·¾·1). An arrow runs from the **real (nominal)** ratio on the edge to the **predicted (measured)** point (coloured by the real dominant substance). Corners show that substance's recovery. |
| **Relative drift** | per mixture, how far the **predicted** composition sits from the **real** one, scaled so the worst case = 100%, grouped by the real dominant substance (THI- / TBZ- / DQ-dominant). |
| **Recovery** | per-substance apparent recovery (below). |

## Metric definitions (the technical terms)

Let `p` = predicted (measured) composition and `r` = real (nominal) composition,
each a 3-vector of fractions summing to 1.

- **"predicted vs real" drift** (the ranked bars) is the **composition distance** —
  the Euclidean distance in fraction space, `‖p − r‖₂`, normalised to the worst
  mixture. An alternative scale-invariant metric is the **Aitchison distance**
  (centred-log-ratio): with `clr(x) = log x − mean(log x)`,
  `d_A(p, r) = ‖clr(p) − clr(r)‖₂`. Both are implemented in
  `composition.py` (`composition_distance`, `aitchison_distance`); the plot shows
  the composition distance for clarity and keeps the Aitchison term here.

- **apparent recovery** of substance *i* in a mixture = `p_i / r_i × 100`
  (only where `r_i > 0`). Reported per substance as **mean ± SE** over the true
  **mixtures** (≥2 nominal components); pure / dilution maps are excluded because
  their low-signal misattribution inflates the spread. The dots on the bar show
  the individual per-mixture values.

  `>100%` = over-reported on the surface (larger SERS cross-section, e.g. THI);
  `<100%` = under-reported (e.g. TBZ). This is an **intensity** recovery, not a
  concentration recovery — intensity composition ≠ mole ratio. For a true
  concentration recovery, supply a dilution-series calibration CSV and each map's
  spiked concentration; `unmix_map(..., calib_path=...)` then returns per-pixel
  absolute concentration via the Langmuir isotherm.
