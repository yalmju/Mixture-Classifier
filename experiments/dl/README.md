# DL quantification — experiment scripts

Reproduction scripts for the physics-informed DL mixture-quantification study
(see `../../docs/dl_quantification_findings.md`). The reusable model code lives in
`../../dl_quantify.py`; these are the one-off validation / benchmark drivers.

**Data paths.** Each script locates the real data by scanning `S:/Google Drive` for the
`ACF_PEST_DB` folder (Pure references, `Ratio/Binary`, `Ratio/Tertiary`,
`Ratio/results/calibration_spectra.csv`). Adjust the `root`/`acf` lines if the dataset
moves. Run from the repo root: `python experiments/dl/<script>.py`.

| script | what it does |
|---|---|
| `dl_validate_real.py` | Track A on real binary mixtures — DL/NNLS composition vs true ratio |
| `dl_validate_B.py` | Track B residual (mean-spectrum), leave-one-map-out, vs NNLS & linear RF |
| `dl_validate_B_pixel.py` | Track B per-pixel, leave-one-MAP-out (shows per-pixel adds no info) |
| `dl_binary_to_ternary.py` | Train on 28 binary → predict 7 unseen ternary (DL beats linear RF) |
| `binary_hard.py` | Buried weaker-adsorber recovery in binary, by competition pair |
| `buried_recovery.py` | Buried non-THI recovery on ternary → `buried_recovery.png` |
| `combined_loo.py` | Full 35-map LOO (binary+ternary), full-spectrum DL wins (20.5%) |
| `baselines_compare.py` | NNLS / linear RF / PLS / RandomForest / DL comparison |
| `dl_spectrum.py` | Full-spectrum DL vs PLS on ternary |
| `physics_embedded.py` | Differentiable competitive-Langmuir fit to mixtures vs PLS |
| `bench_ensemble.py` | Red-point fixes: single DL vs bagged ensemble vs PLS+DL hybrid |
| `export_predictions.py` | Writes `docs/mixture_predictions.csv` |
| `triangle_v2.py` | Ternary classification triangle → `docs/classify_triangle.png` |
| `make_pdf.py` | Builds `docs/update_2026-07-29.pdf` |

Concentration convention for the real mixtures: filename code × 10 µM
(1→10 µM, 10→100 µM, 100→1 mM, 0.1→1 µM, 0.01→0.1 µM).
