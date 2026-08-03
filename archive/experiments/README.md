# Experiments and report tools

UNMIXR's production code lives at the repository root. Nothing under this
directory is imported by `UNMIXR.py`; these scripts reproduce benchmarks,
research figures, and PDF summaries.

Run scripts from the repository root so imports resolve consistently.

## Layout

- `benchmarks/` — maintained model, detection, adsorption, and robustness comparisons.
- `concentration/` — absolute-concentration studies and dataset/session comparisons.
- `reports/` — PDF summary generators retained because project summaries are produced regularly.
- `archive/legacy_dl/` — superseded validation and plotting scripts retained for provenance.

Examples:

```powershell
python experiments/benchmarks/model_benchmark.py
python experiments/concentration/concentration_ratiomix.py
python experiments/reports/make_pdf_0730.py
```

Reusable implementations belong in the root modules (`dl_model.py`,
`dl_quantify.py`, `dl_explain.py`, and `triangle_figs.py`), not in experiment
drivers. New studies should reuse those modules rather than copy their training,
normalization, or plotting logic.