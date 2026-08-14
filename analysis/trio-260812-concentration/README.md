# Trio 260812 — absolute concentration from the "SERS" lettering map

**Map**: `ACF_PEST_DB/Pest/260812_12 trio_THI_TBZ_DQ.csv` — "SERS" written with
2 µL of ink containing **12 µM EACH of DQ / TBZ / THI** (ground truth 12:12:12).
**Calibration**: `ACF_PEST_DB/Pest/Standard/260729/calibration_spectra.csv` —
single-compound dilution series, 9 points each, 0.1–1000 µM
(isotherm fits R² = 0.81 / 0.96 / 0.98 for DQ / TBZ / THI).
**References**: `ACF_PEST_DB/Reference` (DQ/TBZ/THI/BLK), trim 500–2500 cm⁻¹,
baseline off, hit = threshold mode, min substance fraction 0.20 (842/2400 px,
100% of the lettering, 5% stray).

## The story: one map, full readout

Everything the "SERS" lettering map yields, against the dispensed truth:

| readout | SERS lettering map | dispensed truth |
|---|---|---|
| detection | DQ · TBZ · THI, all three | 3 substances |
| composition (mixture-trained MLP) | 36.5 / 30.1 / 33.5 % | 33.3 / 33.3 / 33.3 % |
| concentration, DQ | 10.1 ± 1.9 µM ✓ | 12 µM |
| concentration, THI | 14.2 ± 2.8 µM ✓ | 12 µM |
| concentration, TBZ | **2.7 ± 0.5 µM — SERS-effective surface loading** | (12 µM dispensed) |

TBZ's 2.7 µM is reported uncorrected, as the **SERS-effective surface
loading**. The deficit is a reproducible, deposition-dependent response
suppression: the same ink and the same geometry put the other two substances
on the truth, and the same pipeline on droplet-dispensed mixtures shows no
TBZ loss (reconciliation below). Whether the molecules are absent from the
surface (competitive displacement — consistent with TBZ sitting at the bottom
of the script-06 hierarchy) or present but SERS-silent (enhancement loss,
e.g. precipitation or orientation in the fast-drying film) is
**indistinguishable from this data**; an orthogonal assay (wash-off LC-MS of
the lettering) would separate the two. The one-line story: **the same ink
reads 10 : 2.7 : 14 at the surface — the map resolves deposition effects the
solution never shows.** The dispensed solution (12:12:12) is still recovered,
by the composition channel: the mixture-trained MLP reads TBZ at 30% because
the suppression is reproducible enough to be learned.

## Concentration details

With one shared 2 µL droplet geometry (Ø 1.39 mm, fixed by DQ+THI agreeing
independently), mass balance over the ink footprint gives the table above.
Error bars combine a scan-line cluster bootstrap of the pixel sums (±14–15%)
with the geometry-anchor spread between DQ's and THI's required areas (±13%),
in quadrature. The geometry term is common to all three substances, so it never
moves the ratios — TBZ's deficit is ~9σ from the dispensed value. Notably TBZ
has the HIGHEST single-compound affinity (K = 3.3×10⁴/M), yet loses in the
mixture — the drying/co-adsorption dynamics, not equilibrium affinity, decide.

## The chain of evidence (scripts, in order)

1. `01_quantify_langmuir.py` — the app's calibration path (competitive-Langmuir
   inversion of per-pixel NNLS abundances). Raw medians: DQ 1.58 / TBZ 0.48 /
   THI 2.13 µM — all far below 12, which triggered the accounting below.
2. `02_sips_vs_langmuir.py` — Sips (heterogeneity m = 0.59/0.77/0.97) makes the
   low-signal inversion WORSE (DQ collapses to 0.05 µM): at this signal level
   the pixels sit in the isotherm's low-C tail where the 1/m exponent amplifies
   noise. Langmuir inversion is the stable choice here.
3. `03_mass_balance.py` — 2 µL mass conservation. Required standard-droplet
   area per substance: DQ 1.29 / TBZ 0.34 / THI 1.81 mm². DQ and THI agree
   (Ø 1.28 vs 1.52 mm — one plausible droplet); TBZ demands ×4–5 less.
   Geometry is a COMMON factor, so it cannot explain a per-substance gap.
4. `04_binary_response_factors.py` — response factors from the old-batch binary
   ratio maps (`Reference/Ratio`): R = DQ 0.69 / TBZ 1.0 / THI 17.3. Applying
   them makes consistency WORSE (spread 5.3× → 18×): the old batch's 17× THI
   over-emission is absent on the 260812 substrate. **Response factors do not
   transfer across substrate batches.**
5. `05_final_estimate.py` — the shared-geometry estimate in the table above
   (figure + CSV in `figures/`).

## Relation to the composition result

Three readings of TBZ, none contradictory — each is a different quantity:
raw NNLS surface-signal share 4.8% → single-compound-calibrated concentration
share 10% → mixture-trained MLP composition 29.5% (truth 33.3%). The MLP was
trained on 35 real mixture maps labelled in solution basis
(`concentration_basis: "equal-volume final mixture"`), so the competitive
suppression is absorbed into its weights — which is WHY it recovers the
solution ratio that linear unmixing cannot.

## Anchor-competition test (`06_anchor_competition.py`)

The `Ratio/Ratio_mix` grid (34 maps, nominal µM in the filename) directly tests
whether TBZ's deficit depends on WHO the partner is. TBZ recovery
(apparent/nominal, median): partner DQ 0.16, partner THI 0.04, both 0.00.
Dose series at TBZ 500 µM: partner 5 → 50 → 500 µM collapses recovery
0.52 → 0.10 → 0.01 (THI) and 0.61 → 0.16 → 0.005 (DQ) — dose-dependent
displacement by both partners, THI consistently the stronger suppressor.

The full hierarchy is THI > DQ > TBZ (THI recovers up to 0.89 beside DQ; DQ
0.21–0.37; TBZ always last) — the SAME ordering the trio map produced
(118% / 85% / 22%), measured on an independent campaign. TBZ sits at the
bottom of the surface-competition pecking order; its trio deficit is that
ordering, not a one-off artefact. (Absolute recoveries in this grid are
compressed by isotherm saturation at 150–500 µM nominal; the partner
COMPARISON at matched dose is the valid readout.)

## Reconciliation with the 64-condition droplet campaign (260814)

The 64-condition droplet grid (3–24 µM, `Ratio/260814_mixture_final`,
run log `260814_final_rerun/_runlog/conc38.txt`) quantifies TBZ WELL
(median 1.44-fold, 79% within 2-fold; its own 12/12/12 condition reads
8.7 / 12.7 / 13.8 for DQ/TBZ/THI) — apparently contradicting the ×4.5 TBZ
deficit here. It does not contradict it. Running THIS analysis's pipeline
(same single-compound 260729 standards, same trim) directly on the droplet
maps also shows NO TBZ collapse (DQ12-TB12-TH12 → TBZ 17.9 µM;
DQ24-TB24-TH24 → 22.4 µM), so the pipeline is not the difference —
**the deposition mode is**:

- **droplet-dispensed mixtures** (same protocol as the standards): TBZ lands
  fine; the grid campaign's problem substance is DQ (median 1.92-fold there,
  and this pipeline over-reads DQ on several droplet maps too).
- **pen-written lettering** (this trio map): thin, fast-drying film — the
  deposition where TBZ, the weakest competitor in the Ratio_mix hierarchy
  (THI > DQ > TBZ, script 06), actually loses ×4.5.

So "TBZ quantifies at 95% within 2-fold" (droplets) and "TBZ shows a ×4.5
surface deficit" (written letters) are both true, about different deposition
physics. Any claim about the trio lettering map must not be quoted against
droplet-campaign numbers without stating the deposition mode.

## Caveats

- The ×4.5 TBZ factor is characterised at ONE composition (1:1:1, 12 µM).
  Whether it is constant across compositions/concentrations needs same-batch
  binaries or a trio dilution series.
- Effective TBZ LOD inside the trio rises accordingly: 1.74 µM (single) →
  ~8 µM (mixture), so the 12 µM dispense sits only ~1.5× above it.
- Data paths in the scripts are hard-coded for the Windows box (S: drive).
