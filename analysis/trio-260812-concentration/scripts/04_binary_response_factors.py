"""Response factors from the OLD-batch binary ratio maps (Reference/Ratio) and
why they must NOT be applied here: the old batch's 17x THI over-emission is
absent on the 260812 substrate, so applying R widens the mass-balance
inconsistency from 5.3x to 18x. Response factors are batch-specific."""
import os
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
from real_data import load_map, MIXES, _response_factors
from competitive import fit_B
from unmix import _baseline_removed

RATIO = r"S:/Google Drive/내 드라이브/ACF_PEST_DB/Reference/Ratio"
OUT = Path(__file__).resolve().parents[1] / "out"

r = pickle.load(open(OUT / "res_cal.pkl", "rb"))
nb = [r.comps[i] for i in r.nonbg]
pures = r.templates[r.nonbg]
order = [0, 2, 1] if nb == ["DQ", "TBZ", "THI"] else None   # MIXES is [DQ,THI,TBZ]
assert order is not None, nb

B_mix, names = {}, []
for k in MIXES:
    p = os.path.join(RATIO, k + "_corrected.csv")
    if not os.path.exists(p):
        continue
    wn_b, cube, mean, _ = load_map(p)
    m = (np.asarray(wn_b, float) >= 500) & (np.asarray(wn_b, float) <= 2500)
    y = _baseline_removed(mean[None, m], False)[0]
    if len(y) != pures.shape[1]:
        continue
    B_mix[k] = fit_B(y, pures)[0]
    names.append(k)
R = _response_factors(names, B_mix, {k: [MIXES[k][j] for j in order]
                                     for k in names}, nb)
print("binaries:", names)
print("R (old batch):", {n: round(float(v), 3) for n, v in zip(nb, R)})
print("verdict: do NOT apply to 260812 — see README (consistency 5.3x -> 18x).")
