"""Anchor-competition test on the Ratio_mix grid (nominal µM in the filename).

For every map: mean spectrum -> NNLS B against the pure templates -> competitive
Langmuir inversion with the 260729 standards -> apparent µM per substance.
Recovery = apparent / nominal. The question: does TBZ's recovery depend on WHO
the partner is (THI vs DQ) and how much of it there is?
"""
import os
import re
import sys
import glob
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

MIX = r"S:/Google Drive/내 드라이브/ACF_PEST_DB/Ratio/Ratio_mix"
CAL = r"S:/Google Drive/내 드라이브/ACF_PEST_DB/Pest/Standard/260729/calibration_spectra.csv"
OUT = Path(__file__).resolve().parents[1] / "out"

from real_data import load_map
from competitive import fit_B
from unmix import _baseline_removed
from io_utils import load_calibration_csv
from calibration import calibrate

r = pickle.load(open(OUT / "res_cal.pkl", "rb"))
nb = [r.comps[i] for i in r.nonbg]                 # DQ, TBZ, THI
pures = r.templates[r.nonbg]

axis_c, names_c, dils = load_calibration_csv(CAL)
cidx = {n: k for k, n in enumerate(names_c)}
mc = (axis_c >= 500) & (axis_c <= 2500)
aligned = [(np.asarray(dils[cidx[c]][0], float),
            _baseline_removed(np.asarray(dils[cidx[c]][1], float)[:, mc], False))
           for c in nb]
calib = calibrate(aligned, pures, nb)

pat = re.compile(r"DQ(\d+)-TB(\d+)-THI?(\d+)", re.I)
rows = []
for p in sorted(glob.glob(os.path.join(MIX, "*.csv"))):
    m = pat.search(os.path.basename(p))
    if not m:
        print("skip (name)", os.path.basename(p)); continue
    nom = {"DQ": float(m.group(1)), "TBZ": float(m.group(2)),
           "THI": float(m.group(3))}
    wn_b, cube, mean, _ = load_map(p)
    wn_b = np.asarray(wn_b, float)
    sel = (wn_b >= 500) & (wn_b <= 2500)
    y = _baseline_removed(mean[None, sel], False)[0]
    if len(y) != pures.shape[1]:
        print("skip (axis)", os.path.basename(p)); continue
    B, _ = fit_B(y, pures)
    th = np.clip(B / calib.gA, 0, None)
    tot = min(th.sum(), 0.999)
    C = th / (calib.K * (1.0 - tot)) * 1e6         # apparent µM
    rows.append((os.path.basename(p)[:-4], nom, C))

# ---- TBZ recovery by partner ----
print(f"{'map':24s} {'nomTBZ':>7s} {'appTBZ':>8s} {'recov':>7s}  partner")
tbz_rows = []
for name, nom, C in rows:
    if nom["TBZ"] <= 0:
        continue
    partners = [s for s in ("DQ", "THI") if nom[s] > 0]
    ptxt = "+".join(f"{s}{nom[s]:.0f}" for s in partners) or "none"
    rec = C[nb.index("TBZ")] / nom["TBZ"]
    tbz_rows.append((name, nom, rec, partners))
    print(f"{name:24s} {nom['TBZ']:7.0f} {C[nb.index('TBZ')]:8.2f} "
          f"{rec:7.3f}  {ptxt}")

import collections
groups = collections.defaultdict(list)
for name, nom, rec, partners in tbz_rows:
    key = "+".join(sorted(partners)) or "solo"
    groups[key].append(rec)
print("\nTBZ recovery by partner set (median [n]):")
for k, v in sorted(groups.items()):
    print(f"  {k:10s}: {np.median(v):.3f}  [n={len(v)}]")

# same summary for the other two substances
for target in ("DQ", "THI"):
    groups = collections.defaultdict(list)
    for name, nom, C in rows:
        if nom[target] <= 0:
            continue
        partners = "+".join(sorted(s for s in nb
                                   if s != target and nom[s] > 0)) or "solo"
        groups[partners].append(C[nb.index(target)] / nom[target])
    print(f"\n{target} recovery by partner set (median [n]):")
    for k, v in sorted(groups.items()):
        print(f"  {k:10s}: {np.median(v):.3f}  [n={len(v)}]")
