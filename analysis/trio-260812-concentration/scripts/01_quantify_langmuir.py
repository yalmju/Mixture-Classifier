"""Per-pixel absolute concentration for the 260812 trio map — the app's own
calibration path (NNLS abundances -> competitive-Langmuir inversion against the
single-compound dilution series). Saves the UnmixResult for the later scripts.

Windows box paths (S: drive)."""
import os
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
from unmix import unmix_map

DATA = r"S:/Google Drive/내 드라이브/ACF_PEST_DB/Reference"
TEST = r"S:/Google Drive/내 드라이브/ACF_PEST_DB/Pest/260812_12 trio_THI_TBZ_DQ.csv"
CAL = r"S:/Google Drive/내 드라이브/ACF_PEST_DB/Pest/Standard/260729/calibration_spectra.csv"
OUT = Path(__file__).resolve().parents[1] / "out"
OUT.mkdir(exist_ok=True)

r = unmix_map(data_dir=DATA, test_path=TEST, method="nnls", baseline=False,
              trim=(500, 2500), min_frac=0.20, hit_mode="threshold",
              calib_path=CAL, progress=lambda m: None)
nb = [r.comps[i] for i in r.nonbg]
um = r.conc * 1e6
print("hit px:", int(r.hit.sum()), "/", r.n_pixels)
print("isotherm R2:", dict(zip(nb, np.round(r.calib_r2, 3))))
for k, nm in enumerate(nb):
    v = um[r.hit, k]; v = v[np.isfinite(v) & (v > 0)]
    print(f"{nm}: median {np.median(v):.3g} uM  P10-P90 "
          f"{np.percentile(v, 10):.3g}-{np.percentile(v, 90):.3g}")
pickle.dump(r, open(OUT / "res_cal.pkl", "wb"))
print("saved", OUT / "res_cal.pkl")
