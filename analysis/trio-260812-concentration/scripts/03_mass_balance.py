"""2 µL mass-conservation accounting.

The Langmuir curve reads a pixel as "solution concentration at the STANDARD
deposition geometry", so with equal 2 µL volumes the standard's wetted area
required to make the truth come out is  A_std = a_px * sum(C_app) / 12 µM.
If geometry explained everything, all three substances would demand the SAME
A_std. They don't: TBZ demands x4-5 less — a per-substance surface deficit."""
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "out"
TRUTH = 12.0
A_PX = 0.01                                   # mm^2 (100 µm pixel)

r = pickle.load(open(OUT / "res_cal.pkl", "rb"))
nb = [r.comps[i] for i in r.nonbg]
um = r.conc * 1e6
hit = r.hit
print(f"ink footprint {hit.sum()} px = {hit.sum() * A_PX:.2f} mm^2")
req = {}
for k, nm in enumerate(nb):
    v = um[hit, k]
    s = float(np.nansum(np.where(np.isfinite(v) & (v > 0), v, 0.0)))
    req[nm] = A_PX * s / TRUTH
    print(f"{nm}: sum C_app {s:.1f} uM*px -> A_std needed {req[nm]:.2f} mm^2 "
          f"(droplet dia {2 * np.sqrt(req[nm] / np.pi):.2f} mm)")
v = list(req.values())
print(f"spread {max(v) / min(v):.1f}x  (1.0x = geometry explains all)")
