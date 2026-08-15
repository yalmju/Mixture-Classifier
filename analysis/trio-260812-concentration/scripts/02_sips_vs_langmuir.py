"""Sips vs Langmuir inversion at this map's signal level.

Sips (m<1) makes the low-coverage inversion WORSE: the pixels sit in the
isotherm's low-C tail where the 1/m exponent amplifies noise downward."""
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
from io_utils import load_calibration_csv
from calibration import calibrate, fit_sips
from competitive import fit_B
from unmix import _baseline_removed

CAL = r"S:/Google Drive/내 드라이브/ACF_PEST_DB/Pest/Standard/260729/calibration_spectra.csv"
OUT = Path(__file__).resolve().parents[1] / "out"

r = pickle.load(open(OUT / "res_cal.pkl", "rb"))
nb = [r.comps[i] for i in r.nonbg]
pures = r.templates[r.nonbg]
hit = r.hit

axis_c, names_c, dils = load_calibration_csv(CAL)
cidx = {n: k for k, n in enumerate(names_c)}
mc = (axis_c >= 500) & (axis_c <= 2500)
aligned = [(np.asarray(dils[cidx[c]][0], float),
            _baseline_removed(np.asarray(dils[cidx[c]][1], float)[:, mc], False))
           for c in nb]
calib = calibrate(aligned, pures, nb)
gA_s = np.zeros(len(nb)); K_s = np.zeros(len(nb)); m_s = np.zeros(len(nb))
for k in range(len(nb)):
    gA_s[k], K_s[k], m_s[k] = fit_sips(calib.C_series[k], calib.B_series[k])
print("Sips m:", dict(zip(nb, np.round(m_s, 3))))

spectra = np.asarray(r.spectra, float)
idx = np.where(hit)[0]
B = np.stack([fit_B(spectra[i], pures)[0] for i in idx])

def med(C):
    return {nm: float(np.median(C[np.isfinite(C[:, k]) & (C[:, k] > 0), k]))
            for k, nm in enumerate(nb)}

th = np.clip(B / calib.gA[None, :], 0, None)
tot = np.minimum(th.sum(1), 0.999)
C_L = th / (calib.K[None, :] * (1 - tot)[:, None]) * 1e6
th = np.clip(B / gA_s[None, :], 0, None)
tot = np.minimum(th.sum(1), 0.999)
with np.errstate(invalid="ignore"):
    C_S = np.power(th / (1 - tot)[:, None], 1.0 / m_s[None, :]) / K_s[None, :] * 1e6
print("median uM  Langmuir:", {k: round(v, 3) for k, v in med(C_L).items()})
print("median uM  Sips    :", {k: round(v, 3) for k, v in med(C_S).items()})
