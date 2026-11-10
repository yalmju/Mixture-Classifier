"""Run physics_max_rescue_experiment with calibration/reference axes aligned."""
from __future__ import annotations

import numpy as np

import physics_max_rescue_experiment as exp


def physics_pretrain_fixed(pure, calib_csv, P, mask):
    ax, names, dils = exp.load_calibration_csv(calib_csv)
    dil = []
    for substance in exp.SUB:
        concentrations, spectra = dils[names.index(substance)]
        spectra = np.asarray(spectra)
        if spectra.shape[1] != P.shape[1]:
            raise ValueError(
                f"Calibration/reference axis mismatch: {spectra.shape[1]} vs {P.shape[1]} channels"
            )
        dil.append((concentrations, spectra))
    cal = exp.calibrate(dil, P, exp.SUB)
    m = np.ones(3)
    m[0] = exp.fit_sips(cal.C_series[0], cal.B_series[0])[2]
    rng = np.random.default_rng(0)
    Xp, Cp = exp.simulate_mixtures(
        P, cal.K, cal.gA, 5000, rng, noise=.015, baseline=.03,
        gain_lo=.8, gain_hi=1.25, iso_m=m, nuisance=None,
    )
    return (
        exp._composition_features(Xp, "log1p_raw").astype(np.float32),
        np.asarray([exp._ratio(c) for c in Cp], np.float32),
        cal,
        m,
    )


exp.physics_pretrain = physics_pretrain_fixed

if __name__ == "__main__":
    exp.main()
