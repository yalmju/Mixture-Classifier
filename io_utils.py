"""io_utils.py — CSV load / export helpers for UNMIXR (numpy + stdlib only).

Formats
-------
Spectra CSV (one column per spectrum, shared wavenumber axis):
    wavenumber, name1, name2, ...
    400.0,      0.03,  0.04,  ...

Calibration CSV (a dilution series; one row per measured standard):
    compound, concentration_M, <wn1>, <wn2>, ...
    DQ,       1e-6,            0.02,  0.05,  ...
"""
from __future__ import annotations

import csv
import numpy as np


def load_spectra_csv(path):
    """Return (axis (n_feat,), names (list), spectra (n_names, n_feat))."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    names = [h for h in header[1:] if h != ""]
    axis, cols = [], []
    for r in rows[1:]:
        vals = [v for v in r if str(v).strip() != ""]
        if len(vals) < 2:
            continue
        axis.append(float(vals[0]))
        cols.append([float(v) for v in vals[1:len(names) + 1]])
    return np.array(axis), names, np.array(cols).T


def load_calibration_csv(path):
    """Return (axis, {compound: (concentrations (k,), spectra (k, n_feat))})."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    axis = np.array([float(x) for x in header[2:] if x.strip() != ""])
    n = len(axis)
    acc = {}
    for r in rows[1:]:
        if len(r) < 3:
            continue
        comp = r[0].strip()
        conc = float(r[1])
        spec = [float(v) for v in r[2:2 + n]]
        acc.setdefault(comp, ([], []))
        acc[comp][0].append(conc)
        acc[comp][1].append(spec)
    names = list(acc.keys())
    dilutions = [(np.array(acc[c][0]), np.array(acc[c][1])) for c in names]
    return axis, names, dilutions


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
