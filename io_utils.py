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
import glob
import os
import re

import numpy as np


# concentration parsed from a filename like 1nM / 10uM / 100nM / 1mM / 500uM / 1M
_CONC_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([mµunp]?)M(?![a-zA-Z])")
_CONC_UNIT = {"": 1.0, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9, "p": 1e-12}


def parse_concentration(name):
    """Molar value from a filename/label ('1mM'→1e-3, '100nM'→1e-7, '1uM'→1e-6),
    or None if no concentration token is found."""
    m = _CONC_RE.search(name)
    if not m:
        return None
    return float(m.group(1)) * _CONC_UNIT.get(m.group(2), 1.0)


def _compound_from_folder(folder):
    """Compound name from a dilution-series folder name: 'conc_tbz' -> 'TBZ'."""
    b = os.path.basename(os.path.normpath(folder))
    b = re.sub(r"^conc[_\- ]*", "", b, flags=re.IGNORECASE).strip("_- ")
    return b.upper() if b else "compound"


def load_calibration_folder(folder):
    """Build ONE compound's dilution series from a folder of per-concentration map
    CSVs (each file = one standard; the concentration is read from the filename).
    Returns (axis, compound_name, concentrations (k,), spectra (k, n_feat)) with the
    per-map MEAN spectrum as each standard. Raises if fewer than 2 usable files."""
    from real_data import load_map                       # local: avoid import cycle
    files = (sorted(glob.glob(os.path.join(folder, "*_corrected.csv")))
             or sorted(glob.glob(os.path.join(folder, "*.csv"))))
    concs, specs, axis, skipped = [], [], None, []
    for p in files:
        base = os.path.splitext(os.path.basename(p))[0]
        c = parse_concentration(base)
        if c is None:
            skipped.append(base); continue
        wn, _cube, mean, _coord = load_map(p)
        if axis is not None and len(wn) != len(axis):
            raise ValueError(f"{base}: {len(wn)} points vs {len(axis)} — the "
                             "dilution maps must share one wavenumber axis.")
        axis = wn; concs.append(c); specs.append(mean)
    if len(concs) < 2:
        raise ValueError(
            f"need ≥2 concentration files with a parseable name (1nM, 10uM, 1mM…) "
            f"in {os.path.basename(folder)}; found {len(concs)}"
            + (f", skipped {skipped}" if skipped else ""))
    order = np.argsort(concs)
    return (axis, _compound_from_folder(folder),
            np.array(concs)[order], np.array(specs)[order])


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
