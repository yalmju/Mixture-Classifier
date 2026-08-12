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


# concentration parsed from a filename like 1nM / 10uM / 100nM / 1mM / 1_mM / 500uM
_CONC_RE = re.compile(r"(\d+(?:\.\d+)?)[\s_\-]*([mµunp]?)M(?![a-zA-Z])")
_CONC_UNIT = {"": 1.0, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9, "p": 1e-12}
_NON_MAP = {"samples.csv", "mixtures.csv", "preprocess.json", "colors.json"}


def parse_concentration(name):
    """Molar value from a filename/label ('1mM'/'1_mM'→1e-3, '100nM'→1e-7,
    '1uM'→1e-6), or None if no concentration token is found."""
    m = _CONC_RE.search(name)
    if not m:
        return None
    return float(m.group(1)) * _CONC_UNIT.get(m.group(2), 1.0)


def read_spectra_table(path):
    """Read a wavenumber-in-rows spectra table (column 0 = wavenumber, every other
    column = one spectrum), tab / comma / whitespace separated — the common Raman
    map/point export. Returns (wavenumbers (m,), spectra (n_spectra, m))."""
    with open(path, encoding="utf-8-sig") as f:
        head = f.readline()
    delim = "\t" if "\t" in head else ("," if "," in head else None)
    try:
        data = np.loadtxt(path, delimiter=delim)
    except ValueError:                                      # a header line → skip it
        data = np.loadtxt(path, delimiter=delim, skiprows=1)
    data = np.atleast_2d(data)
    if data.shape[1] < 2:                                   # single column → 1 spectrum
        return np.arange(len(data)), data.reshape(1, -1)
    return data[:, 0], data[:, 1:].T


def _is_map_csv(path):
    """True if the file is a UNMIXR map CSV (starts with an 'X num' header row)."""
    try:
        with open(path, encoding="utf-8-sig") as f:
            return f.readline().strip().lower().startswith("x num")
    except Exception:
        return False


def _compound_from_folder(folder):
    """Compound name from a dilution-series folder name: 'conc_tbz' -> 'TBZ'."""
    b = os.path.basename(os.path.normpath(folder))
    b = re.sub(r"^conc[_\- ]*", "", b, flags=re.IGNORECASE).strip("_- ")
    return b.upper() if b else "compound"


def load_calibration_folder(folder):
    """Build ONE compound's dilution series from a folder of per-concentration files
    (each file = one standard; the concentration is read from the filename, e.g.
    1nM / 10uM / 1_mM). Accepts UNMIXR map CSVs (uses the per-map mean spectrum) and
    plain spectra tables (.txt/.csv with wavenumber in column 0, spectra in the other
    columns — each column kept as a replicate). Returns (axis, compound_name,
    concentrations (k,), spectra (k, n_feat)); raises if fewer than 2 usable files."""
    from real_data import load_map                       # local: avoid import cycle
    # prefer baseline-corrected exports — match any '…corrected….csv' so map exports
    # like 'DQ_1mM_10 points_corrected_map.csv' are included, not only '…_corrected.csv'
    files = sorted(glob.glob(os.path.join(folder, "*corrected*.csv")))
    if not files:
        files = (sorted(glob.glob(os.path.join(folder, "*.csv")))
                 + sorted(glob.glob(os.path.join(folder, "*.txt"))))
    files = [p for p in files if os.path.basename(p).lower() not in _NON_MAP]
    concs, specs, axis, skipped, n_files = [], [], None, [], 0
    for p in files:
        base = os.path.splitext(os.path.basename(p))[0]
        c = parse_concentration(base)
        if c is None:
            skipped.append(base); continue
        try:
            if _is_map_csv(p):
                wn, _cube, mean, _coord = load_map(p); rows = [mean]
            else:
                wn, sp = read_spectra_table(p); rows = list(sp)  # each column = replicate
        except Exception as exc:
            skipped.append(f"{base} ({type(exc).__name__})"); continue
        if axis is not None and len(wn) != len(axis):
            raise ValueError(f"{base}: {len(wn)} points vs {len(axis)} — the "
                             "dilution files must share one wavenumber axis.")
        axis = wn; n_files += 1
        for r in rows:
            concs.append(c); specs.append(r)
    if n_files < 2:
        raise ValueError(
            f"need ≥2 concentration files with a parseable name (1nM, 10uM, 1_mM…) "
            f"in {os.path.basename(folder)}; found {n_files}"
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


def _is_number(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def load_calibration_csv(path):
    """Return (axis, {compound: (concentrations (k,), spectra (k, n_feat))})."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    wn_cols = [x for x in header[2:] if x != ""]
    # this loader needs the SPECTRA form (compound, concentration_M, <wavenumbers…>).
    # A common mistake is feeding calibration_curve.csv / calibration_fit.csv, whose
    # 3rd column is a label ('B', 'model', …) — fail with an actionable message
    # instead of a bare float('B') ValueError.
    if not wn_cols or not all(_is_number(x) for x in wn_cols):
        got = header[2] if len(header) > 2 else "(none)"
        raise ValueError(
            f"'{os.path.basename(path)}' is not a calibration-SPECTRA CSV: its 3rd "
            f"column header is '{got}', expected a wavenumber. Load "
            f"'calibration_spectra.csv' (header: compound, concentration_M, <wavenumbers…>) "
            f"or the original dilution-series folder — not calibration_curve.csv / "
            f"calibration_fit.csv (those are report tables, not spectra).")
    axis = np.array([float(x) for x in wn_cols])
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


def write_readme(folder, title, sections, figures=None):
    """Write README.md into ``folder``: a title, each section (heading → list of markdown
    lines, in insertion order), then a Figures list of (name, description). So the exported
    PNG/CSV — read out of context — still say WHAT they are, HOW they were made, and the
    RESULT. ``sections`` is an ordered dict {heading: [lines]}."""
    lines = [f"# {title}", ""]
    for heading, body in sections.items():
        lines.append(f"## {heading}")
        lines += list(body)
        lines.append("")
    if figures:
        lines.append("## Figures")
        lines += [f"- {name}.png — {desc}" for name, desc in figures]
    with open(os.path.join(folder, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
