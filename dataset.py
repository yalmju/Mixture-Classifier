"""dataset.py — turn a data folder into a dataset spec, so the app is not tied to
any particular substances. The DQ / THI / TBZ / BLK pesticides are just the
example that ships; any set of reference SERS maps works.

Layout of a data folder:

    Reference/<Class>_corrected.csv   one pure substance per file  (required)
    Ratio/<Mix>_corrected.csv         a measured mixture map        (optional)
    mixtures.csv                      Mix,<Class1>,<Class2>,…  nominal parts (optional)

- A reference class name = file stem with a trailing "_corrected" removed.
- A class whose name is blank / blk / background / bg / none (any case) is treated
  as the no-substance class and is kept LAST in the class order.
- Mixtures come from mixtures.csv when present; callers may fall back to a built-in
  default (the pesticide example) when it is absent.

Pure filename / CSV plumbing — no numpy, so it is safe to import anywhere.
"""
from __future__ import annotations

import csv
import glob
import os

BLANK_ALIASES = {"blk", "blank", "background", "bg", "none"}
_SUFFIX = "_corrected"


def class_name(path):
    """Reference class name from a map path: stem minus a trailing '_corrected'."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.lower().endswith(_SUFFIX):
        stem = stem[: -len(_SUFFIX)]
    return stem


def is_blank(name):
    return name.lower() in BLANK_ALIASES


def reference_dir(data_dir):
    """Accept either the data root (maps under Reference/) or the folder of maps
    itself. Returns whichever actually holds reference CSVs."""
    sub = os.path.join(data_dir, "Reference")
    if glob.glob(os.path.join(sub, "*.csv")):
        return sub
    return data_dir


def discover_references(data_dir):
    """Return an ordered dict-like list of (class_name, path) for every reference
    map found: non-blank classes alphabetical, blank class(es) last. Prefers
    *_corrected.csv; if none, falls back to any *.csv in the reference dir."""
    rd = reference_dir(data_dir)
    files = sorted(glob.glob(os.path.join(rd, "*_corrected.csv")))
    if not files:
        files = sorted(glob.glob(os.path.join(rd, "*.csv")))
    pairs = [(class_name(p), p) for p in files]
    non_blank = [(n, p) for n, p in pairs if not is_blank(n)]
    blank = [(n, p) for n, p in pairs if is_blank(n)]
    non_blank.sort(key=lambda t: t[0].lower())
    return non_blank + blank


def ratio_map_path(data_dir, mix_name):
    """Path to a mixture map for `mix_name`, or None. Looks under Ratio/ (then the
    data root) for <name>_corrected.csv then <name>.csv."""
    roots = [os.path.join(data_dir, "Ratio"), data_dir]
    for root in roots:
        for fn in (mix_name + "_corrected.csv", mix_name + ".csv"):
            p = os.path.join(root, fn)
            if os.path.exists(p):
                return p
    return None


def load_mixtures(data_dir, comps):
    """Mixture composition manifest as {mix_name: [parts aligned to `comps`]}.

    Reads mixtures.csv (header: Mix, <comp>, <comp>, …). Columns may be any subset
    of the components; missing/blank cells are 0. Returns None when the file is
    absent, so the caller can fall back to a built-in default."""
    p = os.path.join(data_dir, "mixtures.csv")
    if not os.path.exists(p):
        return None
    with open(p, newline="") as f:
        rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    if not rows:
        return {}
    header = [h.strip() for h in rows[0]]
    col = {h: i for i, h in enumerate(header)}
    out = {}
    for r in rows[1:]:
        name = r[0].strip()
        if not name:
            continue
        vec = []
        for c in comps:
            j = col.get(c)
            cell = r[j].strip() if (j is not None and j < len(r)) else ""
            vec.append(float(cell) if cell else 0.0)
        out[name] = vec
    return out
