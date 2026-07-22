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
import re

BLANK_ALIASES = {"blk", "blank", "background", "bg", "none"}
_SUFFIX = "_corrected"
_BATCH_RE = re.compile(r"^(.*?)[ _\-]+(\d+)$")


def base_and_batch(name):
    """Split a reference name into (substance, batch#) so repeat measurements of
    the SAME substance group together instead of becoming separate classes:
    'THI_2' -> ('THI', 2), 'TBZ 3' -> ('TBZ', 3), 'THI' -> ('THI', 1). A name that
    is only a number, or has no base before the number, is left as-is."""
    m = _BATCH_RE.match(name)
    if m and m.group(1).strip(" _-"):
        return m.group(1).strip(" _-"), int(m.group(2))
    return name, 1


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


def load_manifest(data_dir):
    """Read samples.csv (columns: file, class[, batch]) -> {abs_path: (class,
    batch)}, or None if absent. Lets the Sampling tab pin how files map to
    classes/batches, overriding the filename heuristic."""
    p = os.path.join(data_dir, "samples.csv")
    if not os.path.exists(p):
        return None
    rd = reference_dir(data_dir)
    out = {}
    with open(p, newline="") as f:
        rows = [r for r in csv.reader(f) if r and r[0].strip()]
    for r in rows[1:]:
        fn = r[0].strip()
        cls = r[1].strip() if len(r) > 1 else ""
        batch = int(r[2]) if len(r) > 2 and r[2].strip().isdigit() else 1
        cand = fn if os.path.isabs(fn) else os.path.join(rd, fn)
        if not os.path.exists(cand):
            cand = os.path.join(data_dir, fn)
        out[os.path.abspath(cand)] = (cls, batch)
    return out


def save_manifest(data_dir, rows):
    """Write <data_dir>/samples.csv from rows of (filename, class, batch)."""
    p = os.path.join(data_dir, "samples.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "class", "batch"])
        for fn, cls, batch in rows:
            w.writerow([fn, cls, batch])
    return p


def discover_dataset(data_dir):
    """Group reference maps into classes, merging batches of the same substance.

    Returns an ordered list ``[(class_name, [(batch, path), …]), …]`` — non-blank
    classes alphabetical, blank class(es) last, batches sorted. Uses samples.csv
    when present (Sampling tab), otherwise auto-groups by base name so 'THI' and
    'THI_2' land in one 'THI' class rather than two."""
    refs = discover_references(data_dir)
    manifest = load_manifest(data_dir)
    groups = {}
    for name, path in refs:
        cls = batch = None
        if manifest is not None:
            hit = manifest.get(os.path.abspath(path))
            if hit and hit[0]:
                cls, batch = hit
        if cls is None:
            cls, batch = base_and_batch(name)
        groups.setdefault(cls, []).append((batch, path))
    non_blank = sorted((c for c in groups if not is_blank(c)), key=str.lower)
    blank = [c for c in groups if is_blank(c)]
    return [(c, sorted(groups[c])) for c in (non_blank + blank)]


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
