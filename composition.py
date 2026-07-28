"""composition.py — reuse the repo's NNLS unmixing (unmix.unmix_map) to read out,
per reference/mixture map: the per-pixel composition (for a colour blend), the
nominal→measured drift, and the apparent (intensity) recovery. GUI-independent;
results are cached to <data_dir>/composition_cache.pkl so re-plots are instant."""
import os
import glob
import pickle

import numpy as np

from unmix import unmix_map
from validate import parse_mixture_label
from dataset import reference_dir

SUBSTANCES = ["DQ", "TBZ", "THI"]     # canonical column order for every fraction row
_CACHE = "composition_cache.pkl"


def map_files(data_dir):
    """The maps to analyse: the known-ratio mixtures in Reference/Ratio plus one
    pure DQ / TBZ corner (the THI corner is covered by the THI* dilutions)."""
    ref = reference_dir(data_dir)
    files = sorted(glob.glob(os.path.join(ref, "Ratio", "*.csv")))
    for s in ("DQ", "TBZ"):
        p = os.path.join(ref, f"{s}-1.csv")
        if os.path.exists(p):
            files.append(p)
    return files


def nominal_fraction(basename):
    """Nominal composition (DQ, TBZ, THI fractions) parsed from the file name, or
    None if unrecognised. e.g. 'DQ1TH3' → [0.25, 0, 0.75]; 'THI1' → [0, 0, 1]."""
    r = parse_mixture_label(basename, SUBSTANCES)
    if not r:
        return None
    v = np.array([float(r.get(s, 0.0)) for s in SUBSTANCES])
    tot = v.sum()
    return v / tot if tot > 0 else None


def _one_map(data_dir, path, baseline):
    r = unmix_map(data_dir, path, method="nnls", baseline=baseline)
    nb = [r.comps[i] for i in r.nonbg]            # substance names in the result's order
    frac = np.zeros((r.ratio_nb.shape[0], len(SUBSTANCES)))
    for s in SUBSTANCES:
        if s in nb:
            frac[:, SUBSTANCES.index(s)] = r.ratio_nb[:, nb.index(s)]
    hit = np.asarray(r.hit, bool)
    mean = frac[hit].mean(axis=0) if hit.any() else frac.mean(axis=0)
    return dict(name=os.path.splitext(os.path.basename(path))[0],
                frac=frac, coords=np.asarray(r.coords, float), hit=hit,
                reliab=getattr(r, "reliab", None), mean=mean,
                nominal=nominal_fraction(os.path.basename(path)))


def compute_composition(data_dir, baseline=True, files=None, progress=None, use_cache=True):
    """Unmix each map and return a list of per-map dicts: name, frac (n_pix×3),
    coords, hit, reliab, mean (3,), nominal (3,) or None. ``files`` is an explicit
    list of map CSVs; if None, the Reference/Ratio mixtures are auto-discovered."""
    files = list(files) if files else map_files(data_dir)
    if not files:
        raise FileNotFoundError(
            "no maps to analyse — load map CSVs, or set a data folder with a "
            "Reference/Ratio/ folder of known-ratio mixtures (e.g. DQ1TH1_corrected.csv).")
    cache_path = os.path.join(data_dir, _CACHE)
    cache = {}
    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}
    out = []
    for i, p in enumerate(files):
        if progress:
            progress(f"unmixing {os.path.basename(p)}  ({i + 1}/{len(files)})")
        key = (os.path.abspath(p), round(os.path.getmtime(p), 3), bool(baseline))
        rec = cache.get(key)
        if rec is None:
            rec = _one_map(data_dir, p, baseline)
            cache[key] = rec
        out.append(rec)
    if use_cache:
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)
        except Exception:
            pass
    return out


# ---- geometry / metrics shared by the plots -------------------------------
# ternary corners: DQ top, THI bottom-left, TBZ bottom-right (per the spec)
_TOP = np.array([0.5, np.sqrt(3) / 2])       # DQ
_BL = np.array([0.0, 0.0])                    # THI
_BR = np.array([1.0, 0.0])                    # TBZ


def bary(frac):
    """(DQ, TBZ, THI) fractions → (x, y) inside the ternary triangle."""
    f = np.asarray(frac, float)
    return f[0] * _TOP + f[2] * _BL + f[1] * _BR


def composition_distance(nominal, measured):
    """Euclidean distance between two 3-fraction vectors."""
    return float(np.linalg.norm(np.asarray(nominal) - np.asarray(measured)))


def aitchison_distance(nominal, measured, eps=1e-3):
    """Aitchison (centred-log-ratio) distance between two compositions."""
    a = np.clip(np.asarray(nominal, float), eps, None); a = a / a.sum()
    b = np.clip(np.asarray(measured, float), eps, None); b = b / b.sum()
    la = np.log(a) - np.log(a).mean()
    lb = np.log(b) - np.log(b).mean()
    return float(np.linalg.norm(la - lb))
