"""
SERS Compound Discriminator
============================
Single-file app that:
  1. Loads one or more *reference* SERS map CSVs (e.g., Ref_TBZ.csv, Ref_THI.csv).
     Each reference contributes a mean spectrum used as the compound signature.
  2. Loads one *testing* SERS mapping CSV (3D cube: Y x X x wavenumbers).
  3. Optionally applies arPLS baseline correction (per-pixel for test, per-reference).
  4. Computes three discrimination metrics for every pixel:
       - Pearson correlation + softmax across references (probability)
       - Cosine similarity per reference (0..1)
       - NNLS unmixing weights per reference (contribution)
  5. Shows an interactive viewer with:
       - intensity map per reference (WN dropdown of top peaks)
       - probability / similarity / unmixing map per reference (metric toggle)
       - "Save All" -> PNGs + CSV exports

CSV format (input, identical to sers_baseline.py / sers_map_visualizer_v8.py):
    X num,<nx>
    Y num,<ny>
    X,Y,wn1,wn2,...
    x0,y0,i00,i01,...
    ...

Run:
    python sers_discriminator.py
"""

import csv
import io
import os
import sys
from pathlib import Path

# UTF-8 console (Korean Windows etc.)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks
from scipy.optimize import nnls

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button, RadioButtons, TextBox
import matplotlib.patches as mpatches

# ---- Global style: Arial 18pt for all output text (with DejaVu fallback) ----
plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 18
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["figure.titlesize"] = 20
plt.rcParams["axes.unicode_minus"] = True

import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk


# =============================================================================
# Defaults / constants
# =============================================================================

BUILD_TAG = "build 2026-05-20 r2"

DEFAULT_LAMBDA = 1e6
DEFAULT_RATIO = 1e-6
DEFAULT_MAX_ITER = 100

DEFAULT_TOP_PEAKS = 5
DEFAULT_PEAK_PROMINENCE_FRAC = 0.05  # 5% of (max-min) of the mean spectrum

# Pastel colormaps: white -> light pastel -> deeper pastel. Pre-built so
# downstream code can pass either a Colormap instance or its registered name.
_PASTEL_CMAP_STOPS = {
    "PastelRed":    ["#ffffff", "#ffe4e4", "#ffb8b8", "#f08a8a"],
    "PastelGreen":  ["#ffffff", "#e4f5e4", "#b8e0b8", "#8ac78a"],
    "PastelBlue":   ["#ffffff", "#e4ecf7", "#b8c8e8", "#8aaee5"],
    "PastelOrange": ["#ffffff", "#fff0d8", "#ffd49a", "#f0a85a"],
    "PastelPurple": ["#ffffff", "#f0e4f7", "#d8b8e8", "#b88ae5"],
    "PastelGrey":   ["#ffffff", "#ececec", "#c8c8c8", "#9a9a9a"],
}
PASTEL_CMAPS = {
    name: LinearSegmentedColormap.from_list(name, stops, N=256)
    for name, stops in _PASTEL_CMAP_STOPS.items()
}
REF_CMAPS = ["PastelRed", "PastelGreen", "PastelBlue",
             "PastelOrange", "PastelPurple", "PastelGrey"]
# Pastel marker / overlay tints (a touch darker than the cmap max so they
# remain readable as line colors on white backgrounds).
REF_TINTS = [
    (0.92, 0.55, 0.55),  # pastel red
    (0.55, 0.78, 0.55),  # pastel green
    (0.55, 0.65, 0.88),  # pastel blue
    (0.94, 0.66, 0.35),  # pastel orange
    (0.72, 0.55, 0.88),  # pastel purple
    (0.60, 0.60, 0.60),  # pastel grey
]


def _resolve_cmap(name_or_cmap):
    """Accept a cmap name (built-in pastel, matplotlib name) OR a Colormap
    instance OR a hex color like '#fae630'. Returns a Colormap.

    Hex codes used to crash plt.get_cmap; now they're handed off to the more
    permissive resolver which builds a custom white -> hex pastel cmap on
    the fly.
    """
    if hasattr(name_or_cmap, "_segmentdata") or hasattr(name_or_cmap, "colors"):
        return name_or_cmap
    if isinstance(name_or_cmap, str):
        s = name_or_cmap.strip()
        if s in PASTEL_CMAPS:
            return PASTEL_CMAPS[s]
        # Hex (#abc, #aabbcc) or any other matplotlib-recognized spec:
        # delegate to the permissive resolver used by the GUI color picker.
        cmap_obj, _ = _resolve_user_cmap_and_tint(s)
        if cmap_obj is not None:
            return cmap_obj
    # Last resort — let matplotlib raise with its own helpful message
    return plt.get_cmap(name_or_cmap)


def _resolve_user_cmap_and_tint(text):
    """Resolve user-typed cmap spec into (Colormap, tint_rgb).

    Accepts:
      - Pastel name: PastelRed, PastelGreen, ...
      - Any matplotlib cmap name: Reds, viridis, hot, ...
      - Hex color: #RRGGBB or #RGB -> white-to-hex pastel cmap
    Returns (cmap, tint) or (None, None) if the spec can't be parsed.
    """
    import matplotlib.colors as mcolors
    if text is None:
        return None, None
    text = text.strip()
    if not text:
        return None, None
    # Hex color -> build custom cmap
    if text.startswith("#"):
        try:
            rgb = mcolors.to_rgb(text)
        except Exception:
            return None, None
        # White -> 50% mix -> full color
        mid = tuple(1.0 - 0.5 * (1.0 - c) for c in rgb)  # halfway to color
        cmap = LinearSegmentedColormap.from_list(
            f"user_{text.lstrip('#')}", ["#ffffff", mid, rgb], N=256)
        return cmap, rgb
    # Pastel built-in
    if text in PASTEL_CMAPS:
        cmap = PASTEL_CMAPS[text]
        tint = tuple(cmap(0.95)[:3])
        return cmap, tint
    # Any matplotlib cmap name
    try:
        cmap = plt.get_cmap(text)
        tint = tuple(cmap(0.95)[:3])
        return cmap, tint
    except Exception:
        return None, None

CM1 = r"$\mathrm{cm^{-1}}$"
FIGURE_DPI = 300


# =============================================================================
# CSV I/O
# =============================================================================

def _looks_like_transposed_header(first_row):
    """Detect Format B header: first column is wavenumber/raman/wn or numeric,
    rest of the columns are spectra (spec_01, pixel_0, etc.)."""
    if not first_row:
        return False
    first = first_row[0].strip().lower()
    if first in ("wavenumber", "wavenumbers", "raman_shift",
                 "raman shift", "raman_shift_cm-1", "wn", "cm-1",
                 "cm^-1"):
        return True
    # Or: first cell numeric (header-less transposed file) AND many columns
    try:
        float(first)
        return len(first_row) > 4
    except ValueError:
        return False


def parse_transposed_csv(filepath):
    """Parse a Format B (transposed) CSV.

    Layout:
        wavenumber, spec_01, spec_02, ..., spec_N
        100.0,      i_11,    i_12,    ..., i_1N
        101.55,     i_21,    i_22,    ..., i_2N
        ...

    For a reference, we only need the per-wavenumber spectra (one column per
    spot), so the spatial grid is mostly irrelevant — we guess sqrt(N) x
    sqrt(N) if perfect-square, otherwise 1 x N (a flat strip). The returned
    dict matches parse_xy_map_csv's so downstream code is unchanged.
    """
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    rows = [r for r in rows if r and any(c.strip() for c in r)]
    if not rows:
        raise ValueError(f"Empty CSV: {filepath}")

    # Detect optional header
    try:
        float(rows[0][0])
        data_start = 0
    except ValueError:
        data_start = 1
    if data_start >= len(rows):
        raise ValueError(f"No numeric rows in {filepath}")

    n_specs = len(rows[data_start]) - 1
    if n_specs < 1:
        raise ValueError(
            f"Transposed format expected wavenumber + 1+ spectra columns "
            f"in {filepath}; got {len(rows[data_start])} column(s)."
        )

    wn_list = []
    intens_T = []  # rows = wavenumbers, cols = pixels
    for r in rows[data_start:]:
        try:
            wn_list.append(float(r[0]))
        except ValueError:
            continue
        spec_row = []
        for v in r[1:1 + n_specs]:
            s = v.strip()
            if not s:
                spec_row.append(0.0)
            else:
                try:
                    spec_row.append(float(s))
                except ValueError:
                    spec_row.append(0.0)
        if len(spec_row) != n_specs:
            # Pad shorter rows so np.array works
            spec_row += [0.0] * (n_specs - len(spec_row))
        intens_T.append(spec_row)

    wavenumbers = np.array(wn_list, dtype=float)
    intens_T = np.array(intens_T, dtype=float)
    # Transpose so shape is (n_pixels, n_wavenumbers) for cube building
    intens = intens_T.T

    # Guess a square-ish grid
    side = int(round(np.sqrt(n_specs)))
    if side * side == n_specs:
        nx, ny = side, side
    else:
        nx, ny = n_specs, 1

    cube = intens.reshape(ny, nx, len(wavenumbers))
    # Synthetic spatial coords (no real X/Y in transposed format)
    x_coords = np.arange(nx, dtype=float) * 100.0
    y_coords = (np.arange(ny, dtype=float) * 100.0)[::-1]

    print(f"  [Format B / transposed] {filepath.name}: "
          f"{len(wavenumbers)} wn x {n_specs} spectra "
          f"(guessed grid {nx}x{ny})")

    return {
        "x_num": nx,
        "y_num": ny,
        "wavenumbers": wavenumbers,
        "x_coords": x_coords,
        "y_coords": y_coords,
        "cube": cube,
        "pixel_order": [(float(x), float(y))
                        for y in y_coords for x in x_coords],
    }


def parse_csv_auto(filepath):
    """Format-detecting parser.

    Tries the XY-mapping format first (`X num`/`Y num` header). If that
    fails, falls back to the transposed (`wavenumber, spec_01, ...`) format.
    Raises a clear error if neither parser succeeds.
    """
    filepath = Path(filepath)
    # Peek at first non-empty row
    with open(filepath, "r", encoding="utf-8-sig") as f:
        peek_rows = []
        for _ in range(3):
            line = f.readline()
            if not line:
                break
            r = next(csv.reader([line]))
            if r:
                peek_rows.append(r)
    if not peek_rows:
        raise ValueError(f"Empty CSV: {filepath}")

    first = peek_rows[0][0].strip().lower()
    if first == "x num":
        return parse_xy_map_csv(filepath)
    if _looks_like_transposed_header(peek_rows[0]):
        return parse_transposed_csv(filepath)
    # Unknown — try XY first, fall back to transposed with clearer message
    try:
        return parse_xy_map_csv(filepath)
    except Exception as e_xy:
        try:
            return parse_transposed_csv(filepath)
        except Exception as e_tp:
            raise ValueError(
                f"Could not parse '{filepath.name}' as either format.\n"
                f"  XY-map parser said: {e_xy}\n"
                f"  Transposed parser said: {e_tp}"
            ) from e_tp


def parse_xy_map_csv(filepath):
    """
    Parse an XY-mapping CSV (the format produced by sers_baseline / converter).

    Returns dict:
        x_num, y_num     : declared grid dimensions
        wavenumbers      : 1D array
        x_coords, y_coords : sorted unique x, y arrays
        cube             : (ny, nx, n_wn) float array
        pixel_order      : list of (x, y) tuples in original row order
    """
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    x_num = y_num = None
    header_row_idx = None
    for i, r in enumerate(rows):
        if not r:
            continue
        first = r[0].strip().lower()
        if first == "x num" and len(r) > 1 and r[1].strip():
            try:
                x_num = int(float(r[1]))
            except ValueError:
                pass
        elif first == "y num" and len(r) > 1 and r[1].strip():
            try:
                y_num = int(float(r[1]))
            except ValueError:
                pass
        elif first == "x" and len(r) > 2 and r[1].strip().lower() == "y":
            try:
                float(r[2])
                header_row_idx = i
                break
            except ValueError:
                continue

    if header_row_idx is None:
        raise ValueError(f"Could not find 'X,Y,wn,...' header row in {filepath}")

    header = rows[header_row_idx]
    wavenumbers = []
    for v in header[2:]:
        s = v.strip()
        if not s:
            break
        try:
            wavenumbers.append(float(s))
        except ValueError:
            break
    wavenumbers = np.array(wavenumbers, dtype=float)
    n_wn = len(wavenumbers)
    if n_wn == 0:
        raise ValueError(f"No numeric wavenumbers in header of {filepath}")

    x_vals, y_vals, spectra, pixel_order = [], [], [], []
    for r in rows[header_row_idx + 1:]:
        if not r or len(r) < 3:
            continue
        if not r[0].strip() or not r[1].strip():
            continue
        try:
            x = float(r[0])
            y = float(r[1])
        except ValueError:
            continue
        spec = []
        ok = True
        for v in r[2: 2 + n_wn]:
            s = v.strip()
            if not s:
                spec.append(np.nan)
            else:
                try:
                    spec.append(float(s))
                except ValueError:
                    ok = False
                    break
        if not ok or len(spec) != n_wn:
            continue
        x_vals.append(x)
        y_vals.append(y)
        spectra.append(spec)
        pixel_order.append((x, y))

    if not spectra:
        raise ValueError(f"No spectral rows in {filepath}")

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    spectra = np.array(spectra, dtype=float)

    unique_x = np.sort(np.unique(x_vals))
    # use top->bottom convention to match the visualizer
    unique_y = np.sort(np.unique(y_vals))[::-1]
    if x_num is None:
        x_num = len(unique_x)
    if y_num is None:
        y_num = len(unique_y)

    cube = np.full((len(unique_y), len(unique_x), n_wn), np.nan)
    x_index = {v: i for i, v in enumerate(unique_x)}
    y_index = {v: i for i, v in enumerate(unique_y)}
    for x, y, spec in zip(x_vals, y_vals, spectra):
        cube[y_index[y], x_index[x]] = spec
    cube = np.nan_to_num(cube, nan=0.0)

    return {
        "x_num": x_num,
        "y_num": y_num,
        "wavenumbers": wavenumbers,
        "x_coords": unique_x,
        "y_coords": unique_y,
        "cube": cube,
        "pixel_order": pixel_order,
    }


# =============================================================================
# arPLS baseline correction
# =============================================================================

def arpls(y, lam=DEFAULT_LAMBDA, ratio=DEFAULT_RATIO, max_iter=DEFAULT_MAX_ITER):
    """arPLS baseline estimate of a 1D spectrum."""
    y = np.asarray(y, dtype=float)
    N = len(y)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(N - 2, N), format="csc")
    H = lam * D.T.dot(D)
    w = np.ones(N)
    z = y.copy()
    for _ in range(max_iter):
        W = sparse.diags(w, 0, format="csc")
        Z = W + H
        z = spsolve(Z, w * y)
        d = y - z
        dn = d[d < 0]
        if len(dn) == 0:
            break
        m = np.mean(dn)
        s = np.std(dn)
        if s == 0:
            break
        w_new = 1.0 / (1.0 + np.exp(2.0 * (d - (2 * s - m)) / s))
        if np.linalg.norm(w_new - w) / (np.linalg.norm(w) + 1e-12) < ratio:
            break
        w = w_new
    return z


def baseline_correct_cube(cube, lam, ratio, max_iter, label="cube", report=True):
    """Apply arPLS to every pixel of an (ny, nx, n_wn) cube. Returns corrected cube."""
    ny, nx, n_wn = cube.shape
    out = np.zeros_like(cube)
    total = ny * nx
    count = 0
    for yi in range(ny):
        for xi in range(nx):
            spec = cube[yi, xi, :]
            base = arpls(spec, lam=lam, ratio=ratio, max_iter=max_iter)
            out[yi, xi, :] = spec - base
            count += 1
            if report and (count % 50 == 0 or count == total):
                print(f"  [{label}] arPLS {count}/{total} pixels", end="\r")
    if report:
        print(f"  [{label}] arPLS {total}/{total} pixels  done.")
    return out


def baseline_correct_spectrum(spec, lam, ratio, max_iter):
    base = arpls(spec, lam=lam, ratio=ratio, max_iter=max_iter)
    return spec - base


# =============================================================================
# Wavenumber alignment
# =============================================================================

def align_to_axis(src_wn, src_spec, target_wn):
    """Linearly interpolate src_spec (on src_wn) onto target_wn. Out-of-range -> 0."""
    return np.interp(target_wn, src_wn, src_spec, left=0.0, right=0.0)


def maybe_crop(wn, cube_or_spec, wn_min, wn_max):
    """Return (wn_cropped, data_cropped) using a wn min/max range. None values skipped."""
    mask = np.ones_like(wn, dtype=bool)
    if wn_min is not None:
        mask &= wn >= wn_min
    if wn_max is not None:
        mask &= wn <= wn_max
    if cube_or_spec.ndim == 1:
        return wn[mask], cube_or_spec[mask]
    return wn[mask], cube_or_spec[..., mask]


# =============================================================================
# Reference handling
# =============================================================================

def reference_mean_spectrum(cube):
    """Mean across all pixels -> 1D reference signature."""
    ny, nx, n_wn = cube.shape
    return cube.reshape(-1, n_wn).mean(axis=0)


def detect_top_peaks(wn, spec, n_peaks=DEFAULT_TOP_PEAKS,
                     prominence_frac=DEFAULT_PEAK_PROMINENCE_FRAC):
    """Find top-N peaks by prominence in a 1D spectrum.

    Returns list of (wn_value, intensity) sorted by descending prominence.
    Always returns at least one entry (argmax fallback).
    """
    spec = np.asarray(spec, dtype=float)
    if spec.size < 3:
        idx = int(np.argmax(spec))
        return [(float(wn[idx]), float(spec[idx]))]

    span = float(spec.max() - spec.min())
    if span <= 0:
        idx = int(np.argmax(spec))
        return [(float(wn[idx]), float(spec[idx]))]
    prom = prominence_frac * span

    try:
        peak_idx, props = find_peaks(spec, prominence=prom)
    except Exception:
        peak_idx, props = np.array([], dtype=int), {}

    if len(peak_idx) == 0:
        # fallback: top-N by raw intensity
        order = np.argsort(spec)[::-1][:n_peaks]
        return [(float(wn[i]), float(spec[i])) for i in order]

    proms = props.get("prominences", np.full(len(peak_idx), 1.0))
    order = np.argsort(proms)[::-1][:n_peaks]
    return [(float(wn[peak_idx[i]]), float(spec[peak_idx[i]])) for i in order]


# =============================================================================
# Discrimination metrics
# =============================================================================

def _flat_pixels(cube):
    """(ny, nx, n_wn) -> (ny*nx, n_wn)"""
    ny, nx, n_wn = cube.shape
    return cube.reshape(-1, n_wn), ny, nx


def pearson_softmax_map(cube, ref_specs, temperature=1.0):
    """Pearson per pixel per ref, then softmax across refs.

    Args:
        cube      : (ny, nx, n_wn)
        ref_specs : (n_ref, n_wn) reference mean spectra (aligned)
        temperature: softmax temperature (higher = softer)

    Returns:
        prob_maps : (n_ref, ny, nx) probabilities summing to 1 across refs
        corr_maps : (n_ref, ny, nx) raw Pearson correlations
    """
    flat, ny, nx = _flat_pixels(cube)
    n_pix, n_wn = flat.shape
    n_ref = ref_specs.shape[0]

    # Standardize
    f_mean = flat.mean(axis=1, keepdims=True)
    f_std = flat.std(axis=1, keepdims=True) + 1e-12
    f_norm = (flat - f_mean) / f_std

    r_mean = ref_specs.mean(axis=1, keepdims=True)
    r_std = ref_specs.std(axis=1, keepdims=True) + 1e-12
    r_norm = (ref_specs - r_mean) / r_std

    # Pearson = mean(x_z * y_z)
    corr = (f_norm @ r_norm.T) / n_wn  # (n_pix, n_ref)

    # Softmax across references (per pixel)
    logits = corr / max(temperature, 1e-9)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    prob = e / (e.sum(axis=1, keepdims=True) + 1e-12)

    corr_maps = corr.T.reshape(n_ref, ny, nx)
    prob_maps = prob.T.reshape(n_ref, ny, nx)
    return prob_maps, corr_maps


def cosine_map(cube, ref_specs):
    """Cosine similarity per pixel per reference. Clipped to [0,1]."""
    flat, ny, nx = _flat_pixels(cube)
    f_norm = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12)
    r_norm = ref_specs / (np.linalg.norm(ref_specs, axis=1, keepdims=True) + 1e-12)
    cos = f_norm @ r_norm.T  # (n_pix, n_ref)
    cos = np.clip(cos, 0.0, 1.0)
    return cos.T.reshape(ref_specs.shape[0], ny, nx)


def nnls_map(cube, ref_specs):
    """NNLS unmixing per pixel: weights >= 0 of each ref.

    Returns (weight_maps, weight_norm_maps):
      weight_maps     : raw NNLS weights (n_ref, ny, nx)
      weight_norm_maps: weights normalized per pixel so they sum to 1
                        (zero-pixel rows stay zero).
    """
    flat, ny, nx = _flat_pixels(cube)
    n_pix, n_wn = flat.shape
    n_ref = ref_specs.shape[0]
    A = ref_specs.T  # (n_wn, n_ref) — columns are reference spectra
    W = np.zeros((n_pix, n_ref))
    for i in range(n_pix):
        try:
            w, _ = nnls(A, flat[i])
        except Exception:
            w = np.zeros(n_ref)
        W[i] = w
        if (i + 1) % 200 == 0 or i + 1 == n_pix:
            print(f"  NNLS {i+1}/{n_pix} pixels", end="\r")
    print()

    raw = W.T.reshape(n_ref, ny, nx)
    s = W.sum(axis=1, keepdims=True)
    s[s <= 0] = 1.0
    norm = (W / s).T.reshape(n_ref, ny, nx)
    return raw, norm


def cls_map(cube, ref_specs):
    """Classical Least Squares (unconstrained) per pixel.

    Returns:
      raw_maps  : (n_ref, ny, nx) signed coefficients (can be negative)
      norm_maps : (n_ref, ny, nx) clipped-at-zero then renormalized so rows sum to 1
      neg_pct   : (n_ref,) % of pixels where that ref's coefficient is negative
    """
    flat, ny, nx = _flat_pixels(cube)
    n_pix, n_wn = flat.shape
    n_ref = ref_specs.shape[0]
    A = ref_specs.T  # (n_wn, n_ref)
    # Solve once for all pixels: A @ W.T = flat.T  -> W = lstsq(A, flat.T).T
    W, *_ = np.linalg.lstsq(A, flat.T, rcond=None)
    W = W.T  # (n_pix, n_ref)

    raw = W.T.reshape(n_ref, ny, nx)
    neg_pct = (W < 0).mean(axis=0) * 100.0  # per-ref % negative

    W_clip = np.clip(W, 0.0, None)
    s = W_clip.sum(axis=1, keepdims=True)
    s[s <= 0] = 1.0
    norm = (W_clip / s).T.reshape(n_ref, ny, nx)
    return raw, norm, neg_pct


def kfold_cv_references(ref_cubes_aligned, ref_names, k=5, seed=42):
    """K-fold cross-validation on reference pixels.

    For each fold:
        - train references = mean spectrum from pixels NOT in this fold
        - test pixels      = pixels IN this fold (across all refs)
        - classifier       = argmax(NNLS weights) using train references

    Returns dict with:
        confusion (n_ref x n_ref), precision/recall/f1 (per ref),
        accuracy, macro_f1, n_folds, ref_names, n_test_pixels.
    """
    n_refs = len(ref_cubes_aligned)
    if n_refs == 0:
        return None
    n_wn = ref_cubes_aligned[0].shape[2]

    # Flatten each ref cube into (n_pix_i, n_wn) and remember per-pixel labels
    flats = [c.reshape(-1, n_wn) for c in ref_cubes_aligned]
    n_per = [f.shape[0] for f in flats]

    rng = np.random.default_rng(seed)
    fold_of = [rng.integers(0, k, size=n_per[i]) for i in range(n_refs)]

    confusion = np.zeros((n_refs, n_refs), dtype=int)
    n_test_total = 0

    for fold in range(k):
        # Build per-fold train references
        train_refs = np.zeros((n_refs, n_wn))
        for i in range(n_refs):
            mask = fold_of[i] != fold
            if not mask.any():
                train_refs[i] = flats[i].mean(axis=0)
            else:
                train_refs[i] = flats[i][mask].mean(axis=0)

        A = train_refs.T  # (n_wn, n_refs)
        for ref_i in range(n_refs):
            mask = fold_of[ref_i] == fold
            test_pixels = flats[ref_i][mask]
            for px in test_pixels:
                try:
                    w, _ = nnls(A, px)
                except Exception:
                    w = np.zeros(n_refs)
                pred = int(np.argmax(w)) if w.sum() > 0 else ref_i
                confusion[ref_i, pred] += 1
                n_test_total += 1

    # Per-class precision, recall, F1
    precision = np.zeros(n_refs)
    recall = np.zeros(n_refs)
    f1 = np.zeros(n_refs)
    for i in range(n_refs):
        tp = confusion[i, i]
        fn = confusion[i, :].sum() - tp
        fp = confusion[:, i].sum() - tp
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = precision[i] + recall[i]
        f1[i] = 2 * precision[i] * recall[i] / denom if denom > 0 else 0.0

    accuracy = float(np.trace(confusion) / max(confusion.sum(), 1))
    macro_f1 = float(f1.mean())
    return {
        "confusion": confusion,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "n_folds": k,
        "ref_names": list(ref_names),
        "n_test_pixels": n_test_total,
    }


def synthetic_mixture_benchmark(ref_specs, n_synth=300, noise_level=0.05,
                                 seed=42):
    """Generate synthetic pixels as Dirichlet-weighted mixtures of refs +
    Gaussian noise. Recover weights via NNLS and compare to ground truth.
    """
    n_refs, n_wn = ref_specs.shape
    rng = np.random.default_rng(seed)
    # Mix of pure-ref pixels (1/3) and mixture pixels (2/3) to stress-test both
    pure_count = n_synth // 3
    mix_count = n_synth - pure_count
    true_w = np.zeros((n_synth, n_refs))
    # Pure
    for i in range(pure_count):
        true_w[i, rng.integers(0, n_refs)] = 1.0
    # Mixture (Dirichlet)
    if mix_count > 0:
        true_w[pure_count:] = rng.dirichlet(np.ones(n_refs), size=mix_count)

    # Synthetic spectra
    pixels = true_w @ ref_specs
    sigma = max(noise_level * float(np.std(pixels)), 1e-6)
    pixels = pixels + rng.normal(0, sigma, size=pixels.shape)

    A = ref_specs.T
    pred_raw = np.zeros((n_synth, n_refs))
    for i in range(n_synth):
        try:
            w, _ = nnls(A, pixels[i])
        except Exception:
            w = np.zeros(n_refs)
        pred_raw[i] = w
    s = pred_raw.sum(axis=1, keepdims=True)
    s[s <= 0] = 1.0
    pred_norm = pred_raw / s

    rmse_per_ref = np.sqrt(((pred_norm - true_w) ** 2).mean(axis=0))
    overall_rmse = float(np.sqrt(((pred_norm - true_w) ** 2).mean()))
    r_per_ref = []
    for i in range(n_refs):
        if true_w[:, i].std() > 0 and pred_norm[:, i].std() > 0:
            r = float(np.corrcoef(true_w[:, i], pred_norm[:, i])[0, 1])
        else:
            r = 0.0
        r_per_ref.append(r)
    return {
        "true_weights": true_w,
        "pred_norm": pred_norm,
        "rmse_per_ref": rmse_per_ref,
        "overall_rmse": overall_rmse,
        "correlation_per_ref": r_per_ref,
        "n_synth": n_synth,
        "noise_level": noise_level,
    }


def synthetic_noise_sweep(ref_specs, n_per_level=150,
                           noise_levels=(0.01, 0.05, 0.10, 0.20, 0.30, 0.50),
                           seed=42):
    """Stress test: synthetic_mixture_benchmark at multiple noise levels.

    Returns dict:
        noise_levels      : list of noise fractions tested
        overall_rmse      : (n_levels,) overall RMSE per noise level
        rmse_per_ref      : (n_levels, n_refs) per-ref RMSE at each level
        r_per_ref         : (n_levels, n_refs) per-ref Pearson r at each level
        n_per_level       : samples per noise level
        ref_names_hint    : None (filled by caller for plotting)
    """
    n_refs = ref_specs.shape[0]
    overall = np.zeros(len(noise_levels))
    rmse_pr = np.zeros((len(noise_levels), n_refs))
    r_pr    = np.zeros((len(noise_levels), n_refs))
    for i, noise in enumerate(noise_levels):
        out = synthetic_mixture_benchmark(
            ref_specs, n_synth=n_per_level, noise_level=float(noise),
            seed=seed + i,
        )
        overall[i] = out["overall_rmse"]
        rmse_pr[i] = out["rmse_per_ref"]
        r_pr[i] = out["correlation_per_ref"]
    return {
        "noise_levels": list(noise_levels),
        "overall_rmse": overall,
        "rmse_per_ref": rmse_pr,
        "r_per_ref": r_pr,
        "n_per_level": n_per_level,
    }


def confidence_maps(prob_3d):
    """Top1-top2 gap + Shannon entropy from a sum-to-1-per-pixel score map.

    Args:
        prob_3d : (n_ref, ny, nx) values that sum (approximately) to 1
                  along axis 0 per pixel.
    Returns:
        gap     : (ny, nx) top1 - top2   in [0, 1]; higher = more confident
        entropy : (ny, nx) Shannon entropy in [0, log2(n_ref)]; lower = more
                  confident.  Reported in bits (log base 2).
    """
    n_ref, ny, nx = prob_3d.shape
    flat = prob_3d.reshape(n_ref, -1)  # (n_ref, n_pix)
    sorted_desc = np.sort(flat, axis=0)[::-1]
    top1 = sorted_desc[0]
    top2 = sorted_desc[1] if n_ref > 1 else np.zeros_like(top1)
    gap = (top1 - top2).reshape(ny, nx)

    # Normalize per-pixel so entropy is well-defined even if rounding errors
    # make rows not quite sum to 1.
    s = flat.sum(axis=0, keepdims=True)
    s[s <= 0] = 1.0
    p = flat / s
    p = np.clip(p, 1e-12, 1.0)
    ent = -np.sum(p * np.log2(p), axis=0).reshape(ny, nx)
    return gap, ent


def argmax_and_agreement(*score_maps):
    """Compute per-method argmax + agreement-across-methods.

    Args:
        *score_maps : each is (n_ref, ny, nx). Pixel's predicted label =
                      ref index with the largest score in that method.

    Returns:
        argmax_per_method : list of (ny, nx) int label maps, one per input
        agreement         : (ny, nx) int in {1, ..., n_methods}.
                            n_methods = all methods agree on a single label;
                            1       = every method disagrees with every other.
        consensus         : (ny, nx) int label = majority vote across methods;
                            -1 if no majority (e.g. 3-way tie with 3 methods
                            and 3 different labels).
    """
    n_methods = len(score_maps)
    if n_methods == 0:
        raise ValueError("argmax_and_agreement needs at least one score map")
    ny, nx = score_maps[0].shape[1], score_maps[0].shape[2]
    n_ref = score_maps[0].shape[0]

    argmaxes = [np.argmax(s, axis=0) for s in score_maps]
    stack = np.stack(argmaxes, axis=0)  # (n_methods, ny, nx)

    # Per-pixel majority vote
    consensus = np.full((ny, nx), -1, dtype=int)
    agreement = np.zeros((ny, nx), dtype=int)
    for yi in range(ny):
        for xi in range(nx):
            col = stack[:, yi, xi]
            counts = np.bincount(col, minlength=n_ref)
            max_count = counts.max()
            # majority winners
            winners = np.where(counts == max_count)[0]
            if max_count > n_methods // 2:
                consensus[yi, xi] = int(winners[0])
            elif len(winners) == 1:
                consensus[yi, xi] = int(winners[0])
            else:
                consensus[yi, xi] = -1
            agreement[yi, xi] = int(max_count)
    return argmaxes, agreement, consensus


def mcr_als(cube, ref_specs, max_iter=12, tol=1e-5, normalize_S=True,
            report=True):
    """Hand-rolled MCR-ALS with non-negativity on both C and S.

    Initialization: S0 = ref_specs (assumed >= 0).
    Iteration:
        C : non-negative least squares per pixel  (rows of T solved against S^T)
        S : non-negative least squares per wn     (cols of T solved against C)
        Optional row-normalize S to unit max each step; rescale C accordingly.

    Args:
        cube       : (ny, nx, n_wn) test cube
        ref_specs  : (n_ref, n_wn) initial spectra estimate
        max_iter   : max ALS iterations
        tol        : relative change in residual norm at which to stop
        normalize_S: if True, scale each S row to unit max each iteration
                     (fixes the scaling indeterminacy)
        report     : print progress

    Returns:
        C_map   : (n_ref, ny, nx) concentrations
        C_contrib: (n_ref, ny, nx) per-pixel normalized contributions (sum to 1)
        S       : (n_ref, n_wn) resolved spectra
        resid   : final residual norm
        n_iter  : actual iterations run
    """
    flat, ny, nx = _flat_pixels(cube)
    n_pix, n_wn = flat.shape
    n_ref = ref_specs.shape[0]

    # Ensure non-negative starting S
    S = np.clip(ref_specs.copy(), 0.0, None).astype(float)
    # Normalize rows of S to unit max for stability
    if normalize_S:
        for k in range(n_ref):
            m = S[k].max()
            if m > 0:
                S[k] /= m

    C = np.zeros((n_pix, n_ref))
    prev_resid = np.inf
    actual_iter = 0
    for it in range(max_iter):
        # ---- Solve C: for each pixel x_i, solve S^T @ c_i ~= x_i with c_i >= 0 ----
        A_C = S.T  # (n_wn, n_ref)
        for i in range(n_pix):
            try:
                C[i], _ = nnls(A_C, flat[i])
            except Exception:
                pass
        # ---- Solve S: for each wn j, solve C @ s_j ~= x_:,j with s_j >= 0 ----
        for j in range(n_wn):
            try:
                S[:, j], _ = nnls(C, flat[:, j])
            except Exception:
                pass
        # ---- Normalize rows of S; absorb scale into C ----
        if normalize_S:
            for k in range(n_ref):
                m = S[k].max()
                if m > 0:
                    S[k] /= m
                    C[:, k] *= m
        # ---- Residual ----
        recon = C @ S
        resid = float(np.linalg.norm(flat - recon))
        actual_iter = it + 1
        rel = abs(prev_resid - resid) / (prev_resid + 1e-12)
        if report:
            print(f"  MCR-ALS iter {actual_iter:2d}/{max_iter}  "
                  f"resid={resid:.4g}  rel_change={rel:.2e}", end="\r")
        if rel < tol:
            break
        prev_resid = resid
    if report:
        print()

    C_map = C.T.reshape(n_ref, ny, nx)
    s = C.sum(axis=1, keepdims=True)
    s[s <= 0] = 1.0
    C_contrib = (C / s).T.reshape(n_ref, ny, nx)
    return C_map, C_contrib, S, resid, actual_iter


# =============================================================================
# Tk launcher dialog
# =============================================================================

def _ask_save_selection(current_metric_name):
    """Pop a small Tk dialog with checkboxes letting the user pick which
    output groups to save. Returns a selection dict (or None if cancelled).

    Categories (all default ON for Essential preset = current metric +
    combined RGB + mean spectrum + summary CSVs; All preset = everything).
    """
    # Important: matplotlib's TkAgg backend already runs a Tk mainloop while
    # the figure is showing. Creating a NEW Tk() and calling its mainloop()
    # nests two mainloops, and on Windows that leaves the inner dialog
    # unresponsive (buttons "click" silently). Use Toplevel attached to the
    # existing root + wait_window() so events flow correctly.
    parent = tk._default_root
    own_root = False
    if parent is None:
        parent = tk.Tk()
        parent.withdraw()
        own_root = True

    root = tk.Toplevel(parent)
    root.title("Save selection")
    w, h = 620, 720
    x = (root.winfo_screenwidth() - w) // 2
    y = max(0, (root.winfo_screenheight() - h) // 2)
    root.geometry(f"{w}x{h}+{x}+{y}")
    root.minsize(540, 500)
    root.resizable(True, True)
    root.transient(parent)

    # Default = "Essential" preset
    flags = {
        "intensity":          tk.BooleanVar(value=True),
        "current_metric_only": tk.BooleanVar(value=True),
        "all_metrics":        tk.BooleanVar(value=False),
        "combined_rgb":       tk.BooleanVar(value=True),
        "mean_spectrum":      tk.BooleanVar(value=True),
        "mcr_spectra":        tk.BooleanVar(value=False),
        "histograms":         tk.BooleanVar(value=False),
        "confidence":         tk.BooleanVar(value=False),
        "agreement":          tk.BooleanVar(value=True),
        "validation":         tk.BooleanVar(value=True),
        "per_metric_csvs":    tk.BooleanVar(value=False),
        "summary_csvs":       tk.BooleanVar(value=True),
        "raw_versions":       tk.BooleanVar(value=False),
    }
    result = {"selection": None}

    def apply_preset(preset):
        if preset == "essential":
            vals = {
                "intensity": True, "current_metric_only": True,
                "all_metrics": False, "combined_rgb": True,
                "mean_spectrum": True, "mcr_spectra": False,
                "histograms": False, "confidence": False,
                "agreement": True, "validation": True,
                "per_metric_csvs": False, "summary_csvs": True,
                "raw_versions": False,
            }
        elif preset == "all":
            vals = {k: True for k in flags}
            vals["current_metric_only"] = False
        elif preset == "current_only":
            vals = {k: False for k in flags}
            vals["intensity"] = True
            vals["current_metric_only"] = True
            vals["combined_rgb"] = True
            vals["summary_csvs"] = True
        else:
            return
        for k, v in vals.items():
            flags[k].set(v)

    # ---- UI ----
    pad = {"padx": 8, "pady": 2}

    # Buttons FIRST (packed to the bottom). Pinning them this way guarantees
    # Save / Cancel stay visible even if the section frames overflow the
    # initial window height.
    btn_frame = ttk.Frame(root)
    btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

    presets = ttk.LabelFrame(root, text="Presets", padding=6)
    presets.pack(fill=tk.X, **pad)
    ttk.Button(presets, text="Essential (current metric + summary)",
               command=lambda: apply_preset("essential")).pack(side=tk.LEFT,
                                                                padx=4)
    ttk.Button(presets, text="Current metric only",
               command=lambda: apply_preset("current_only")).pack(side=tk.LEFT,
                                                                   padx=4)
    ttk.Button(presets, text="All",
               command=lambda: apply_preset("all")).pack(side=tk.LEFT, padx=4)

    maps_frame = ttk.LabelFrame(root, text="Maps (per-ref PNGs)", padding=6)
    maps_frame.pack(fill=tk.X, **pad)
    ttk.Checkbutton(maps_frame, text="Intensity @ selected WN",
                    variable=flags["intensity"]).pack(anchor="w")
    ttk.Checkbutton(
        maps_frame,
        text=f"Only the metric currently displayed in the viewer "
             f"({current_metric_name})",
        variable=flags["current_metric_only"],
    ).pack(anchor="w")
    ttk.Checkbutton(maps_frame, text="All 9 metric maps "
                    "(NNLS norm, MCR contrib, CLS norm, Pearson prob/corr, "
                    "Cosine, NNLS raw, CLS raw, MCR conc)",
                    variable=flags["all_metrics"]).pack(anchor="w")

    rgb_frame = ttk.LabelFrame(root, text="Combined RGB / Spectra", padding=6)
    rgb_frame.pack(fill=tk.X, **pad)
    ttk.Checkbutton(rgb_frame,
                    text="Combined RGB (NNLS / MCR / CLS, 3 PNGs)",
                    variable=flags["combined_rgb"]).pack(anchor="w")
    ttk.Checkbutton(rgb_frame, text="Mean spectrum overlay",
                    variable=flags["mean_spectrum"]).pack(anchor="w")
    ttk.Checkbutton(rgb_frame, text="MCR-ALS resolved spectra plot",
                    variable=flags["mcr_spectra"]).pack(anchor="w")
    ttk.Checkbutton(rgb_frame, text="Histograms vs each reference",
                    variable=flags["histograms"]).pack(anchor="w")

    rel_frame = ttk.LabelFrame(root, text="Reliability", padding=6)
    rel_frame.pack(fill=tk.X, **pad)
    ttk.Checkbutton(rel_frame,
                    text="Confidence gap + entropy maps (NNLS + MCR)",
                    variable=flags["confidence"]).pack(anchor="w")
    ttk.Checkbutton(rel_frame,
                    text="Argmax + method-agreement + consensus maps",
                    variable=flags["agreement"]).pack(anchor="w")
    ttk.Checkbutton(rel_frame, text="Classifier validation "
                    "(K-fold CV confusion + synth scatter)",
                    variable=flags["validation"]).pack(anchor="w")

    csv_frame = ttk.LabelFrame(root, text="CSVs and raw variants", padding=6)
    csv_frame.pack(fill=tk.X, **pad)
    ttk.Checkbutton(csv_frame,
                    text="Per-metric pixel CSVs (5 metrics + intensity)",
                    variable=flags["per_metric_csvs"]).pack(anchor="w")
    ttk.Checkbutton(csv_frame, text="Summary CSVs "
                    "(statistics_summary + reliability_summary)",
                    variable=flags["summary_csvs"]).pack(anchor="w")
    ttk.Checkbutton(csv_frame, text="Raw (no-axis) PNG variants "
                    "for image maps (suffix _raw.png)",
                    variable=flags["raw_versions"]).pack(anchor="w")

    # Button handlers (btn_frame was created earlier and pinned to bottom)
    def on_ok():
        result["selection"] = {
            k: v.get() for k, v in flags.items()
        }
        result["selection"]["current_metric_name"] = current_metric_name
        root.destroy()

    def on_cancel():
        result["selection"] = None
        root.destroy()

    ttk.Button(btn_frame, text="  Save  ", command=on_ok).pack(
        side=tk.RIGHT, padx=16)
    ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(
        side=tk.RIGHT, padx=8)
    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.grab_set()        # modal
    root.focus_set()
    root.wait_window()     # block here until the dialog is destroyed
    if own_root:
        try:
            parent.destroy()
        except Exception:
            pass
    return result["selection"]


class LauncherDialog:
    """Dialog: select refs + test, baseline checkboxes, params, output folder."""

    def __init__(self):
        self.result = None
        self.ref_files = []
        self.test_file = None

        self.root = tk.Tk()
        self.root.title("SERS Compound Discriminator")
        w, h = 760, 660
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # ---- Reference files ----
        ref_frame = ttk.LabelFrame(
            self.root, text="1. Reference CSV(s)  (one mean spectrum per file)",
            padding=8,
        )
        ref_frame.pack(fill=tk.BOTH, expand=False, **pad)

        list_row = ttk.Frame(ref_frame)
        list_row.pack(fill=tk.BOTH, expand=True)
        self.ref_listbox = tk.Listbox(list_row, height=5, selectmode=tk.EXTENDED)
        self.ref_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scr = ttk.Scrollbar(list_row, orient="vertical",
                            command=self.ref_listbox.yview)
        self.ref_listbox.configure(yscrollcommand=scr.set)
        scr.pack(side=tk.LEFT, fill=tk.Y)

        btn_col = ttk.Frame(ref_frame)
        btn_col.pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_col, text="Add...", command=self._add_refs).pack(fill=tk.X, pady=2)
        ttk.Button(btn_col, text="Remove", command=self._remove_refs).pack(fill=tk.X, pady=2)
        ttk.Button(btn_col, text="Clear", command=self._clear_refs).pack(fill=tk.X, pady=2)

        # ---- Test file ----
        test_frame = ttk.LabelFrame(
            self.root, text="2. Testing SERS mapping CSV",
            padding=8,
        )
        test_frame.pack(fill=tk.X, **pad)
        self.test_var = tk.StringVar(value="(no file)")
        ttk.Label(test_frame, textvariable=self.test_var, width=70).pack(
            side=tk.LEFT, padx=4)
        ttk.Button(test_frame, text="Browse...", command=self._browse_test).pack(
            side=tk.LEFT)

        # ---- Baseline ----
        bl_frame = ttk.LabelFrame(self.root, text="3. Baseline correction (arPLS)",
                                  padding=8)
        bl_frame.pack(fill=tk.X, **pad)

        self.bl_refs_var = tk.BooleanVar(value=True)
        self.bl_test_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bl_frame, text="Apply arPLS to reference files",
                        variable=self.bl_refs_var).grid(row=0, column=0, columnspan=2,
                                                         sticky="w", padx=4)
        ttk.Checkbutton(bl_frame, text="Apply arPLS to testing file",
                        variable=self.bl_test_var).grid(row=1, column=0, columnspan=2,
                                                         sticky="w", padx=4)

        ttk.Label(bl_frame, text="Lambda:").grid(row=2, column=0, sticky="e", padx=4)
        self.lam_var = tk.StringVar(value=f"{DEFAULT_LAMBDA:.0e}")
        ttk.Entry(bl_frame, textvariable=self.lam_var, width=12).grid(
            row=2, column=1, sticky="w")
        ttk.Label(bl_frame, text="Ratio:").grid(row=2, column=2, sticky="e", padx=4)
        self.ratio_var = tk.StringVar(value=f"{DEFAULT_RATIO:.0e}")
        ttk.Entry(bl_frame, textvariable=self.ratio_var, width=12).grid(
            row=2, column=3, sticky="w")
        ttk.Label(bl_frame, text="Max iter:").grid(row=2, column=4, sticky="e", padx=4)
        self.iter_var = tk.StringVar(value=str(DEFAULT_MAX_ITER))
        ttk.Entry(bl_frame, textvariable=self.iter_var, width=8).grid(
            row=2, column=5, sticky="w")

        # ---- MCR-ALS ----
        mcr_frame = ttk.LabelFrame(
            self.root, text="4. MCR-ALS settings", padding=8,
        )
        mcr_frame.pack(fill=tk.X, **pad)

        self.mcr_run_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(mcr_frame, text="Run MCR-ALS (slow for large maps)",
                        variable=self.mcr_run_var).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=4)
        ttk.Label(mcr_frame, text="Max iterations:").grid(
            row=0, column=2, sticky="e", padx=4)
        self.mcr_iter_var = tk.StringVar(value="12")
        ttk.Entry(mcr_frame, textvariable=self.mcr_iter_var, width=6).grid(
            row=0, column=3, sticky="w")
        ttk.Label(mcr_frame, text="Tolerance:").grid(
            row=0, column=4, sticky="e", padx=4)
        self.mcr_tol_var = tk.StringVar(value="1e-5")
        ttk.Entry(mcr_frame, textvariable=self.mcr_tol_var, width=8).grid(
            row=0, column=5, sticky="w")

        # ---- WN crop ----
        crop_frame = ttk.LabelFrame(
            self.root, text="5. Wavenumber range (optional crop)",
            padding=8,
        )
        crop_frame.pack(fill=tk.X, **pad)
        ttk.Label(crop_frame, text="Min:").pack(side=tk.LEFT, padx=4)
        self.wn_min_var = tk.StringVar(value="")
        ttk.Entry(crop_frame, textvariable=self.wn_min_var, width=10).pack(side=tk.LEFT)
        ttk.Label(crop_frame, text="Max:").pack(side=tk.LEFT, padx=(12, 4))
        self.wn_max_var = tk.StringVar(value="")
        ttk.Entry(crop_frame, textvariable=self.wn_max_var, width=10).pack(side=tk.LEFT)
        ttk.Label(crop_frame,
                  text="cm-1  (blank = use full overlapping range)").pack(
            side=tk.LEFT, padx=8)

        # ---- Output ----
        out_frame = ttk.LabelFrame(self.root, text="6. Output folder", padding=8)
        out_frame.pack(fill=tk.X, **pad)
        self.out_var = tk.StringVar(value="")
        ttk.Entry(out_frame, textvariable=self.out_var, width=60).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(out_frame, text="Browse...",
                   command=self._browse_output).pack(side=tk.LEFT)

        # ---- Run / Cancel ----
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill=tk.X, **pad)
        ttk.Button(btn_frame, text="  Run  ", command=self._on_run).pack(
            side=tk.LEFT, padx=12)
        ttk.Button(btn_frame, text="Cancel", command=self._on_cancel).pack(
            side=tk.LEFT, padx=12)

    def _add_refs(self):
        paths = filedialog.askopenfilenames(
            title="Select reference SERS CSV(s)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        for p in paths:
            if p not in self.ref_files:
                self.ref_files.append(p)
                self.ref_listbox.insert(tk.END, os.path.basename(p))
        if self.ref_files and not self.out_var.get().strip():
            self.out_var.set(os.path.join(
                os.path.dirname(self.ref_files[0]), "discriminator_output"))

    def _remove_refs(self):
        for idx in reversed(list(self.ref_listbox.curselection())):
            self.ref_listbox.delete(idx)
            del self.ref_files[idx]

    def _clear_refs(self):
        self.ref_listbox.delete(0, tk.END)
        self.ref_files.clear()

    def _browse_test(self):
        p = filedialog.askopenfilename(
            title="Select testing SERS mapping CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p:
            self.test_file = p
            self.test_var.set(os.path.basename(p))
            if not self.out_var.get().strip():
                self.out_var.set(os.path.join(
                    os.path.dirname(p), "discriminator_output"))

    def _browse_output(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.out_var.set(p)

    def _on_run(self):
        if not self.ref_files:
            messagebox.showerror("Error", "Add at least one reference CSV.")
            return
        if not self.test_file:
            messagebox.showerror("Error", "Select a testing CSV.")
            return
        for fp in self.ref_files + [self.test_file]:
            if not os.path.isfile(fp):
                messagebox.showerror("Error", f"File not found:\n{fp}")
                return
        try:
            lam = float(self.lam_var.get())
            ratio = float(self.ratio_var.get())
            max_iter = int(self.iter_var.get())
            if lam <= 0 or ratio <= 0 or max_iter <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error",
                                 "Lambda / ratio must be positive numbers; "
                                 "Max iter a positive integer.")
            return

        wn_min = wn_max = None
        try:
            if self.wn_min_var.get().strip():
                wn_min = float(self.wn_min_var.get())
            if self.wn_max_var.get().strip():
                wn_max = float(self.wn_max_var.get())
        except ValueError:
            messagebox.showerror("Error", "Wavenumber crop values must be numbers.")
            return
        if wn_min is not None and wn_max is not None and wn_min >= wn_max:
            messagebox.showerror("Error", "WN min must be less than WN max.")
            return

        out_dir = self.out_var.get().strip()
        if not out_dir:
            out_dir = os.path.join(os.path.dirname(self.test_file),
                                   "discriminator_output")

        try:
            mcr_max_iter = int(self.mcr_iter_var.get())
            mcr_tol = float(self.mcr_tol_var.get())
            if mcr_max_iter < 1 or mcr_tol <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error",
                                 "MCR-ALS Max iterations must be a positive "
                                 "integer; Tolerance must be a positive number.")
            return

        self.result = {
            "ref_files": list(self.ref_files),
            "test_file": self.test_file,
            "baseline_refs": bool(self.bl_refs_var.get()),
            "baseline_test": bool(self.bl_test_var.get()),
            "lam": lam,
            "ratio": ratio,
            "max_iter": max_iter,
            "wn_min": wn_min,
            "wn_max": wn_max,
            "out_dir": out_dir,
            "run_mcr": bool(self.mcr_run_var.get()),
            "mcr_max_iter": mcr_max_iter,
            "mcr_tol": mcr_tol,
        }
        self.root.destroy()

    def _on_cancel(self):
        self.result = None
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.result


# =============================================================================
# Processing pipeline
# =============================================================================

def process(config):
    """Load refs + test, baseline-correct, align, compute discriminations.

    Returns a dict with everything the viewer needs.
    """
    ref_paths = [Path(p) for p in config["ref_files"]]
    test_path = Path(config["test_file"])
    lam = config["lam"]
    ratio = config["ratio"]
    max_iter = config["max_iter"]
    wn_min = config["wn_min"]
    wn_max = config["wn_max"]

    print(f"\n{'='*60}\nSERS Compound Discriminator\n{'='*60}")

    # ---- Load references ----
    print(f"\nLoading {len(ref_paths)} reference file(s)...")
    refs = []
    for p in ref_paths:
        d = parse_csv_auto(p)
        mean_spec = reference_mean_spectrum(d["cube"])
        refs.append({
            "path": p,
            "name": p.stem,
            "wn": d["wavenumbers"],
            "mean": mean_spec,
            "cube": d["cube"],  # kept for K-fold CV
        })
        print(f"  {p.name}: {len(d['wavenumbers'])} pts "
              f"({d['wavenumbers'][0]:.1f}-{d['wavenumbers'][-1]:.1f} {CM1}), "
              f"{d['cube'].shape[0]}x{d['cube'].shape[1]} pixels")

    # ---- Load test ----
    print(f"\nLoading test file: {test_path.name}")
    test = parse_csv_auto(test_path)
    ny, nx, n_wn_t = test["cube"].shape
    n_pix = ny * nx
    print(f"  Auto-detected map: {nx} x {ny} = {n_pix} pixels, "
          f"{n_wn_t} wavenumber points "
          f"({test['wavenumbers'][0]:.1f}-{test['wavenumbers'][-1]:.1f} {CM1})")
    if n_pix >= 2000:
        print(f"  NOTE: large map ({n_pix} pixels). MCR-ALS may take a while; "
              "consider lowering max iterations or unchecking 'Run MCR-ALS'.")

    # ---- Determine common (overlapping) wn axis. Use test axis as the target ----
    target_wn = test["wavenumbers"].copy()
    lo = max(target_wn.min(), max(r["wn"].min() for r in refs))
    hi = min(target_wn.max(), min(r["wn"].max() for r in refs))
    if wn_min is not None:
        lo = max(lo, wn_min)
    if wn_max is not None:
        hi = min(hi, wn_max)
    if hi <= lo:
        raise ValueError("No overlap between reference and test wavenumber ranges "
                         "after applying crop.")
    mask = (target_wn >= lo) & (target_wn <= hi)
    common_wn = target_wn[mask]
    print(f"\nCommon wavenumber range: {common_wn[0]:.1f}-{common_wn[-1]:.1f} "
          f"{CM1}  ({len(common_wn)} points)")

    # ---- Crop test cube to common wn ----
    test_cube = test["cube"][:, :, mask]

    # ---- Baseline-correct test cube ----
    if config["baseline_test"]:
        print("\nApplying arPLS to test cube (per pixel)...")
        test_cube = baseline_correct_cube(test_cube, lam, ratio, max_iter, label="test")
    else:
        print("\nSkipping baseline correction for test (already corrected).")

    # ---- Align refs onto common_wn, baseline-correct each ----
    ref_specs = []
    ref_top_peaks = []
    ref_cubes_aligned = []  # for K-fold CV
    for r in refs:
        aligned = align_to_axis(r["wn"], r["mean"], common_wn)
        if config["baseline_refs"]:
            aligned = baseline_correct_spectrum(aligned, lam, ratio, max_iter)
        # Top peaks
        peaks = detect_top_peaks(common_wn, aligned, n_peaks=DEFAULT_TOP_PEAKS)
        ref_specs.append(aligned)
        ref_top_peaks.append(peaks)
        print(f"  [{r['name']}] top peaks: " +
              ", ".join(f"{w:.1f}" for w, _ in peaks))

        # Build pixel-wise aligned cube for this ref (each pixel's spectrum
        # interpolated onto common_wn; not baseline-corrected per pixel since
        # that'd be slow + reference data is generally clean enough that the
        # mean-spectrum correction we already did is representative).
        cube = r["cube"]
        ry, rx, rwn = cube.shape
        flat_orig = cube.reshape(-1, rwn)
        flat_aligned = np.array([align_to_axis(r["wn"], s, common_wn)
                                 for s in flat_orig])
        ref_cubes_aligned.append(flat_aligned.reshape(ry, rx, len(common_wn)))

    ref_specs = np.array(ref_specs)
    if config["baseline_refs"]:
        print("Applied arPLS to each reference's mean spectrum.")
    else:
        print("Skipped baseline correction for references.")

    # ---- Compute discrimination metrics ----
    print("\nComputing Pearson correlation + softmax probabilities...")
    prob_maps, corr_maps = pearson_softmax_map(test_cube, ref_specs)
    print("Computing cosine similarity maps...")
    cos_maps = cosine_map(test_cube, ref_specs)
    print("Computing NNLS unmixing maps...")
    nnls_raw, nnls_norm = nnls_map(test_cube, ref_specs)
    print("Computing CLS unmixing maps...")
    cls_raw, cls_norm, cls_neg_pct = cls_map(test_cube, ref_specs)
    if config.get("run_mcr", True):
        if n_pix >= 3000:
            print(f"  WARNING: large map ({n_pix} pixels) + MCR-ALS will be "
                  "slow (~tens of seconds to minutes). Consider unchecking "
                  "'Run MCR-ALS' in the launcher to skip it.")
        print("Running MCR-ALS...")
        mcr_conc, mcr_contrib, mcr_S, mcr_resid, mcr_iter = mcr_als(
            test_cube, ref_specs,
            max_iter=config.get("mcr_max_iter", 12),
            tol=config.get("mcr_tol", 1e-5),
        )
        print(f"  MCR-ALS converged in {mcr_iter} iterations  "
              f"(resid={mcr_resid:.4g})")
    else:
        print("Skipping MCR-ALS (disabled). Falling back to NNLS results.")
        mcr_conc = nnls_raw.copy()
        mcr_contrib = nnls_norm.copy()
        mcr_S = ref_specs.copy()
        mcr_resid = float("nan")
        mcr_iter = 0

    # ---- Classifier validation on REFERENCE data (independent of test) ----
    print("\nValidating classifier on reference data...")
    cv_k = int(config.get("cv_k", 5))
    cv_result = kfold_cv_references(ref_cubes_aligned,
                                    [r["name"] for r in refs], k=cv_k)
    if cv_result is not None:
        print(f"  K-fold ({cv_k}) cross-validation:")
        print(f"    accuracy   = {cv_result['accuracy']:.4f}")
        print(f"    macro F1   = {cv_result['macro_f1']:.4f}")
        for i, name in enumerate(cv_result["ref_names"]):
            print(f"    {name:>30s}  prec={cv_result['precision'][i]:.3f}  "
                  f"rec={cv_result['recall'][i]:.3f}  "
                  f"F1={cv_result['f1'][i]:.3f}")

    print("Running synthetic mixture benchmark (5% noise baseline)...")
    synth_result = synthetic_mixture_benchmark(
        ref_specs,
        n_synth=int(config.get("synth_n", 300)),
        noise_level=float(config.get("synth_noise", 0.05)),
    )
    print(f"  Synthetic (N={synth_result['n_synth']}, "
          f"noise={synth_result['noise_level']*100:.0f}%): "
          f"overall NNLS RMSE = {synth_result['overall_rmse']:.4f}  "
          "(note: ideal linear mixture - this is the upper bound)")
    for i, name in enumerate([r["name"] for r in refs]):
        print(f"    {name:>30s}  RMSE={synth_result['rmse_per_ref'][i]:.3f}  "
              f"r={synth_result['correlation_per_ref'][i]:.3f}")

    print("Running synthetic NOISE SWEEP (1% -> 50%) — the realistic stress "
          "test...")
    noise_sweep_result = synthetic_noise_sweep(
        ref_specs,
        n_per_level=int(config.get("sweep_n", 150)),
    )
    ref_names_disp = [r["name"] for r in refs]
    for i, noise in enumerate(noise_sweep_result["noise_levels"]):
        rmse = noise_sweep_result["overall_rmse"][i]
        per_ref = "  ".join(
            f"{nm}={noise_sweep_result['rmse_per_ref'][i, j]:.3f}"
            for j, nm in enumerate(ref_names_disp)
        )
        print(f"  noise={noise*100:>3.0f}%  overall RMSE={rmse:.4f}   "
              f"per-ref: {per_ref}")

    # ---- Reliability metrics ----
    print("\nComputing reliability metrics (confidence + agreement)...")
    gap_nnls, ent_nnls = confidence_maps(nnls_norm)
    gap_mcr,  ent_mcr  = confidence_maps(mcr_contrib)
    argmaxes, agreement, consensus = argmax_and_agreement(
        nnls_norm, mcr_contrib, cls_norm)
    argmax_nnls, argmax_mcr, argmax_cls = argmaxes

    # Summary stats: % pixels with full / partial / no-majority agreement
    n_pix_arr = consensus.size
    n_full   = int((agreement == 3).sum())
    n_two    = int((agreement == 2).sum())
    n_split  = int((agreement == 1).sum())
    print(f"  Method agreement (NNLS / MCR / CLS):")
    print(f"    3/3 agree:  {n_full:>6d} pixels ({100*n_full/n_pix_arr:.1f}%)")
    print(f"    2/3 agree:  {n_two:>6d} pixels ({100*n_two/n_pix_arr:.1f}%)")
    print(f"    no majority:{n_split:>6d} pixels ({100*n_split/n_pix_arr:.1f}%)")
    print(f"  Mean NNLS confidence gap: {float(gap_nnls.mean()):.3f}  "
          f"(0 = ambiguous, 1 = perfectly confident)")
    print(f"  Mean NNLS entropy:        {float(ent_nnls.mean()):.3f} bits  "
          f"(0 = certain, log2(n_ref) = uniform)")

    return {
        "test_path": test_path,
        "test_cube": test_cube,
        "x_coords": test["x_coords"],
        "y_coords": test["y_coords"],
        "wn": common_wn,
        "refs": refs,            # original ref info (name, etc.)
        "ref_specs": ref_specs,  # aligned, possibly baseline-corrected
        "ref_top_peaks": ref_top_peaks,
        "prob_maps": prob_maps,
        "corr_maps": corr_maps,
        "cos_maps": cos_maps,
        "nnls_raw": nnls_raw,
        "nnls_norm": nnls_norm,
        "cls_raw": cls_raw,
        "cls_norm": cls_norm,
        "cls_neg_pct": cls_neg_pct,
        "mcr_conc": mcr_conc,
        "mcr_contrib": mcr_contrib,
        "mcr_S": mcr_S,
        "mcr_resid": mcr_resid,
        "mcr_iter": mcr_iter,
        # Reliability metrics
        "confidence_gap_nnls": gap_nnls,
        "confidence_gap_mcr": gap_mcr,
        "entropy_nnls": ent_nnls,
        "entropy_mcr": ent_mcr,
        "argmax_nnls": argmax_nnls,
        "argmax_mcr": argmax_mcr,
        "argmax_cls": argmax_cls,
        "agreement": agreement,
        "consensus": consensus,
        "agreement_summary": {
            "full": n_full, "two": n_two, "split": n_split,
            "n_pix": n_pix_arr,
        },
        # Classifier validation (computed from reference data, independent
        # of the test sample)
        "cv_result": cv_result,
        "synth_result": synth_result,
        "noise_sweep_result": noise_sweep_result,
        "out_dir": Path(config["out_dir"]),
    }


# =============================================================================
# Interactive viewer
# =============================================================================

def show_viewer(result):
    refs = result["refs"]
    n_ref = len(refs)
    cube = result["test_cube"]
    wn = result["wn"]
    x_coords = result["x_coords"]
    y_coords = result["y_coords"]
    extent = [x_coords.min(), x_coords.max(),
              y_coords.min(), y_coords.max()]
    ny, nx, n_wn = cube.shape
    mean_test = cube.reshape(-1, n_wn).mean(axis=0)

    # Cmap + tint for each ref
    cmaps = [REF_CMAPS[i % len(REF_CMAPS)] for i in range(n_ref)]
    tints = [REF_TINTS[i % len(REF_TINTS)] for i in range(n_ref)]

    metrics = [
        "NNLS norm", "MCR contrib", "CLS norm",
        "Pearson prob", "Cosine sim",
        "Pearson corr", "NNLS raw", "MCR conc", "CLS raw",
    ]
    current_metric = {"name": "NNLS norm"}
    # selected wn per reference (start at its top peak)
    selected_wn = [peaks[0][0] for peaks in result["ref_top_peaks"]]

    # ---- Figure layout ----
    # Choose a per-map axes width and derive height from the test map's
    # aspect ratio so rectangular maps don't squish or stretch.
    map_aspect = ny / max(nx, 1)  # height-over-width
    per_map_w_in = 3.2
    per_map_h_in = max(per_map_w_in * map_aspect, 1.6)
    # Overall figure size: row of n_ref maps wide + side padding for radios,
    # plus rows for spectrum (top), 2 map rows, RGB preview (bottom).
    fig_w = max(per_map_w_in * n_ref + 3.5, 11)
    fig_h = (per_map_h_in * 2.0 +    # 2 map rows
             3.0 +                    # spectrum
             per_map_h_in * 0.9 +     # RGB preview
             1.8)                     # padding + buttons + radios
    fig_h = max(fig_h, 9.5)
    fig = plt.figure(figsize=(fig_w, fig_h))
    try:
        fig.canvas.manager.set_window_title(
            f"SERS Discriminator - {result['test_path'].name}")
    except Exception:
        pass
    fig.suptitle(
        f"SERS Compound Discriminator - {result['test_path'].name}   "
        f"[{BUILD_TAG}]",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # Top: mean spectrum (will toggle to a single-pixel spectrum on map clicks)
    spec_ax = fig.add_axes([0.06, 0.78, 0.90, 0.15])
    test_line, = spec_ax.plot(wn, mean_test, color="#222", lw=1.2,
                               label="Test mean")
    # Track what the test line currently shows so we can revert + tag titles
    spec_state = {"mode": "mean", "pixel": None}
    spec_overlay_lines = []  # keep refs so cmap changes can recolor
    for i, (r, spec) in enumerate(zip(refs, result["ref_specs"])):
        # normalize ref for overlay
        scale = (mean_test.max() / (spec.max() + 1e-12)) * 0.9
        ln, = spec_ax.plot(wn, spec * scale, color=tints[i], lw=0.9, alpha=0.8,
                            label=f"{r['name']} (scaled)")
        spec_overlay_lines.append(ln)
    spec_ax.set_xlabel(f"Wavenumber ({CM1})", fontsize=9)
    spec_ax.set_ylabel("Intensity", fontsize=9)
    spec_ax.set_title("Mean test spectrum vs. reference spectra (scaled)",
                      fontsize=10)
    spec_ax.grid(alpha=0.3)
    spec_ax.legend(loc="upper right", fontsize=8, ncol=min(n_ref + 1, 4))
    spec_ax.tick_params(labelsize=8)
    spec_ax.set_xlim(wn[0], wn[-1])

    # vertical lines for currently selected wn per ref
    sel_lines = []
    for i in range(n_ref):
        ln = spec_ax.axvline(selected_wn[i], color=tints[i], lw=1.4, alpha=0.7)
        sel_lines.append(ln)

    # Row 1: intensity maps. Each axes-box has a width fixed by n_ref. We
    # tweak the *height* in figure coords to roughly match the data aspect
    # ratio (clamped) so very rectangular maps don't leave huge whitespace.
    # The actual pixel rendering still uses aspect="equal".
    map_w = 0.72 / n_ref
    map_h_data = map_w * (fig_w / fig_h) * (ny / max(nx, 1))
    map_h = max(min(map_h_data, 0.22), 0.12)
    row1_y = 0.53
    row2_y = 0.26
    start_x = (1.0 - (map_w * n_ref + 0.02 * (n_ref - 1))) / 2

    # Default RGB channel assignment (numeric: 1=R, 2=G, 3=B, 0=off)
    rgb_default = ["1", "2", "3"] + ["0"] * max(0, n_ref - 3)
    rgb_assignment = list(rgb_default[:n_ref])

    intensity_axes = []
    intensity_ims = []
    intensity_cbars = []
    wn_textboxes = []
    ch_buttons = []         # channel popup-menu buttons (one per ref)
    cmap_textboxes = []
    color_swatch_btns = []  # paint-palette buttons (one per ref)
    vmin_textboxes = []     # intensity scale min per ref
    vmax_textboxes = []     # intensity scale max per ref
    # Make cmaps + tints mutable lists so user can swap per ref
    cmaps = list(cmaps)
    tints = list(tints)

    for i in range(n_ref):
        ax_x = start_x + i * (map_w + 0.02)
        ax = fig.add_axes([ax_x, row1_y, map_w, map_h])
        # intensity at currently selected wn
        idx = int(np.argmin(np.abs(wn - selected_wn[i])))
        img = cube[:, :, idx]
        im = ax.imshow(img, cmap=_resolve_cmap(cmaps[i]),
                       extent=extent, origin="upper",
                       aspect="equal", interpolation="nearest")
        ax.set_title(f"Intensity: {refs[i]['name']} @ {wn[idx]:.1f} {CM1}",
                     fontsize=11, fontweight="bold", color=tints[i])
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        ax.tick_params(labelsize=9)
        cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        intensity_axes.append(ax)
        intensity_ims.append(im)
        intensity_cbars.append(cbar)

        # Row 2 (tb_y): wn + ch + cmap textbox + paint-palette swatch button
        tb_y = row1_y - 0.045
        ax_tb = fig.add_axes([ax_x + map_w * 0.03, tb_y,
                              map_w * 0.27, 0.025])
        tb = TextBox(ax_tb, "wn ", initial=f"{wn[idx]:.1f}",
                     color="#ffffff", hovercolor="#f5f5f5")
        tb.label.set_fontsize(8)
        wn_textboxes.append(tb)

        ax_ch = fig.add_axes([ax_x + map_w * 0.34, tb_y,
                              map_w * 0.14, 0.025])
        btn_ch = Button(ax_ch,
                        f"ch {_ch_to_display(rgb_assignment[i])}",
                        color="#f5f5f5", hovercolor="#e2e2e2")
        btn_ch.label.set_fontsize(8)
        ch_buttons.append(btn_ch)

        ax_cm = fig.add_axes([ax_x + map_w * 0.52, tb_y,
                              map_w * 0.34, 0.025])
        tb_cm = TextBox(ax_cm, "cmap ", initial=str(cmaps[i]),
                        color="#ffffff", hovercolor="#f5f5f5")
        tb_cm.label.set_fontsize(8)
        cmap_textboxes.append(tb_cm)

        # Paint-palette swatch button — click to open the standard Windows
        # color picker. Width is small (just a colored chip).
        ax_sw = fig.add_axes([ax_x + map_w * 0.88, tb_y,
                              map_w * 0.10, 0.025])
        btn_sw = Button(ax_sw, "", color=tints[i], hovercolor=tints[i])
        # Border to make the swatch visible against the page
        for spine in ax_sw.spines.values():
            spine.set_edgecolor("#666")
            spine.set_linewidth(1.0)
        color_swatch_btns.append(btn_sw)

        # Intensity scale row: vmin / vmax textboxes for this map. Default to
        # auto data range; user can type any number to clip the displayed
        # range.
        sc_y = row1_y - 0.078
        cur_vmin = float(img.min())
        cur_vmax = float(img.max() + 1e-9)
        im.set_clim(cur_vmin, cur_vmax)

        ax_vmin = fig.add_axes([ax_x + map_w * 0.04, sc_y,
                                map_w * 0.30, 0.025])
        tb_vmin = TextBox(ax_vmin, "min ", initial=f"{cur_vmin:.1f}",
                          color="#ffffff", hovercolor="#f5f5f5")
        tb_vmin.label.set_fontsize(8)
        vmin_textboxes.append(tb_vmin)

        ax_vmax = fig.add_axes([ax_x + map_w * 0.52, sc_y,
                                map_w * 0.30, 0.025])
        tb_vmax = TextBox(ax_vmax, "max ", initial=f"{cur_vmax:.1f}",
                          color="#ffffff", hovercolor="#f5f5f5")
        tb_vmax.label.set_fontsize(8)
        vmax_textboxes.append(tb_vmax)

        # Peak hint row (just informational, small text)
        peak_y = row1_y - 0.108
        ax_lbl = fig.add_axes([ax_x, peak_y, map_w, 0.018])
        ax_lbl.axis("off")
        peaks = result["ref_top_peaks"][i]
        peak_text = "peaks: " + " ".join(f"{w:.0f}" for w, _ in peaks)
        ax_lbl.text(0.5, 0.5, peak_text, transform=ax_lbl.transAxes,
                    fontsize=8, color=tints[i], ha="center", va="center")

    # Row 2: probability / metric maps
    prob_axes = []
    prob_ims = []
    prob_cbars = []
    for i in range(n_ref):
        ax_x = start_x + i * (map_w + 0.02)
        ax = fig.add_axes([ax_x, row2_y, map_w, map_h])
        im = ax.imshow(result["nnls_norm"][i], cmap=_resolve_cmap(cmaps[i]),
                       extent=extent, origin="upper",
                       aspect="equal", interpolation="nearest",
                       vmin=0.0, vmax=1.0)
        ax.set_title(f"{current_metric['name']}: {refs[i]['name']}",
                     fontsize=11, fontweight="bold", color=tints[i])
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=8)
        ax.tick_params(labelsize=7)
        cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        prob_axes.append(ax)
        prob_ims.append(im)
        prob_cbars.append(cbar)

    # Metric radio (left side)
    radio_ax = fig.add_axes([0.005, 0.16, 0.105, 0.26], facecolor="#f7f7f7")
    radio_ax.set_title("Metric", fontsize=10, fontweight="bold", pad=4)
    radio = RadioButtons(radio_ax, labels=tuple(metrics), active=0)
    for lbl in radio.labels:
        lbl.set_fontsize(8)

    def metric_arrays(name):
        if name == "Pearson prob":
            return result["prob_maps"], (0.0, 1.0), "probability"
        if name == "Cosine sim":
            return result["cos_maps"], (0.0, 1.0), "cosine"
        if name == "NNLS norm":
            return result["nnls_norm"], (0.0, 1.0), "contribution"
        if name == "CLS norm":
            return result["cls_norm"], (0.0, 1.0), "contribution"
        if name == "MCR contrib":
            return result["mcr_contrib"], (0.0, 1.0), "contribution"
        if name == "Pearson corr":
            return result["corr_maps"], (-1.0, 1.0), "Pearson r"
        if name == "NNLS raw":
            arr = result["nnls_raw"]
            return arr, (float(arr.min()), float(arr.max() + 1e-9)), "weight (raw)"
        if name == "CLS raw":
            arr = result["cls_raw"]
            return arr, (float(arr.min()), float(arr.max() + 1e-9)), "coeff (signed)"
        if name == "MCR conc":
            arr = result["mcr_conc"]
            return arr, (float(arr.min()), float(arr.max() + 1e-9)), "concentration"
        return result["nnls_norm"], (0.0, 1.0), "contribution"

    # Combined RGB preview (below metric maps)
    rgb_h = 0.18
    rgb_w = 0.30
    rgb_x = (1.0 - rgb_w) / 2
    rgb_y = 0.04
    rgb_ax = fig.add_axes([rgb_x, rgb_y, rgb_w, rgb_h])
    rgb_init = np.zeros((ny, nx, 3))
    rgb_im = rgb_ax.imshow(rgb_init, extent=extent, origin="upper",
                            aspect="equal", interpolation="nearest")
    rgb_ax.set_title("Combined RGB contribution (current metric)",
                     fontsize=9, fontweight="bold")
    rgb_ax.set_xlabel("X", fontsize=8)
    rgb_ax.set_ylabel("Y", fontsize=8)
    rgb_ax.tick_params(labelsize=7)
    # Legend (right of RGB axes)
    legend_text = []
    for i, r in enumerate(refs):
        legend_text.append(f"{rgb_assignment[i]}: {r['name']}")
    rgb_legend = rgb_ax.text(
        1.02, 0.98, "\n".join(legend_text),
        transform=rgb_ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    def build_rgb(arr_norm):
        """Build the combined RGB preview using the user's currently chosen
        tints (so the picker color shows up in the combined image)."""
        return _build_pastel_rgb(arr_norm, rgb_assignment, tints=tints)

    def _normalize_for_rgb(arr, vmin, vmax):
        """Scale arr to [0,1] using vmin/vmax."""
        span = max(vmax - vmin, 1e-12)
        out = (arr - vmin) / span
        return np.clip(out, 0.0, 1.0)

    def on_metric(name):
        current_metric["name"] = name
        arr, (vmin, vmax), label = metric_arrays(name)
        for i in range(n_ref):
            prob_ims[i].set_data(arr[i])
            prob_ims[i].set_clim(vmin, vmax)
            prob_axes[i].set_title(f"{name}: {refs[i]['name']}",
                                   fontsize=9, fontweight="bold", color=tints[i])
            prob_cbars[i].set_label(label, fontsize=8)
        # Update combined RGB
        arr_norm = _normalize_for_rgb(arr, vmin, vmax)
        rgb_im.set_data(build_rgb(arr_norm))
        rgb_ax.set_title(f"Combined RGB ({name})",
                         fontsize=9, fontweight="bold")
        fig.canvas.draw_idle()

    radio.on_clicked(on_metric)

    def update_rgb_legend():
        # Map numeric channel back to a human label for the legend
        ch_label = {"1": "R", "2": "G", "3": "B", "0": "-"}
        legend_text = []
        for i, r in enumerate(refs):
            disp = _ch_to_display(rgb_assignment[i])
            legend_text.append(f"{disp}({ch_label.get(disp, '-')}): {r['name']}")
        rgb_legend.set_text("\n".join(legend_text))

    def refresh_rgb_display():
        arr, (vmin, vmax), _ = metric_arrays(current_metric["name"])
        arr_norm = _normalize_for_rgb(arr, vmin, vmax)
        rgb_im.set_data(build_rgb(arr_norm))
        update_rgb_legend()
        fig.canvas.draw_idle()

    def _apply_channel(idx, value):
        """Set ref idx's channel and refresh RGB. value is '0'/'1'/'2'/'3'."""
        canon = _ch_to_display(value)
        rgb_assignment[idx] = canon
        ch_buttons[idx].label.set_text(f"ch {canon}")
        refresh_rgb_display()

    def _get_tk_root():
        """Get the matplotlib figure's Tk root. Falls back to tk._default_root,
        then to a hidden Tk()."""
        try:
            w = fig.canvas.get_tk_widget()
            return w.winfo_toplevel()
        except Exception:
            pass
        if tk._default_root is not None:
            return tk._default_root
        return None

    def _cycle_channel(idx):
        cycle = {"0": "1", "1": "2", "2": "3", "3": "0"}
        cur = _ch_to_display(rgb_assignment[idx])
        _apply_channel(idx, cycle[cur])

    def make_ch_popup(idx):
        def handler(_event):
            print(f"[ch button {idx + 1} clicked]")
            tk_root = _get_tk_root()
            if tk_root is None:
                _cycle_channel(idx)
                return
            try:
                menu = tk.Menu(tk_root, tearoff=0)
                for label, val in [
                    ("1  -  Red",   "1"),
                    ("2  -  Green", "2"),
                    ("3  -  Blue",  "3"),
                    ("0  -  off",   "0"),
                ]:
                    menu.add_command(
                        label=label,
                        command=lambda v=val, j=idx: _apply_channel(j, v),
                    )
                x = tk_root.winfo_pointerx()
                y = tk_root.winfo_pointery()
                # Do NOT call grab_release here — tk_popup is non-blocking,
                # and releasing the grab immediately would close the menu
                # before the user can pick anything.
                menu.tk_popup(x, y)
            except Exception as e:
                print(f"[ch popup failed: {e}; falling back to cycle]")
                _cycle_channel(idx)
        return handler

    for i, btn_ch in enumerate(ch_buttons):
        btn_ch.on_clicked(make_ch_popup(i))

    def apply_cmap_to_ref(idx, spec_text):
        """Set ref idx's colormap from a textual spec (pastel name, mpl name,
        or #hex). Returns True on success, False if spec is unparseable."""
        cmap_obj, tint = _resolve_user_cmap_and_tint(spec_text)
        if cmap_obj is None:
            return False
        cmaps[idx] = spec_text.strip()
        tints[idx] = tint
        intensity_ims[idx].set_cmap(cmap_obj)
        prob_ims[idx].set_cmap(cmap_obj)
        intensity_axes[idx].title.set_color(tint)
        prob_axes[idx].title.set_color(tint)
        spec_overlay_lines[idx].set_color(tint)
        sel_lines[idx].set_color(tint)
        # Sync the textbox to the canonical spec we stored
        try:
            cmap_textboxes[idx].set_val(cmaps[idx])
        except Exception:
            pass
        # Update swatch button color
        btn = color_swatch_btns[idx]
        btn.color = tint
        btn.hovercolor = tint
        btn.ax.set_facecolor(tint)
        # Rebuild spectrum-overlay legend
        spec_ax.legend(loc="upper right", fontsize=8,
                       ncol=min(n_ref + 1, 4))
        # *** Re-render the combined RGB so the user's new color actually
        # shows up in it (this was the bug — the RGB was stuck on the
        # initial pastel R/G/B). ***
        try:
            refresh_rgb_display()
        except Exception:
            # refresh_rgb_display may not be in scope yet during init —
            # that's fine, the initial render will pick the new tints up.
            pass
        fig.canvas.draw_idle()
        return True

    def make_cmap_handler(idx):
        def handler(text):
            if not apply_cmap_to_ref(idx, text):
                # invalid -> revert text
                cmap_textboxes[idx].set_val(str(cmaps[idx]))
        return handler

    for i, tb_cm in enumerate(cmap_textboxes):
        tb_cm.on_submit(make_cmap_handler(i))

    # ---- Paint-style color picker per reference ----
    def make_picker_handler(idx):
        def handler(_event):
            import matplotlib.colors as mcolors
            try:
                current_hex = mcolors.to_hex(tints[idx])
            except Exception:
                current_hex = "#888888"
            # Use the existing matplotlib Tk root by creating a temporary
            # hidden Toplevel as parent. This keeps the dialog modal-feeling
            # without spawning a competing Tk mainloop.
            try:
                # askcolor returns ((r,g,b), "#rrggbb") or (None, None)
                picked = colorchooser.askcolor(
                    color=current_hex,
                    title=f"Pick color for {refs[idx]['name']}",
                )
            except Exception as e:
                print(f"Color picker failed: {e}")
                return
            if picked is None or picked[1] is None:
                return  # user cancelled
            hex_color = picked[1]
            apply_cmap_to_ref(idx, hex_color)
        return handler

    for i, btn_sw in enumerate(color_swatch_btns):
        btn_sw.on_clicked(make_picker_handler(i))

    # Initialize RGB display
    arr0, (vmin0, vmax0), _ = metric_arrays(current_metric["name"])
    rgb_im.set_data(build_rgb(_normalize_for_rgb(arr0, vmin0, vmax0)))

    # WN textbox handler
    def make_wn_handler(idx):
        def handler(text):
            try:
                v = float(text)
            except ValueError:
                return
            j = int(np.argmin(np.abs(wn - v)))
            actual = float(wn[j])
            selected_wn[idx] = actual
            img = cube[:, :, j]
            intensity_ims[idx].set_data(img)
            intensity_ims[idx].set_clim(float(img.min()), float(img.max() + 1e-9))
            intensity_axes[idx].set_title(
                f"Intensity: {refs[idx]['name']} @ {actual:.1f} {CM1}",
                fontsize=9, fontweight="bold", color=tints[idx])
            sel_lines[idx].set_xdata([actual, actual])
            wn_textboxes[idx].set_val(f"{actual:.1f}")
            fig.canvas.draw_idle()
        return handler

    for i, tb in enumerate(wn_textboxes):
        tb.on_submit(make_wn_handler(i))

    # Intensity vmin / vmax textbox handlers
    def make_vmin_handler(idx):
        def handler(text):
            try:
                v = float(text)
            except ValueError:
                return
            cur_vmax = intensity_ims[idx].get_clim()[1]
            if v >= cur_vmax:
                v = cur_vmax - 1e-6
            intensity_ims[idx].set_clim(v, cur_vmax)
            vmin_textboxes[idx].set_val(f"{v:.1f}")
            fig.canvas.draw_idle()
        return handler

    def make_vmax_handler(idx):
        def handler(text):
            try:
                v = float(text)
            except ValueError:
                return
            cur_vmin = intensity_ims[idx].get_clim()[0]
            if v <= cur_vmin:
                v = cur_vmin + 1e-6
            intensity_ims[idx].set_clim(cur_vmin, v)
            vmax_textboxes[idx].set_val(f"{v:.1f}")
            fig.canvas.draw_idle()
        return handler

    for i, tb in enumerate(vmin_textboxes):
        tb.on_submit(make_vmin_handler(i))
    for i, tb in enumerate(vmax_textboxes):
        tb.on_submit(make_vmax_handler(i))

    # Click in spectrum to retarget the FIRST reference (or use textboxes)
    active_ref = {"idx": 0}

    # Active-ref selector
    if n_ref > 1:
        ax_ref = fig.add_axes([0.01, 0.50, 0.10, 0.22], facecolor="#f7f7f7")
        ax_ref.set_title("Active ref\n(click spec)", fontsize=9,
                         fontweight="bold", pad=4)
        ref_radio = RadioButtons(
            ax_ref, labels=tuple(r["name"] for r in refs), active=0)
        for lbl, tint in zip(ref_radio.labels, tints):
            lbl.set_color(tint)
            lbl.set_fontsize(8)
            lbl.set_fontweight("bold")

        def on_ref(label):
            for i, r in enumerate(refs):
                if r["name"] == label:
                    active_ref["idx"] = i
                    return

        ref_radio.on_clicked(on_ref)

    def _show_pixel_spectrum(xi, yi):
        spec = cube[yi, xi, :]
        test_line.set_ydata(spec)
        spec_state["mode"] = "pixel"
        spec_state["pixel"] = (xi, yi)
        spec_ax.set_title(
            f"Pixel spectrum @ X={x_coords[xi]:.0f}, Y={y_coords[yi]:.0f}  "
            f"(click 'Mean' to revert)",
            fontsize=10, fontweight="bold",
        )
        # Adjust y-limits to the pixel spectrum range (fast, no full relim).
        y_min = float(min(spec.min(), 0.0))
        y_max = float(spec.max() * 1.08 + 1e-9)
        spec_ax.set_ylim(y_min, y_max)
        fig.canvas.draw_idle()

    def _show_mean_spectrum():
        test_line.set_ydata(mean_test)
        spec_state["mode"] = "mean"
        spec_state["pixel"] = None
        spec_ax.set_title("Mean spectrum across all pixels",
                          fontsize=10, fontweight="bold")
        spec_ax.relim()
        spec_ax.autoscale_view(scalex=False, scaley=True)
        fig.canvas.draw_idle()

    _show_mean_spectrum()  # set initial title style

    def on_click(event):
        # Ignore clicks while the matplotlib pan/zoom toolbar is active
        toolbar = getattr(fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return
        if event.xdata is None or event.ydata is None:
            return

        # Click in the spectrum panel -> retarget the active ref's wn
        if event.inaxes is spec_ax:
            idx = active_ref["idx"]
            make_wn_handler(idx)(f"{event.xdata:.1f}")
            return

        # Click on any intensity or metric map -> show that pixel's spectrum
        all_map_axes = list(intensity_axes) + list(prob_axes)
        if event.inaxes in all_map_axes:
            xi = int(np.argmin(np.abs(x_coords - event.xdata)))
            yi = int(np.argmin(np.abs(y_coords - event.ydata)))
            if 0 <= xi < cube.shape[1] and 0 <= yi < cube.shape[0]:
                _show_pixel_spectrum(xi, yi)
                print(f"[pixel-click spectrum] x={x_coords[xi]:.1f} "
                      f"y={y_coords[yi]:.1f}")

    fig.canvas.mpl_connect("button_press_event", on_click)

    # Save buttons — to the left of the RGB axes
    ax_save = fig.add_axes([0.01, 0.135, 0.10, 0.04])
    btn_save = Button(ax_save, "Save All",
                      color="#cde7ff", hovercolor="#9fcbff")
    btn_save.label.set_fontsize(10)
    btn_save.label.set_fontweight("bold")

    ax_save_sel = fig.add_axes([0.01, 0.085, 0.10, 0.04])
    btn_save_sel = Button(ax_save_sel, "Save...",
                          color="#ffe9b0", hovercolor="#ffd270")
    btn_save_sel.label.set_fontsize(10)
    btn_save_sel.label.set_fontweight("bold")

    # "Mean" revert button — switches the top panel back to the test mean
    # spectrum after a pixel click.
    ax_mean = fig.add_axes([0.01, 0.035, 0.10, 0.04])
    btn_mean = Button(ax_mean, "Mean spec",
                      color="#e5f0d8", hovercolor="#c9e3a8")
    btn_mean.label.set_fontsize(10)
    btn_mean.label.set_fontweight("bold")
    btn_mean.on_clicked(lambda _e: _show_mean_spectrum())

    status_ax = fig.add_axes([0.78, 0.02, 0.21, 0.04])
    status_ax.axis("off")
    status_text = status_ax.text(
        0.0, 0.5, f"Output: {result['out_dir'].name}",
        transform=status_ax.transAxes, fontsize=8, va="center",
    )

    def _revert_label_after(btn, original_text, delay_ms=900):
        """Non-blocking label revert. matplotlib's start_event_loop can
        reorder queued button events, so we use a one-shot timer instead."""
        try:
            t = fig.canvas.new_timer(interval=delay_ms)
            t.single_shot = True

            def _do():
                try:
                    btn.label.set_text(original_text)
                    fig.canvas.draw_idle()
                except Exception:
                    pass
            t.add_callback(_do)
            t.start()
        except Exception:
            btn.label.set_text(original_text)
            fig.canvas.draw_idle()

    def on_save(_):
        print("[Save All button clicked]")
        try:
            save_all(result, selected_wn, fig, mean_test, rgb_assignment,
                     cmaps_override=cmaps, tints_override=tints)
            btn_save.label.set_text("Saved ✓")
            status_text.set_text(f"Saved to: {result['out_dir']}")
            fig.canvas.draw_idle()
            _revert_label_after(btn_save, "Save All")
        except Exception as e:
            print(f"Save failed: {e}")
            messagebox.showerror("Save error", str(e))

    btn_save.on_clicked(on_save)

    def on_save_selective(_):
        print("[Save... button clicked]")
        # Pop a Tk dialog with checkboxes; if accepted, pass selection dict.
        try:
            selection = _ask_save_selection(current_metric["name"])
        except Exception as e:
            print(f"Save dialog failed: {e}")
            return
        if selection is None:
            print("[Save... dialog cancelled — no files written]")
            return
        try:
            save_all(result, selected_wn, fig, mean_test, rgb_assignment,
                     cmaps_override=cmaps, tints_override=tints,
                     selection=selection)
            btn_save_sel.label.set_text("Saved ✓")
            status_text.set_text(f"Saved to: {result['out_dir']}")
            fig.canvas.draw_idle()
            _revert_label_after(btn_save_sel, "Save...")
        except Exception as e:
            print(f"Save failed: {e}")
            messagebox.showerror("Save error", str(e))

    btn_save_sel.on_clicked(on_save_selective)

    # Explicit close handler — proves to the user that closing the window
    # does NOT call save_all. If you see [VIEWER WINDOW CLOSED] in the
    # console but no [save_all called], then nothing was written on close.
    def _on_close(_event):
        print("[VIEWER WINDOW CLOSED — exiting (no save triggered)]")

    fig.canvas.mpl_connect("close_event", _on_close)

    plt.show()


# =============================================================================
# Save All
# =============================================================================

def _figsize_for_map(arr_shape, base_long=10.0, extra_w=2.2, min_short=4.5):
    """Compute a figsize that respects the map's aspect ratio.

    arr_shape: (ny, nx). Longer side is set to base_long, the other scaled
    so square pixels render square. extra_w accounts for colorbar + margins.
    """
    ny, nx = arr_shape[0], arr_shape[1]
    if nx >= ny:
        w_map = base_long
        h_map = max(base_long * (ny / max(nx, 1)), min_short)
    else:
        h_map = base_long
        w_map = max(base_long * (nx / max(ny, 1)), min_short)
    return (w_map + extra_w, h_map + 1.4)  # +width for colorbar, +height for title


def _save_map_png(arr, cmap, vmin, vmax, extent, title, label, out_path,
                  cm_pad=0.05, fig_w=None, dpi=None):
    """Publication-quality single map. One file per call (no separate
    'polish' duplicate). Arial 18pt from rcParams, thick spines,
    long ticks, 600 DPI by default."""
    cmap = _resolve_cmap(cmap)
    if fig_w is None:
        fig_size = _figsize_for_map(arr.shape, base_long=11.0,
                                     extra_w=2.6, min_short=5.5)
    else:
        fig_size = (fig_w, fig_w * 0.85)
    if dpi is None:
        dpi = 600

    fig, ax = plt.subplots(figsize=fig_size)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, origin="upper",
                   aspect="equal", interpolation="nearest")
    ax.set_title(title, fontweight="bold", pad=14)
    ax.set_xlabel("X (um)", labelpad=8)
    ax.set_ylabel("Y (um)", labelpad=8)
    for s in ax.spines.values():
        s.set_linewidth(1.8)
    ax.tick_params(width=1.6, length=7)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=cm_pad)
    cbar.set_label(label)
    cbar.outline.set_linewidth(1.4)
    cbar.ax.tick_params(width=1.4, length=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_polish_map_png(arr, cmap, vmin, vmax, extent, title, label,
                          out_path, dpi=600):
    """Publication-ready ('polish') variant of a map.

    - Strict Arial 18pt (inherits rcParams; no fontsize overrides).
    - 600 DPI by default.
    - Thicker spines + longer ticks for slide readability.
    - Tight white background, no extraneous grid.
    """
    cmap = _resolve_cmap(cmap)
    fig_size = _figsize_for_map(arr.shape, base_long=11.0,
                                 extra_w=2.6, min_short=5.5)
    fig, ax = plt.subplots(figsize=fig_size)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, origin="upper",
                   aspect="equal", interpolation="nearest")
    ax.set_title(title, fontweight="bold", pad=14)
    ax.set_xlabel("X (um)", labelpad=8)
    ax.set_ylabel("Y (um)", labelpad=8)
    for s in ax.spines.values():
        s.set_linewidth(1.8)
    ax.tick_params(width=1.6, length=7)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
    cbar.set_label(label)
    cbar.outline.set_linewidth(1.4)
    cbar.ax.tick_params(width=1.4, length=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_polish_combined_rgb(arr_norm, refs, rgb_assignment, extent,
                               out_path, title, tints=None, dpi=600):
    """Publication-ready combined RGB. Same blending as _save_combined_rgb
    but with thicker spines, larger margins, no clipping."""
    rgb = _build_pastel_rgb(arr_norm, rgb_assignment, tints=tints)
    fig_size = _figsize_for_map(rgb.shape[:2], base_long=11.0,
                                 extra_w=2.4, min_short=6.0)
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(rgb, extent=extent, origin="upper",
              aspect="equal", interpolation="nearest")
    ax.set_title(title, fontweight="bold", pad=14)
    ax.set_xlabel("X (um)", labelpad=8)
    ax.set_ylabel("Y (um)", labelpad=8)
    for s in ax.spines.values():
        s.set_linewidth(1.8)
    ax.tick_params(width=1.6, length=7)

    # Legend with tint swatches
    handles = []
    ch_label = {"1": "R", "2": "G", "3": "B"}
    for i, r in enumerate(refs):
        if i >= len(rgb_assignment):
            break
        axis = _ch_to_axis(rgb_assignment[i])
        if axis is None:
            continue
        disp = _ch_to_display(rgb_assignment[i])
        tint = (tints[i] if (tints is not None and i < len(tints))
                else (0.5, 0.5, 0.5))
        handles.append(mpatches.Patch(
            facecolor=tint, edgecolor="#444",
            label=f"ch{disp}: {r['name']}"))
    if handles:
        ax.legend(handles=handles, loc="upper right",
                  framealpha=0.92, edgecolor="#444")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_raw_png(arr, cmap, vmin, vmax, out_path, dpi=400, pixel_scale=20):
    cmap = _resolve_cmap(cmap)
    h, w = arr.shape
    out_w = max(w * pixel_scale, 1)
    out_h = max(h * pixel_scale, 1)
    fig = plt.figure(figsize=(out_w / dpi, out_h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
              origin="upper", aspect="equal", interpolation="nearest")
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def save_all(result, selected_wn, fig_main, mean_test, rgb_assignment=None,
             cmaps_override=None, tints_override=None, selection=None,
             intensity_clims=None):
    """Save outputs to result['out_dir'].

    Args:
        result, selected_wn, fig_main, mean_test : as before
        rgb_assignment : list of 'R'/'G'/'B'/'-' per ref
        cmaps_override : list of cmap specs (strings or Colormap) per ref;
                         overrides the default REF_CMAPS so the GUI's chosen
                         colors persist into saved PNGs
        tints_override : list of RGB tints per ref to match the cmap overrides
        selection      : optional dict of bool flags for which output groups
                         to write. Keys: intensity, current_metric,
                         all_metrics, combined_rgb, mean_spectrum, mcr_spectra,
                         histograms, confidence, agreement, validation,
                         per_metric_csvs, summary_csvs, raw_versions.
                         If None or all-True, save everything (legacy behavior).
                         The viewer also passes current_metric_name when
                         current_metric is True so we know which metric only
                         to save.
    """
    out_dir = result["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = result["test_path"].stem
    refs = result["refs"]
    cube = result["test_cube"]
    wn = result["wn"]
    x_coords = result["x_coords"]
    y_coords = result["y_coords"]
    extent = [x_coords.min(), x_coords.max(),
              y_coords.min(), y_coords.max()]
    if cmaps_override is not None:
        cmaps = list(cmaps_override)
    else:
        cmaps = [REF_CMAPS[i % len(REF_CMAPS)] for i in range(len(refs))]
    if tints_override is not None:
        tints = list(tints_override)
    else:
        tints = [REF_TINTS[i % len(REF_TINTS)] for i in range(len(refs))]
    if rgb_assignment is None:
        rgb_assignment = (["R", "G", "B"] + ["-"] * max(0, len(refs) - 3))[:len(refs)]

    # ---- Selection defaults (everything ON if not provided) ----
    def _sel(key, default=True):
        if selection is None:
            return default
        return bool(selection.get(key, default))
    do_intensity     = _sel("intensity")
    do_all_metrics   = _sel("all_metrics")
    do_current_only  = _sel("current_metric_only", False)
    current_metric   = (selection or {}).get("current_metric_name", "NNLS_norm")
    do_combined_rgb  = _sel("combined_rgb")
    do_mean_spec     = _sel("mean_spectrum")
    do_mcr_spec      = _sel("mcr_spectra")
    do_histograms    = _sel("histograms")
    do_confidence    = _sel("confidence")
    do_agreement     = _sel("agreement")
    do_validation    = _sel("validation")
    do_per_metric_csv = _sel("per_metric_csvs")
    do_summary_csv   = _sel("summary_csvs")
    do_raw           = _sel("raw_versions")

    import traceback
    print(f"\n[save_all CALLED]  Saving to: {out_dir}")
    print(f"  selection preset: "
          f"essential={do_current_only and not do_all_metrics}, "
          f"all_metrics={do_all_metrics}, current_only={do_current_only}")
    # Brief stack so the source of the call is visible
    caller = traceback.extract_stack(limit=3)[-2]
    print(f"  called from: {caller.filename}:{caller.lineno} in {caller.name}")

    # 1. Intensity maps (annotated + raw) at the user-selected wn
    if do_intensity:
        for i, r in enumerate(refs):
            j = int(np.argmin(np.abs(wn - selected_wn[i])))
            actual = float(wn[j])
            img = cube[:, :, j]
            # Honor the user's GUI vmin/vmax when provided; otherwise fall
            # back to the data's own min/max.
            if (intensity_clims is not None
                    and i < len(intensity_clims)
                    and intensity_clims[i] is not None):
                vmin, vmax = intensity_clims[i]
            else:
                vmin = float(img.min())
                vmax = float(img.max() + 1e-9)
            tag = f"{stem}_{r['name']}_intensity_{actual:.0f}cm"
            _save_map_png(
                img, cmaps[i], vmin, vmax, extent,
                title=f"{r['name']} - intensity @ {actual:.1f} {CM1}",
                label="Intensity",
                out_path=out_dir / f"{tag}.png",
            )
            if do_raw:
                _save_raw_png(
                    img, cmaps[i], vmin, vmax,
                    out_dir / f"{tag}_raw.png",
                )
            print(f"  {tag}.png")

    # 2. Probability / metric maps. If 'current_metric_only' is set, only the
    #    metric currently displayed in the viewer is saved.
    metric_set = [
        ("NNLS_norm",    result["nnls_norm"], 0.0, 1.0, "contribution"),
        ("MCR_contrib",  result["mcr_contrib"], 0.0, 1.0, "contribution"),
        ("CLS_norm",     result["cls_norm"], 0.0, 1.0, "contribution"),
        ("Pearson_prob", result["prob_maps"], 0.0, 1.0, "probability"),
        ("Cosine_sim",   result["cos_maps"], 0.0, 1.0, "cosine"),
        ("Pearson_corr", result["corr_maps"], -1.0, 1.0, "Pearson r"),
        ("NNLS_raw",     result["nnls_raw"],
         float(result["nnls_raw"].min()),
         float(result["nnls_raw"].max() + 1e-9), "weight (raw)"),
        ("CLS_raw",      result["cls_raw"],
         float(result["cls_raw"].min()),
         float(result["cls_raw"].max() + 1e-9), "coeff (signed)"),
        ("MCR_conc",     result["mcr_conc"],
         float(result["mcr_conc"].min()),
         float(result["mcr_conc"].max() + 1e-9), "concentration"),
    ]
    # Map "NNLS norm" -> "NNLS_norm" etc. to bridge viewer names to file tags
    name_to_key = {m[0].replace("_", " "): m[0] for m in metric_set}
    current_key = name_to_key.get(current_metric, current_metric)

    if do_all_metrics:
        save_metric_names = [m[0] for m in metric_set]
    elif do_current_only:
        save_metric_names = [current_key]
    else:
        save_metric_names = []

    for metric_name, arr, vmin, vmax, label in metric_set:
        if metric_name not in save_metric_names:
            continue
        for i, r in enumerate(refs):
            tag = f"{stem}_{r['name']}_{metric_name}"
            _save_map_png(
                arr[i], cmaps[i], vmin, vmax, extent,
                title=f"{r['name']} - {metric_name.replace('_', ' ')}",
                label=label,
                out_path=out_dir / f"{tag}.png",
            )
            if do_raw:
                _save_raw_png(
                    arr[i], cmaps[i], vmin, vmax,
                    out_dir / f"{tag}_raw.png",
                )
        print(f"  {metric_name}: {len(refs)} PNGs")

    # 2b. Combined RGB contribution maps (NNLS + MCR + CLS, each as one PNG)
    if do_combined_rgb:
        rgb_sources = [
            ("NNLS_norm",   result["nnls_norm"]),
            ("MCR_contrib", result["mcr_contrib"]),
            ("CLS_norm",    result["cls_norm"]),
        ]
        # If user is in 'current metric only' mode, only emit the RGB matching
        # the currently viewed metric (still falls back to NNLS if the current
        # metric has no RGB analogue).
        if do_current_only:
            rgb_sources = [src for src in rgb_sources
                           if src[0] == current_key] or [rgb_sources[0]]
        for src_name, arr in rgb_sources:
            raw_p = (out_dir / f"{stem}_combined_RGB_{src_name}_raw.png"
                     if do_raw else None)
            _save_combined_rgb(
                arr, refs, rgb_assignment, extent,
                out_path=out_dir / f"{stem}_combined_RGB_{src_name}.png",
                raw_path=raw_p,
                title=f"Combined RGB ({src_name.replace('_', ' ')}) - {stem}",
                tints=tints,
            )
            print(f"  combined_RGB_{src_name}.png")

        # Intensity-based combined RGB — each ref contributes its
        # intensity-at-selected-wn slice, normalized to 0..1 using either
        # the user-set clim (if given) or the data's range.
        ny, nx, _ = cube.shape
        intens_norm = np.zeros((len(refs), ny, nx))
        for i, r in enumerate(refs):
            j = int(np.argmin(np.abs(wn - selected_wn[i])))
            slab = cube[:, :, j]
            if (intensity_clims is not None
                    and i < len(intensity_clims)
                    and intensity_clims[i] is not None):
                vmin, vmax = intensity_clims[i]
            else:
                vmin = float(slab.min())
                vmax = float(slab.max() + 1e-9)
            span = max(vmax - vmin, 1e-12)
            intens_norm[i] = np.clip((slab - vmin) / span, 0.0, 1.0)
        raw_p = (out_dir / f"{stem}_combined_RGB_intensity_raw.png"
                 if do_raw else None)
        _save_combined_rgb(
            intens_norm, refs, rgb_assignment, extent,
            out_path=out_dir / f"{stem}_combined_RGB_intensity.png",
            raw_path=raw_p,
            title=f"Combined RGB (intensity @ selected wn) - {stem}",
            tints=tints,
        )
        print(f"  combined_RGB_intensity.png")

    # 3. Mean spectrum overlay
    if do_mean_spec:
        fig2, ax2 = plt.subplots(figsize=(15, 6))
        ax2.plot(wn, mean_test, color="#444", lw=1.6, label="Test mean")
        for i, (r, spec) in enumerate(zip(refs, result["ref_specs"])):
            scale = (mean_test.max() / (spec.max() + 1e-12)) * 0.9
            ax2.plot(wn, spec * scale, color=tints[i], lw=1.4, alpha=0.85,
                     label=f"{r['name']} (scaled)")
        for i, w_sel in enumerate(selected_wn):
            ax2.axvline(w_sel, color=tints[i], lw=1.6, alpha=0.7)
        ax2.set_xlabel(f"Wavenumber ({CM1})")
        ax2.set_ylabel("Intensity")
        ax2.set_title(f"{stem} - test mean vs. references", fontweight="bold")
        ax2.grid(alpha=0.3)
        ax2.legend(loc="upper right")
        ax2.set_xlim(wn[0], wn[-1])
        fig2.tight_layout()
        fig2.savefig(out_dir / f"{stem}_mean_spectrum_overlay.png",
                     dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig2)
        print(f"  {stem}_mean_spectrum_overlay.png")

    # 4. CSV exports — flatten pixels with (x, y) per row
    ny, nx, _ = cube.shape
    ref_names = [r["name"] for r in refs]

    if do_per_metric_csv:
        # 4a. Probability tables
        _save_pixel_metric_csv(
            out_dir / f"{stem}_pearson_prob.csv",
            result["prob_maps"], x_coords, y_coords, ref_names, "prob",
        )
        _save_pixel_metric_csv(
            out_dir / f"{stem}_cosine_sim.csv",
            result["cos_maps"], x_coords, y_coords, ref_names, "cos",
        )
        _save_pixel_metric_csv(
            out_dir / f"{stem}_nnls_norm.csv",
            result["nnls_norm"], x_coords, y_coords, ref_names, "nnls_norm",
        )
        _save_pixel_metric_csv(
            out_dir / f"{stem}_pearson_corr.csv",
            result["corr_maps"], x_coords, y_coords, ref_names, "corr",
        )
        _save_pixel_metric_csv(
            out_dir / f"{stem}_nnls_raw.csv",
            result["nnls_raw"], x_coords, y_coords, ref_names, "nnls_raw",
        )
        _save_pixel_metric_csv(
            out_dir / f"{stem}_cls_norm.csv",
            result["cls_norm"], x_coords, y_coords, ref_names, "cls_norm",
        )
        _save_pixel_metric_csv(
            out_dir / f"{stem}_cls_raw.csv",
            result["cls_raw"], x_coords, y_coords, ref_names, "cls_raw",
        )
        _save_pixel_metric_csv(
            out_dir / f"{stem}_mcr_contrib.csv",
            result["mcr_contrib"], x_coords, y_coords, ref_names, "mcr_contrib",
        )
        _save_pixel_metric_csv(
            out_dir / f"{stem}_mcr_conc.csv",
            result["mcr_conc"], x_coords, y_coords, ref_names, "mcr_conc",
        )

        # 4b. Intensity-at-selected-wn table
        intensity_arrs = []
        intensity_labels = []
        for i, r in enumerate(refs):
            j = int(np.argmin(np.abs(wn - selected_wn[i])))
            intensity_arrs.append(cube[:, :, j])
            intensity_labels.append(f"{r['name']}@{wn[j]:.1f}")
        intensity_stack = np.stack(intensity_arrs, axis=0)
        _save_pixel_metric_csv(
            out_dir / f"{stem}_selected_intensity.csv",
            intensity_stack, x_coords, y_coords, intensity_labels, "I",
        )

    # 5. Per-reference histograms (Pearson + Cosine)
    if do_histograms:
        for i, r in enumerate(refs):
            corr_pix = result["corr_maps"][i].ravel()
            cos_pix = result["cos_maps"][i].ravel()
            safe = r["name"].replace("/", "_").replace("\\", "_")
            fname = f"{stem}_histograms_vs_{safe}.png"
            _save_histograms(
                out_dir / fname, r["name"], corr_pix, cos_pix,
            )
            print(f"  {fname}")

    # 5b. Reliability maps — confidence (gap + entropy), per-method argmax,
    # agreement across methods, consensus label
    ref_names_list = [r["name"] for r in refs]
    n_ref = len(refs)

    if do_confidence:
        # Confidence: gap (top1 - top2) for NNLS and MCR
        for tag, arr in [("NNLS", result["confidence_gap_nnls"]),
                         ("MCR",  result["confidence_gap_mcr"])]:
            _save_map_png(
                arr, "PastelGreen", 0.0, 1.0, extent,
                title=f"{stem} - confidence gap (top1 - top2, {tag})",
                label="gap (0 = ambiguous, 1 = certain)",
                out_path=out_dir / f"{stem}_confidence_gap_{tag}.png",
            )
        # Entropy: lower = more confident. Use PastelOrange.
        max_ent = float(np.log2(max(n_ref, 2)))
        for tag, arr in [("NNLS", result["entropy_nnls"]),
                         ("MCR",  result["entropy_mcr"])]:
            _save_map_png(
                arr, "PastelOrange", 0.0, max_ent, extent,
                title=f"{stem} - entropy ({tag}, log2 bits)",
                label="entropy (0 = certain, higher = ambiguous)",
                out_path=out_dir / f"{stem}_entropy_{tag}.png",
            )

    if do_agreement:
        # Per-method argmax categorical maps
        for tag, lbl in [("NNLS", result["argmax_nnls"]),
                         ("MCR",  result["argmax_mcr"]),
                         ("CLS",  result["argmax_cls"])]:
            _save_categorical_map_png(
                lbl, ref_names_list, tints, extent,
                title=f"{stem} - argmax label ({tag})",
                out_path=out_dir / f"{stem}_argmax_{tag}.png",
                tie_label="(none)",
            )

        # Agreement & consensus
        _save_agreement_map_png(
            result["agreement"], extent,
            title=f"{stem} - method agreement (NNLS / MCR / CLS)",
            out_path=out_dir / f"{stem}_method_agreement.png",
            n_methods=3,
        )
        _save_categorical_map_png(
            result["consensus"], ref_names_list, tints, extent,
            title=f"{stem} - consensus label (majority across NNLS/MCR/CLS)",
            out_path=out_dir / f"{stem}_consensus_label.png",
            tie_label="(tie / no majority)",
        )
        summ = result["agreement_summary"]
        print(f"  confidence + agreement maps: "
              f"3/3 agree {100*summ['full']/summ['n_pix']:.1f}%, "
              f"2/3 {100*summ['two']/summ['n_pix']:.1f}%, "
              f"no-majority {100*summ['split']/summ['n_pix']:.1f}%")

    # 6. MCR-resolved spectra plot
    if do_mcr_spec:
        mcr_S = result["mcr_S"]
        fig_s, ax_s = plt.subplots(figsize=(15, 6))
        for i, r in enumerate(refs):
            ax_s.plot(wn, mcr_S[i], color=tints[i], lw=1.6,
                      label=f"MCR-ALS: {r['name']}")
        ax_s.set_xlabel(f"Wavenumber ({CM1})")
        ax_s.set_ylabel("Resolved intensity (normalized)")
        ax_s.set_title(
            f"{stem} - MCR-ALS resolved spectra "
            f"(iter={result['mcr_iter']}, resid={result['mcr_resid']:.3g})",
            fontweight="bold")
        ax_s.grid(alpha=0.3)
        ax_s.legend()
        ax_s.set_xlim(wn[0], wn[-1])
        fig_s.tight_layout()
        fig_s.savefig(out_dir / f"{stem}_mcr_resolved_spectra.png",
                      dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig_s)
        print(f"  {stem}_mcr_resolved_spectra.png")

    # 7. Summary CSVs
    if do_summary_csv:
        peak_intensity_stats = []
        for i, r in enumerate(refs):
            peaks = result["ref_top_peaks"][i]
            peak_wn = (peaks[0][0] if peaks
                        else float(wn[int(np.argmax(result["ref_specs"][i]))]))
            j = int(np.argmin(np.abs(wn - peak_wn)))
            peak_intensity = cube[:, :, j].ravel()
            p_mean = float(peak_intensity.mean())
            p_std = float(peak_intensity.std())
            p_rsd = (p_std / abs(p_mean) * 100.0) if abs(p_mean) > 1e-12 else 0.0
            peak_intensity_stats.append((float(wn[j]), p_mean, p_std, p_rsd))

        stats_name = f"{stem}_statistics_summary.csv"
        _save_statistics_summary(
            out_dir / stats_name, refs, result, peak_intensity_stats,
        )
        print(f"  {stats_name}")

        rel_name = f"{stem}_reliability_summary.csv"
        _save_reliability_summary(out_dir / rel_name, result, refs)
        print(f"  {rel_name}")

    # Classifier validation reports (independent of the test sample)
    if do_validation:
        cv_result = result.get("cv_result")
        synth_result = result.get("synth_result")
        if cv_result is not None:
            cv_png = f"{stem}_classifier_CV_confusion.png"
            _save_cv_confusion_png(cv_result, out_dir / cv_png)
            print(f"  {cv_png}")
        if synth_result is not None:
            sc_png = f"{stem}_classifier_synth_scatter.png"
            _save_synth_scatter_png(synth_result,
                                     [r["name"] for r in refs], tints,
                                     out_dir / sc_png)
            print(f"  {sc_png}")
        # Noise-sweep stress test (the more informative curve)
        sweep_result = result.get("noise_sweep_result")
        if sweep_result is not None:
            ns_png = f"{stem}_classifier_noise_sweep.png"
            _save_noise_sweep_png(sweep_result,
                                   [r["name"] for r in refs], tints,
                                   out_dir / ns_png)
            print(f"  {ns_png}")
        if cv_result is not None or synth_result is not None:
            val_csv = f"{stem}_classifier_validation.csv"
            _save_validation_csv(out_dir / val_csv, cv_result, synth_result,
                                  [r["name"] for r in refs])
            print(f"  {val_csv}")

    print("Done.")


def _save_categorical_map_png(label_arr, ref_names, tints, extent, title,
                              out_path, tie_label="(none/tie)"):
    """Discrete categorical label map. label_arr: int (ny, nx) with values in
    {-1, 0, 1, ..., n_ref-1}. -1 means tie / no majority. Each ref index
    rendered in its pastel tint; ties rendered in light grey."""
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    n_ref = len(ref_names)
    # Colors: index -1 -> grey, index 0..n_ref-1 -> tints[i]
    # We'll remap labels to 0..n_ref where 0 = tie, 1..n_ref = ref labels
    remapped = np.where(label_arr < 0, 0, label_arr + 1).astype(int)
    palette = [(0.82, 0.82, 0.82)] + [tints[i] for i in range(n_ref)]
    cmap = mcolors.ListedColormap(palette)
    bounds = np.arange(n_ref + 2) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig_size = _figsize_for_map(label_arr.shape, base_long=10.0,
                                 extra_w=2.6, min_short=5.0)
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(remapped, cmap=cmap, norm=norm, extent=extent, origin="upper",
              aspect="equal", interpolation="nearest")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("X (um)")
    ax.set_ylabel("Y (um)")

    # Legend
    legend_handles = [Patch(facecolor=palette[0], edgecolor="#888",
                            label=tie_label)]
    for i, name in enumerate(ref_names):
        legend_handles.append(
            Patch(facecolor=palette[i + 1], edgecolor="#888", label=name))
    ax.legend(handles=legend_handles, loc="center left",
              bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def _save_agreement_map_png(agreement, extent, title, out_path,
                            n_methods=3):
    """Discrete agreement map: values in {1..n_methods}. Pastel diverging
    palette from light grey (low agreement) -> deeper green (high)."""
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    palette = ["#dddddd", "#e8d18a", "#a3d39c"]  # 1, 2, 3 methods agree
    if n_methods > 3:
        palette = palette + ["#5fb16f"] * (n_methods - 3)
    palette = palette[:n_methods]
    cmap = mcolors.ListedColormap(palette)
    bounds = np.arange(1, n_methods + 2) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig_size = _figsize_for_map(agreement.shape, base_long=10.0,
                                 extra_w=2.6, min_short=5.0)
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(agreement, cmap=cmap, norm=norm, extent=extent, origin="upper",
              aspect="equal", interpolation="nearest")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("X (um)")
    ax.set_ylabel("Y (um)")
    legend_handles = []
    labels = {1: "1/3 (no majority)",
              2: "2/3 (partial agreement)",
              3: f"{n_methods}/{n_methods} (full agreement)"}
    for k in range(1, n_methods + 1):
        legend_handles.append(Patch(facecolor=palette[k - 1], edgecolor="#888",
                                    label=labels.get(k, f"{k}/{n_methods}")))
    ax.legend(handles=legend_handles, loc="center left",
              bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def _ch_to_axis(ch):
    """Normalize a channel spec into an axis index (legacy) or 'include' flag.

    The combined-RGB image now blends user-picked tints directly, so the
    channel number only decides whether a ref participates (any positive
    integer -> include; 0 / '-' / off -> exclude). Backwards-compatible
    with the old R/G/B form so saved configs still work.

        1 / 'R' -> 0  (legacy: red axis; new: include with own tint)
        2 / 'G' -> 1
        3 / 'B' -> 2
        4 .. N  -> 0  (any positive integer -> include)
        0 / '-' / off / blank -> None  (exclude)
    """
    if ch is None:
        return None
    try:
        s = str(ch).strip().upper()
    except Exception:
        return None
    if s in ("0", "-", "OFF", "NONE", ""):
        return None
    if s in ("R", "RED"):
        return 0
    if s in ("G", "GREEN"):
        return 1
    if s in ("B", "BLUE"):
        return 2
    # Numeric: any positive integer counts as "include". We map 1/2/3 to
    # axes 0/1/2 for legacy R/G/B behavior; higher numbers also include
    # but axis defaults to 0 (irrelevant since combined-RGB uses tints).
    try:
        n = int(s)
        if n <= 0:
            return None
        return (n - 1) % 3
    except ValueError:
        return None


def _ch_to_display(ch):
    """Canonical display string from any supported spec. Returns '0' for
    off, or the original positive integer if the input was numeric, else
    falls back to axis+1 for R/G/B legacy."""
    axis = _ch_to_axis(ch)
    if axis is None:
        return "0"
    # Preserve user-typed number when it was already numeric (e.g. '4' stays '4')
    if ch is not None:
        try:
            n = int(str(ch).strip())
            if n > 0:
                return str(n)
        except (ValueError, TypeError):
            pass
    return str(axis + 1)


def _build_pastel_rgb(arr_norm, rgb_assignment, intensity=0.45, tints=None):
    """Combined RGB image.

    NEW BEHAVIOR (when `tints` is given): each ref contributes its actual
    user-picked tint, weighted by its metric score. Pixels with weight 1 in
    one ref show that ref's color exactly; mixtures are weighted averages of
    the user's chosen colors. Pixels with little signal stay near white.

    LEGACY BEHAVIOR (when `tints` is None): subtractive R/G/B blending based
    on the channel assignment, ignoring tint. Kept for backwards
    compatibility but no longer used by the viewer or save_all.

    arr_norm: (n_ref, ny, nx) in 0..1.
    rgb_assignment: per ref, one of 1/"R", 2/"G", 3/"B", 0/"-". Refs with
                    channel 0/"-" are EXCLUDED from the combined image.
    tints: per-ref tint colors (length n_ref, each (r,g,b) in 0..1). When
           provided, this is the color each ref contributes.
    """
    n_ref = arr_norm.shape[0]
    ny, nx = arr_norm.shape[1], arr_norm.shape[2]

    if tints is not None:
        # ---- Weighted blend of user-picked tints ----
        weighted = np.zeros((ny, nx, 3))
        total_w = np.zeros((ny, nx))
        for i in range(n_ref):
            if i >= len(rgb_assignment):
                break
            if _ch_to_axis(rgb_assignment[i]) is None:
                continue  # ch=0 -> exclude
            w = np.clip(arr_norm[i], 0.0, 1.0)
            tint = tints[i] if i < len(tints) else (0.5, 0.5, 0.5)
            for c in range(3):
                weighted[:, :, c] += w * float(tint[c])
            total_w += w
        out = np.ones((ny, nx, 3))
        mask = total_w > 1e-6
        for c in range(3):
            out[:, :, c] = np.where(
                mask,
                weighted[:, :, c] / np.maximum(total_w, 1e-6),
                1.0,
            )
        # Blend toward white when total signal is small so "no signal"
        # regions stay white instead of being colored by a noisy average.
        alpha = np.clip(total_w, 0.0, 1.0)[:, :, None]
        out = alpha * out + (1.0 - alpha) * 1.0
        return np.clip(out, 0.0, 1.0)

    # ---- Legacy R/G/B subtractive blending ----
    rgb = np.ones((ny, nx, 3), dtype=float)
    for i in range(n_ref):
        if i >= len(rgb_assignment):
            break
        ch = _ch_to_axis(rgb_assignment[i])
        if ch is None:
            continue
        v = np.clip(arr_norm[i], 0.0, 1.0)
        for other in (0, 1, 2):
            if other != ch:
                rgb[:, :, other] -= intensity * v
    return np.clip(rgb, 0.0, 1.0)


def _save_combined_rgb(arr_norm, refs, rgb_assignment, extent,
                       out_path, raw_path, title, tints=None,
                       dpi=600):
    """Publication-quality combined RGB. One file per call.
    Arial 18pt, thick spines, long ticks, 600 DPI by default."""
    rgb = _build_pastel_rgb(arr_norm, rgb_assignment, tints=tints)
    ny, nx = rgb.shape[0], rgb.shape[1]

    # Build legend with actual tint swatches when available (so a viewer
    # immediately reads off "this color = this compound" without having to
    # mentally translate ch numbers into the user's chosen palette).
    legend_handles = []
    import matplotlib.patches as mpatches
    for i, r in enumerate(refs):
        if i >= len(rgb_assignment):
            break
        axis = _ch_to_axis(rgb_assignment[i])
        if axis is None:
            continue
        disp = _ch_to_display(rgb_assignment[i])
        tint = (tints[i] if (tints is not None and i < len(tints))
                else None)
        label = f"ch{disp}: {r['name']}"
        if tint is not None:
            legend_handles.append(
                mpatches.Patch(facecolor=tint, edgecolor="#444", label=label)
            )
    legend_lines = None  # we use handle-based legend below if any

    # Annotated polish version — figsize follows the test map's aspect
    # ratio; Arial 18pt + thick spines + long ticks + 600 DPI.
    fig_size = _figsize_for_map(rgb.shape[:2], base_long=11.0,
                                 extra_w=2.4, min_short=6.0)
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(rgb, extent=extent, origin="upper",
              aspect="equal", interpolation="nearest")
    ax.set_title(title, fontweight="bold", pad=14)
    ax.set_xlabel("X (um)", labelpad=8)
    ax.set_ylabel("Y (um)", labelpad=8)
    for s in ax.spines.values():
        s.set_linewidth(1.8)
    ax.tick_params(width=1.6, length=7)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right",
                  framealpha=0.92, edgecolor="#444")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    # Raw upscaled version (no axes / labels) — only if a path was given
    if raw_path is not None:
        pixel_scale = 20
        raw_dpi = 600
        out_w = max(nx * pixel_scale, 1)
        out_h = max(ny * pixel_scale, 1)
        fig2 = plt.figure(figsize=(out_w / raw_dpi, out_h / raw_dpi),
                           dpi=raw_dpi)
        ax2 = fig2.add_axes([0, 0, 1, 1])
        ax2.set_axis_off()
        ax2.imshow(rgb, origin="upper", aspect="equal", interpolation="nearest")
        fig2.savefig(raw_path, dpi=raw_dpi, facecolor="white")
        plt.close(fig2)


def _save_histograms(out_path, ref_name, corr_pixels, cos_pixels):
    """2-panel histogram: Pearson correlation + cosine similarity.
    Pastel bar colors with the standard red mean-line."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    corr_mean = float(np.nanmean(corr_pixels))
    axes[0].hist(corr_pixels, bins=40, color="#a8c4e0",
                 edgecolor="#557291", linewidth=0.4)
    axes[0].axvline(corr_mean, color="#d24a4a", lw=2, ls="--",
                    label=f"Mean: {corr_mean:.3f}")
    axes[0].set_title("Correlation Distribution")
    axes[0].set_xlabel("Pearson Correlation (r)")
    axes[0].set_ylabel("Pixel Count")
    axes[0].legend()

    cos_mean = float(np.nanmean(cos_pixels))
    axes[1].hist(cos_pixels, bins=40, color="#f0c98c",
                 edgecolor="#a07033", linewidth=0.4)
    axes[1].axvline(cos_mean, color="#d24a4a", lw=2, ls="--",
                    label=f"Mean: {cos_mean:.3f}")
    axes[1].set_title("Cosine Similarity Distribution")
    axes[1].set_xlabel("Cosine Similarity")
    axes[1].set_ylabel("Pixel Count")
    axes[1].legend()

    fig.suptitle(f"Distribution — {ref_name}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def _save_statistics_summary(out_path, refs, result, peak_intensity_stats):
    """statistics_summary.csv — one row per reference, columns matching the
    previous tool's output."""
    cols = [
        "Reference",
        "NNLS_Coeff_Mean", "NNLS_Coeff_Std",
        "NNLS_Contrib_Mean", "NNLS_Contrib_Std",
        "NNLS_Contrib_Min", "NNLS_Contrib_Max",
        "CLS_Coeff_Mean", "CLS_Coeff_Std",
        "CLS_Contrib_Mean", "CLS_Contrib_Std",
        "CLS_Negative_Pct",
        "MCR_Conc_Mean", "MCR_Conc_Std",
        "MCR_Contrib_Mean", "MCR_Contrib_Std",
        "Corr_Mean", "Corr_Std", "Corr_Min", "Corr_Max", "Corr_Median",
        "Cosine_Mean", "Cosine_Std", "Cosine_Min", "Cosine_Max", "Cosine_Median",
        "Peak_Wavenumber_cm-1",
        "Peak_Intensity_Mean", "Peak_Intensity_Std", "Peak_RSD_%",
        "N_Pixels_Total",
    ]
    n_ref = len(refs)
    n_pix_total = int(np.prod(result["test_cube"].shape[:2]))

    rows = []
    for i, r in enumerate(refs):
        nnls_raw = result["nnls_raw"][i].ravel()
        nnls_norm = result["nnls_norm"][i].ravel()
        cls_raw = result["cls_raw"][i].ravel()
        cls_norm = result["cls_norm"][i].ravel()
        mcr_conc = result["mcr_conc"][i].ravel()
        mcr_contrib = result["mcr_contrib"][i].ravel()
        corr = result["corr_maps"][i].ravel()
        cos = result["cos_maps"][i].ravel()

        peak_wn, peak_mean, peak_std, peak_rsd = peak_intensity_stats[i]

        rows.append([
            r["name"],
            f"{nnls_raw.mean():.4f}", f"{nnls_raw.std():.4f}",
            f"{nnls_norm.mean():.4f}", f"{nnls_norm.std():.4f}",
            f"{nnls_norm.min():.4f}", f"{nnls_norm.max():.4f}",
            f"{cls_raw.mean():.4f}", f"{cls_raw.std():.4f}",
            f"{cls_norm.mean():.4f}", f"{cls_norm.std():.4f}",
            f"{float(result['cls_neg_pct'][i]):.1f}",
            f"{mcr_conc.mean():.4f}", f"{mcr_conc.std():.4f}",
            f"{mcr_contrib.mean():.4f}", f"{mcr_contrib.std():.4f}",
            f"{corr.mean():.4f}", f"{corr.std():.4f}",
            f"{corr.min():.4f}", f"{corr.max():.4f}", f"{np.median(corr):.4f}",
            f"{cos.mean():.4f}", f"{cos.std():.4f}",
            f"{cos.min():.4f}", f"{cos.max():.4f}", f"{np.median(cos):.4f}",
            f"{peak_wn:.1f}",
            f"{peak_mean:.2f}", f"{peak_std:.2f}", f"{peak_rsd:.2f}",
            str(n_pix_total),
        ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")


def _save_cv_confusion_png(cv_result, out_path):
    """Plot K-fold confusion matrix with annotations + F1 summary."""
    cm = cv_result["confusion"]
    names = cv_result["ref_names"]
    n = len(names)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = cm / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(3.0 + n * 1.7, 3.0 + n * 1.3))
    im = ax.imshow(norm, cmap=PASTEL_CMAPS["PastelGreen"], vmin=0.0, vmax=1.0)
    for i in range(n):
        for j in range(n):
            text_color = "white" if norm[i, j] > 0.6 else "#222"
            # Top: pixel count (n=...) — NOT a percentage.
            # Bottom: row-normalized fraction (this is the accuracy fraction).
            ax.text(j, i,
                    f"n={cm[i, j]}\n({norm[i, j] * 100:.1f}%)",
                    ha="center", va="center", color=text_color)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"Reference {cv_result['n_folds']}-fold CV (NNLS argmax)\n"
        f"Accuracy = {cv_result['accuracy']:.3f}, "
        f"Macro F1 = {cv_result['macro_f1']:.3f}",
        fontweight="bold",
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized fraction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_noise_sweep_png(sweep_result, ref_names, tints, out_path):
    """RMSE vs noise level curve, one line per ref + overall."""
    noise = np.array(sweep_result["noise_levels"]) * 100  # %
    rmse_pr = sweep_result["rmse_per_ref"]
    overall = sweep_result["overall_rmse"]
    n = len(ref_names)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(n):
        ax.plot(noise, rmse_pr[:, i], "o-",
                color=tints[i % len(tints)], lw=2.0, ms=8,
                label=f"{ref_names[i]}")
    ax.plot(noise, overall, "s--", color="#444", lw=2.4, ms=8,
            label="overall")
    ax.set_xlabel("Noise level (% of signal std)")
    ax.set_ylabel("NNLS unmixing RMSE  (true vs predicted weight)")
    ax.set_title("Synthetic noise sweep — degradation curve\n"
                 "(higher noise = more uncertainty in compound contribution)",
                 fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_xlim(0, noise.max() * 1.05)
    ax.set_ylim(0, max(rmse_pr.max(), overall.max()) * 1.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def _save_synth_scatter_png(synth_result, ref_names, tints, out_path):
    n = len(ref_names)
    fig_w = max(5.5 * n, 6.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 5.5))
    if n == 1:
        axes = [axes]
    true_w = synth_result["true_weights"]
    pred = synth_result["pred_norm"]
    for i, name in enumerate(ref_names):
        ax = axes[i]
        ax.scatter(true_w[:, i], pred[:, i], s=18, alpha=0.6,
                   color=tints[i % len(tints)],
                   edgecolor="#444", linewidth=0.3)
        ax.plot([0, 1], [0, 1], color="#888", ls="--", lw=1.2, alpha=0.7)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(f"True weight")
        ax.set_ylabel(f"Predicted weight")
        ax.set_title(
            f"{name}\nRMSE={synth_result['rmse_per_ref'][i]:.3f}, "
            f"r={synth_result['correlation_per_ref'][i]:.3f}",
            fontweight="bold",
        )
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"Synthetic mixture — BEST CASE (ideal linear, "
        f"N={synth_result['n_synth']}, noise={synth_result['noise_level']*100:.0f}%)  "
        f"overall RMSE = {synth_result['overall_rmse']:.3f}\n"
        f"(real measurements degrade more — see the noise sweep PNG)",
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_validation_csv(out_path, cv_result, synth_result, ref_names):
    """Combined classifier validation report."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Section,Metric,Value\n")
        if cv_result is not None:
            f.write(f"K-fold CV,Folds,{cv_result['n_folds']}\n")
            f.write(f"K-fold CV,Test pixels (sum across folds),"
                    f"{cv_result['n_test_pixels']}\n")
            f.write(f"K-fold CV,Accuracy,{cv_result['accuracy']:.4f}\n")
            f.write(f"K-fold CV,Macro F1,{cv_result['macro_f1']:.4f}\n")
            for i, name in enumerate(cv_result["ref_names"]):
                f.write(f"K-fold CV,Precision ({name}),"
                        f"{cv_result['precision'][i]:.4f}\n")
                f.write(f"K-fold CV,Recall ({name}),"
                        f"{cv_result['recall'][i]:.4f}\n")
                f.write(f"K-fold CV,F1 ({name}),"
                        f"{cv_result['f1'][i]:.4f}\n")
            # Raw confusion matrix at the end
            f.write(f"K-fold CV,Confusion matrix headers,"
                    f"{';'.join(cv_result['ref_names'])}\n")
            for i, name in enumerate(cv_result["ref_names"]):
                row_str = ";".join(str(int(v)) for v in cv_result['confusion'][i])
                f.write(f"K-fold CV,Confusion row ({name}),{row_str}\n")
        if synth_result is not None:
            f.write(f"Synthetic mixture,N samples,{synth_result['n_synth']}\n")
            f.write(f"Synthetic mixture,Noise level,"
                    f"{synth_result['noise_level']}\n")
            f.write(f"Synthetic mixture,Overall RMSE (NNLS),"
                    f"{synth_result['overall_rmse']:.4f}\n")
            for i, name in enumerate(ref_names):
                f.write(f"Synthetic mixture,RMSE ({name}),"
                        f"{synth_result['rmse_per_ref'][i]:.4f}\n")
                f.write(f"Synthetic mixture,Correlation r ({name}),"
                        f"{synth_result['correlation_per_ref'][i]:.4f}\n")


def _save_reliability_summary(out_path, result, refs):
    """One-pager reliability summary CSV.

    Columns: Metric, Value, Notes — easy to read in Excel.
    Sections covered:
      - Confidence (NNLS gap, NNLS entropy, MCR gap, MCR entropy)
      - Agreement across NNLS / MCR / CLS
      - Per-ref argmax pixel counts (each method)
    """
    n_pix = result["consensus"].size
    summ = result["agreement_summary"]
    ref_names = [r["name"] for r in refs]

    rows = []
    rows.append(("Section", "Total pixels", str(n_pix)))
    rows.append(("Confidence (NNLS)",
                 "Mean gap top1-top2",
                 f"{float(result['confidence_gap_nnls'].mean()):.4f}"))
    rows.append(("Confidence (NNLS)",
                 "Median gap",
                 f"{float(np.median(result['confidence_gap_nnls'])):.4f}"))
    rows.append(("Confidence (NNLS)",
                 "Mean entropy (bits)",
                 f"{float(result['entropy_nnls'].mean()):.4f}"))
    rows.append(("Confidence (MCR)",
                 "Mean gap top1-top2",
                 f"{float(result['confidence_gap_mcr'].mean()):.4f}"))
    rows.append(("Confidence (MCR)",
                 "Mean entropy (bits)",
                 f"{float(result['entropy_mcr'].mean()):.4f}"))
    rows.append(("Agreement (NNLS/MCR/CLS)",
                 "3/3 agree pixels",
                 f"{summ['full']} ({100*summ['full']/n_pix:.2f}%)"))
    rows.append(("Agreement (NNLS/MCR/CLS)",
                 "2/3 agree pixels",
                 f"{summ['two']} ({100*summ['two']/n_pix:.2f}%)"))
    rows.append(("Agreement (NNLS/MCR/CLS)",
                 "1/3 (no majority) pixels",
                 f"{summ['split']} ({100*summ['split']/n_pix:.2f}%)"))

    # Per-method argmax breakdown
    for tag, key in [("NNLS", "argmax_nnls"),
                     ("MCR",  "argmax_mcr"),
                     ("CLS",  "argmax_cls")]:
        amx = result[key].ravel()
        counts = np.bincount(amx, minlength=len(refs))
        for i, name in enumerate(ref_names):
            rows.append((
                f"Argmax pixel counts ({tag})",
                name,
                f"{int(counts[i])} ({100*counts[i]/n_pix:.2f}%)",
            ))

    # Consensus breakdown
    cons = result["consensus"].ravel()
    n_tie = int((cons < 0).sum())
    rows.append(("Consensus label",
                 "tie / no majority",
                 f"{n_tie} ({100*n_tie/n_pix:.2f}%)"))
    for i, name in enumerate(ref_names):
        c = int((cons == i).sum())
        rows.append(("Consensus label", name,
                     f"{c} ({100*c/n_pix:.2f}%)"))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Section,Metric,Value\n")
        for sec, metric, val in rows:
            f.write(f"{sec},{metric},{val}\n")


def _save_pixel_metric_csv(path, arr3d, x_coords, y_coords, ref_names, prefix):
    """Flatten (n_ref, ny, nx) into a CSV: x, y, prefix_ref1, prefix_ref2, ..."""
    n_ref, ny, nx = arr3d.shape
    cols = ["x", "y"] + [f"{prefix}_{n}" for n in ref_names]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for yi in range(ny):
            for xi in range(nx):
                row = [f"{x_coords[xi]}", f"{y_coords[yi]}"]
                for r_i in range(n_ref):
                    row.append(f"{arr3d[r_i, yi, xi]:.6g}")
                f.write(",".join(row) + "\n")
    print(f"  {path.name}")


# =============================================================================
# Main
# =============================================================================

def main():
    dlg = LauncherDialog()
    config = dlg.run()
    if config is None:
        print("Cancelled.")
        return

    result = process(config)
    show_viewer(result)


if __name__ == "__main__":
    main()
