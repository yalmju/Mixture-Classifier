"""Reading one Raman map of the ink: composition and apparent concentration.

The whole algorithm of the paper in one file. Nothing here imports the app
(dl_model.py, unmix.py, sers_mixture.py, …); only numpy, scipy and scikit-learn.

    one map
      │  cut to 500–2500 cm⁻¹ · bridge detector plateaus
      │
      ├─ gate             every pixel:  non-negative least squares against the class-mean spectra
      │                   → the pixels that carry analyte; ① and ② read these
      │
      │  remove the ALS baseline · clip at 0
      │
      ├─ ① composition    every pixel:  ln(1 + spectrum) → PLS (8 latent variables)
      │                   → shares of DQ, TBZ, THI and bare ink, clipped at 0, summing to 1
      │                   → mean over the pixels → DQ : TBZ : THI
      │
      └─ ② concentration  the map:  ln(1 + mean spectrum) − ln(1 + ink band at 2137 cm⁻¹),
                          500–2000 cm⁻¹ → PLS (8 latent variables) → log10 C of each component
                          → apparent µM, fitted on maps with every component at 3–24 µM

Use
    model = load_model(".../pls_composition_261110.dlm")     # the saved weights
    templates = class_templates({"DQ": [...], "TBZ": [...], "THI": [...], "BLK": [...], ...})
    result = read_map(*load_map("map.csv"), model, templates)
    result["composition"], result["uM"]

Fitting from maps is `training_table` → `fit`; `held_out` scores a stage with whole
conditions left out. `paper_algorithm_check.py` shows that this file gives the same
weights and the same numbers as the pipeline that produced the saved model.
"""
import csv
import hashlib
import os
import pickle

import numpy as np
from scipy.optimize import nnls
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve
from sklearn.cluster import MiniBatchKMeans
from sklearn.cross_decomposition import PLSRegression

ANALYTES = ("DQ", "TBZ", "THI")
BLANK = "BLK"                      # bare ink: the fourth output of the composition stage
RANGE_CM = (500.0, 2500.0)         # spectral range both stages see
CONC_RANGE_CM = (500.0, 2000.0)    # the concentration stage regresses on the fingerprint part
INK_CM = 2137.0                    # ink band used as the internal reference
BAND_HALF_WIDTH_CM = 10.0          # a band height is the maximum within ± this
N_LATENT = 8                       # PLS latent variables, both stages
WINDOW_UM = (3.0, 24.0)            # the concentration stage is fitted on maps inside this
PIXELS_PER_MAP = 100               # pixels a map contributes to a fit
MAX_CLIPPED = 0.01                 # a pixel with more clipped channels than this is left out of fits
GATE_MIN_SHARE = 0.15              # gate: least-squares analyte share a pixel needs
MIN_ANALYTE_SHARE = 0.5            # ②: pixels whose PLS analyte share is below this are left out
UM_LIMITS = (1e-3, 5e3)            # apparent µM is reported inside these bounds


# ─────────────────────────────────────────────────────────────── map file ──

def load_map(path):
    """Map CSV → (wavenumbers, spectra (n_pixels, n_wavenumbers)).

    Row 3 holds the wavenumber axis from the third column on; each later row is
    X, Y, intensities…"""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    wn = np.array([float(v) for v in rows[2][2:] if v.strip() != ""])
    data = []
    for row in rows[3:]:
        cells = [v for v in row if str(v).strip() != ""]
        if len(cells) >= len(wn) + 2:
            data.append([float(v) for v in cells[2:2 + len(wn)]])
    return wn, np.asarray(data, float)


def _between(wn, limits):
    wn = np.asarray(wn, float)
    return (wn >= limits[0]) & (wn <= limits[1])


# ────────────────────────────────────────────────────────── preprocessing ──

_PENALTY = {}


def als_baseline(y, lam=1e5, p=0.01, n_iter=10):
    """Asymmetric least squares baseline (Eilers & Boelens); subtract it from ``y``."""
    n = len(y)
    if (n, lam) not in _PENALTY:
        d = diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(n, n - 2), dtype=float)
        _PENALTY[n, lam] = csc_matrix(lam * (d @ d.T))
    w, z = np.ones(n), y
    for _ in range(n_iter):
        z = spsolve(csc_matrix(diags(w, 0, dtype=float) + _PENALTY[n, lam]), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def desaturate(spectra, min_run=3, top_frac=0.98):
    """Bridge detector-clipped stretches with a straight line between their shoulders.

    Two signatures: a flat top at the spectrum's own maximum, and a strong block that
    zig-zags from channel to channel (what a plateau looks like after an earlier
    baseline removal). Returns (repaired spectra, fraction of clipped channels per spectrum)."""
    X = np.array(spectra, float, order="C")            # row-contiguous: the pixel choice below sums rows
    n, length = X.shape
    clipped_fraction = np.zeros(n)

    def bridge(y, a, b, top):
        lo = y[a - 1] if a > 0 else y[b + 1] if b + 1 < length else top
        hi = y[b + 1] if b + 1 < length else lo
        y[a:b + 1] = np.linspace(lo, hi, b - a + 1)

    for i in range(n):
        y = X[i]
        top = float(y.max())
        if top <= 0 or top < 2.5 * max(float(np.median(y)), 1e-12):
            continue                                   # no peak towers over the spectrum body
        flat = np.flatnonzero(y >= top * top_frac)
        if len(flat) < min_run:
            continue
        clipped = np.zeros(length, bool)
        for run in np.split(flat, np.where(np.diff(flat) > 3)[0] + 1):
            a, b = int(run[0]), int(run[-1])           # dips of 1–2 channels belong to the plateau
            if b - a + 1 < min_run:
                continue                               # a sharp real peak
            bridge(y, a, b, top)
            clipped[a:b + 1] = True
        top = float(y.max())
        strong = np.flatnonzero(y > 0.5 * top)
        if len(strong) >= min_run + 2:
            for run in np.split(strong, np.where(np.diff(strong) > 1)[0] + 1):
                if len(run) < min_run + 2:
                    continue
                d = np.diff(y[run[0]:run[-1] + 1])
                big = np.abs(d) > 0.12 * top
                if int(((d[1:] * d[:-1] < 0) & big[1:] & big[:-1]).sum()) < 3:
                    continue                           # real bands do not alternate like this
                a, b = int(run[0]), int(run[-1])
                bridge(y, a, b, top)
                clipped[a:b + 1] = True
        clipped_fraction[i] = clipped.mean()
    return X, clipped_fraction


def prepare(wn, cube):
    """Every pixel of a map, as the stages read it: cut to 500–2500 cm⁻¹, plateaus
    bridged, ALS baseline removed, clipped at 0. Returns (wavenumbers, spectra)."""
    wn = np.asarray(wn, float)
    keep = _between(wn, RANGE_CM)
    S = desaturate(np.asarray(cube, float)[:, keep])[0]
    return wn[keep], np.clip(np.stack([y - als_baseline(y) for y in S]), 0, None)


def map_seed(map_id, seed=0):
    """The same map always gets the same representative pixels."""
    key = os.path.normcase(os.path.normpath(str(map_id or "map"))).encode("utf-8", "replace")
    return int(seed) ^ int.from_bytes(hashlib.blake2s(key, digest_size=4).digest(), "little")


def representative_pixels(cube, n_px, seed=0):
    """Indices of ``n_px`` measured pixels that span the map: one per k-means cluster of
    spectral shape (32 bins, unit length) and log total intensity. A map with no more
    than ``n_px`` pixels keeps them all."""
    cube = np.ascontiguousarray(cube, float)
    n = len(cube)
    k = min(max(1, int(n_px)), n)
    if k >= n:
        return np.arange(n)
    bins = np.array_split(np.arange(cube.shape[1]), min(32, cube.shape[1]))
    shape = np.column_stack([cube[:, b].mean(axis=1) for b in bins])
    shape /= np.linalg.norm(shape, axis=1, keepdims=True) + 1e-12
    total = np.log1p(np.clip(cube.sum(axis=1), 0, None))
    z = (total - np.median(total)) / (np.std(total) + 1e-12)
    feat = np.column_stack([shape, 0.35 * z])          # keep this order of operations: the choice
                                                       # of pixels changes with the last digit
    km = MiniBatchKMeans(n_clusters=k, random_state=int(seed), n_init=3, batch_size=min(512, n),
                         max_iter=100, reassignment_ratio=0.0)
    labels = km.fit_predict(feat)
    chosen = []
    for j in range(k):                                 # the pixel nearest each cluster centre
        idx = np.where(labels == j)[0]
        if len(idx):
            chosen.append(int(idx[np.argmin(((feat[idx] - km.cluster_centers_[j]) ** 2).sum(axis=1))]))
    chosen = list(dict.fromkeys(chosen))
    while len(chosen) < k:                             # empty clusters: add the farthest pixel left
        rest = np.setdiff1d(np.arange(n), chosen)
        if not len(rest):
            break
        d2 = ((feat[rest, None, :] - feat[chosen][None, :, :]) ** 2).sum(axis=2).min(axis=1)
        chosen.append(int(rest[np.argmax(d2)]))
    return np.asarray(sorted(chosen, key=lambda i: (-total[i], i)), int)


def training_spectra(wn, cube, map_id, n_px=PIXELS_PER_MAP, seed=0):
    """The pixels a map contributes to a fit: as `prepare`, but pixels with more than 1 %
    clipped channels are left out and at most ``n_px`` representative pixels are kept."""
    wn = np.asarray(wn, float)
    keep = _between(wn, RANGE_CM)
    S, clipped = desaturate(np.asarray(cube, float)[:, keep])
    ok = clipped <= MAX_CLIPPED
    if ok.any() and not ok.all():
        S = S[ok]
    S = S[representative_pixels(S, n_px, map_seed(map_id, seed))[:n_px]]
    return wn[keep], np.clip(np.stack([y - als_baseline(y) for y in S]), 0, None)


# ─────────────────────────────────────────────────────────────────── gate ──

def class_templates(maps_by_class):
    """{class: [map paths]} → {class: mean raw spectrum over every pixel of its maps}.
    The classes are the pure analytes and the backgrounds (bare ink, …)."""
    return {name: np.vstack([load_map(p)[1] for p in paths]).mean(axis=0)
            for name, paths in maps_by_class.items()}


def _unit(X):
    X = np.clip(np.atleast_2d(np.asarray(X, float)), 0, None)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def gate(wn, cube, templates, analytes=ANALYTES, min_share=GATE_MIN_SHARE):
    """Which pixels carry analyte. Each raw spectrum (500–2500 cm⁻¹, plateaus bridged,
    unit length, no baseline removal) is fitted by non-negative least squares with the
    class-mean spectra; a pixel passes when the analytes hold at least ``min_share`` of
    the fitted amounts. ``templates`` are on the same wavenumber axis as the map."""
    keep = _between(wn, RANGE_CM)
    X = _unit(desaturate(np.asarray(cube, float)[:, keep])[0])
    names = list(templates)
    T = _unit(np.array([templates[n] for n in names], float)[:, keep])
    A = np.array([nnls(T.T, x)[0] for x in X])
    total = A.sum(axis=1)
    analyte = A[:, [n in analytes for n in names]].sum(axis=1)
    return np.divide(analyte, total, out=np.zeros_like(total), where=total > 0) >= min_share


# ──────────────────────────────────────────────────────────── PLS weights ──

def _weights(pls, n_features):
    """A fitted PLS regression as the linear map it is: prediction = x @ B + b0."""
    b0 = np.asarray(pls.predict(np.zeros((1, n_features))), float).reshape(-1)
    B = np.asarray(pls.predict(np.eye(n_features)), float).reshape(n_features, -1) - b0
    return B, b0


# ──────────────────────────────────────────────────────── ① composition ──

def fit_composition(X, Y, n_latent=N_LATENT):
    """PLS regression of the shares ``Y`` (rows sum to 1; analytes, then bare ink) on
    ln(1 + spectrum) of the prepared pixel spectra ``X``. Returns (B, b0)."""
    F = np.log1p(np.clip(X, 0, None)).astype(np.float32)
    k = max(1, min(int(n_latent), len(F) - 1, F.shape[1]))
    pls = PLSRegression(n_components=k).fit(F, np.asarray(Y, np.float32))
    return _weights(pls, F.shape[1])


def pixel_shares(weights, X):
    """Prepared pixel spectra → shares per pixel, clipped at 0 and summing to 1."""
    B, b0 = weights
    p = np.clip(np.log1p(np.clip(np.atleast_2d(X), 0, None)) @ B + b0, 0, None)
    return p / (p.sum(axis=1, keepdims=True) + 1e-12)


def map_composition(shares, n_analytes=len(ANALYTES)):
    """Shares of the pixels of one map → the analytes' composition, summing to 1."""
    p = np.clip(np.atleast_2d(shares).mean(axis=0)[:n_analytes], 0, None)
    return p / max(p.sum(), 1e-9)


def pixel_composition(shares, n_analytes=len(ANALYTES)):
    """Shares of each pixel → that pixel's composition of the analytes, summing to 1
    (what a composition image shows pixel by pixel)."""
    p = np.clip(np.atleast_2d(shares)[:, :n_analytes], 0, None)
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


# ─────────────────────────────────────────────────────── ② concentration ──

def band_height(wn, X, centre, half_width=BAND_HALF_WIDTH_CM):
    """Height of a band per spectrum: the maximum within ± ``half_width`` of ``centre``."""
    wn = np.asarray(wn, float)
    m = np.abs(wn - centre) <= half_width
    if not m.any():
        m[np.argmin(np.abs(wn - centre))] = True
    return np.clip(np.atleast_2d(X), 0, None)[:, m].max(axis=1)


def ink_normalised_spectrum(wn, X):
    """Prepared pixel spectra of one map → the one row the concentration stage reads:
    ln(1 + mean spectrum) − ln(1 + mean height of the ink band)."""
    X = np.clip(np.atleast_2d(X), 0, None)
    return np.log1p(X.mean(axis=0)) - np.log1p(band_height(wn, X, INK_CM).mean())


def fit_concentration(wn, F, C_uM, n_latent=N_LATENT):
    """PLS regression of log10 µM (one column per analyte, all > 0) on the map rows ``F``,
    using 500–2000 cm⁻¹ only. Returns (B, b0); B is zero outside that range, so the
    weights apply to the whole row."""
    F = np.asarray(F, float)
    keep = _between(wn, CONC_RANGE_CM)
    k = int(min(n_latent, len(F) - 1, keep.sum()))
    pls = PLSRegression(n_components=k, scale=False).fit(F[:, keep], np.log10(np.asarray(C_uM, float)))
    Bk, b0 = _weights(pls, int(keep.sum()))
    B = np.zeros((F.shape[1], len(b0)))
    B[keep] = Bk
    return B, b0


def concentration(weights, F):
    """Map rows → apparent µM of each analyte."""
    B, b0 = weights
    return np.clip(10.0 ** (np.atleast_2d(F) @ B + b0), *UM_LIMITS)


# ───────────────────────────────────────────────────────── the whole read ──

def load_model(path):
    """The saved paper model (.dlm, a pickle) → the weights of both stages."""
    with open(path, "rb") as fh:
        m = pickle.load(fh)
    conc = np.asarray(m["uM"]["coef"], float)          # last row is the intercept
    return {"analytes": list(m["uM"]["subs"]), "outputs": list(m["subs"]),
            "composition": _weights(m["sk"], int(m["n_feat"])),
            "concentration": (conc[:-1], conc[-1])}


def read_map(wn, cube, model, templates=None):
    """One map → composition and apparent µM.

    ``templates`` are the class-mean spectra of the gate (see `class_templates`);
    without them the gate is skipped and every pixel is read."""
    ax, X = prepare(wn, cube)
    hit = (gate(wn, cube, templates, model["analytes"]) if templates is not None
           else np.ones(len(X), bool))
    return read_pixels(ax, X, hit, model)


def read_pixels(ax, X, hit, model):
    """The prepared pixels of one map and their gate → composition and apparent µM."""
    names = model["analytes"]
    shares = pixel_shares(model["composition"], X)
    if not hit.any():
        return {"composition": None, "uM": None, "pixel_shares": shares, "gate": hit}
    comp = map_composition(shares[hit], len(names))
    use = hit & (shares[:, :len(names)].sum(axis=1) >= MIN_ANALYTE_SHARE)
    if not use.any():
        use = hit
    um = concentration(model["concentration"], ink_normalised_spectrum(ax, X[use]))[0]
    return {"composition": dict(zip(names, comp)), "uM": dict(zip(names, um)),
            "pixel_shares": shares, "gate": hit}


# ───────────────────────────────────────────────────────────────── fitting ──

def training_table(samples, analytes=ANALYTES, progress=None):
    """Maps → the pixel table both stages are fitted on.

    ``samples`` is a list of (path, label): a dict {analyte: µM} for a mixture map, an
    analyte name for a pure-substance map, None for a bare-ink map.

    Returns a dict with
      wn     wavenumbers (500–2500 cm⁻¹)
      X      prepared pixel spectra
      Y      shares per pixel: analytes, then bare ink
      map    index into ``paths`` per pixel
      paths  the maps
      uM     µM per map (nan where the map has no concentration label)
      group  per map, what is held out together: mixtures of one composition share a
             group, every pure or bare-ink map is its own"""
    analytes = list(analytes)
    X, Y, owner, paths, uM, group, ax = [], [], [], [], [], [], None
    for k, (path, label) in enumerate(samples):
        if progress:
            progress(f"map {k + 1}/{len(samples)}")
        ax, S = training_spectra(*load_map(path), map_id=path)
        c = np.full(len(analytes), np.nan)
        if isinstance(label, dict):
            c = np.array([float(label.get(a, 0.0)) for a in analytes])
            y = np.r_[c / c.sum(), 0.0]
            g = "r:" + ",".join(f"{v:.6f}" for v in y[:-1])
        else:
            y = np.array([float(a == label) for a in analytes] + [float(label is None)])
            g = path
        X.append(S); Y.append(np.tile(y, (len(S), 1))); owner += [k] * len(S)
        paths.append(path); uM.append(c); group.append(g)
    return {"wn": ax, "X": np.vstack(X), "Y": np.vstack(Y), "map": np.asarray(owner),
            "paths": paths, "uM": np.array(uM), "group": np.array(group, object),
            "analytes": analytes}


def map_rows(table):
    """The concentration stage's input row for every map of a training table."""
    return np.array([ink_normalised_spectrum(table["wn"], table["X"][table["map"] == i])
                     for i in range(len(table["paths"]))])


def in_window(uM, window=WINDOW_UM):
    """Maps whose every component lies inside the fitting window."""
    uM = np.asarray(uM, float)
    with np.errstate(invalid="ignore"):
        return ((uM >= window[0]) & (uM <= window[1])).all(axis=1)


def fit(table):
    """Both stages from a training table → a model `read_map` can use."""
    F, ok = map_rows(table), in_window(table["uM"])
    return {"analytes": list(table["analytes"]), "outputs": list(table["analytes"]) + [BLANK],
            "composition": fit_composition(table["X"], table["Y"]),
            "concentration": fit_concentration(table["wn"], F[ok], table["uM"][ok])}


def held_out(fit_stage, predict, X, Y, groups):
    """Leave one group out at a time: refit on the rest, predict the group. One prediction per row."""
    groups = np.asarray(groups, object)
    P = np.zeros((len(X), np.shape(Y)[1]))
    for g in dict.fromkeys(groups.tolist()):
        out = groups == g
        P[out] = predict(fit_stage(X[~out], Y[~out]), X[out])
    return P


# ───────────────────────────────────────────────────────────────── scoring ──

def composition_error(read, applied):
    """½ Σ|read − applied| in percentage points: the share placed on the wrong component."""
    return 50.0 * np.abs(np.asarray(read, float) - np.asarray(applied, float)).sum(axis=-1)


def fold_error(read, applied):
    """max(read / applied, applied / read) per value; 2 means off by a factor of two."""
    read, applied = np.asarray(read, float), np.asarray(applied, float)
    return np.maximum(read / applied, applied / read)
