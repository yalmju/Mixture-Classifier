"""real_data.py — load the real pesticide SERS maps and run the full analysis.

UI-agnostic (numpy / scipy / sklearn only — no matplotlib backend set here, so it
is safe to import from the Qt app). Produces everything the UNMIXR "Real data"
page shows:

    - single-component 4-class confusion (DQ / THI / TBZ / BLK, per pixel)
    - multi-label detection on the ratio mixtures (outcome grid + micro F1)
    - combination confusion (component SET as one class)
    - response-factor calibration correcting THI over-representation
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

from sers_mixture import (preprocess, SERSMixtureClassifier, AugmentConfig,
                          multilabel_metrics, names_to_indicator)
from competitive import fit_B
from dataset import discover_dataset, ratio_map_path, load_mixtures, is_blank

PEST_DEFAULT = r"S:\Google Drive\내 드라이브\github\Pest_Discriminator"
COMPS = ["DQ", "THI", "TBZ"]
CLASSES4 = ["DQ", "THI", "TBZ", "BLK"]
REF_FILES = {"DQ": "DQ_corrected.csv", "THI": "THI_corrected.csv",
             "TBZ": "TBZ_corrected.csv", "BLK": "blk_corrected.csv"}
MIXES = {
    "DQ1TH1": [1, 1, 0], "DQ1TH3": [1, 3, 0], "DQ3TH1": [3, 1, 0],
    "TB1TH1": [0, 1, 1], "TB1TH3": [0, 3, 1], "TB3TH1": [0, 1, 3],
    "TBZ1DQ1": [1, 0, 1], "TBZ1DQ3": [3, 0, 1], "TBZ3DQ1": [1, 0, 3],
    "THI1": [1, 1, 1],
}


def load_map(path):
    """Return (wavenumbers, cube (n_pixels, n_wn), mean_spectrum, coords (n_pixels, 2)).

    Each data row is  X, Y, <intensities…>  — coords are kept so callers can do a
    spatial (block) train/test split instead of a leaky random pixel split."""
    rows = list(csv.reader(open(path)))
    wn = np.array([float(v) for v in rows[2][2:] if v.strip() != ""])
    n = len(wn)
    data, coords = [], []
    for r in rows[3:]:
        cells = [v for v in r if str(v).strip() != ""]
        if len(cells) < n + 2:
            continue
        coords.append((float(cells[0]), float(cells[1])))
        data.append([float(v) for v in cells[2:2 + n]])
    cube = np.asarray(data, float)
    return wn, cube, cube.mean(axis=0), np.asarray(coords, float)


def _combo(names, comps):
    order = [c for c in comps if c in names]
    return "+".join(order) if order else "(none)"


def per_pixel_vote(cube, pures, comps, frac_thr=0.15, pix_thr=0.15):
    """Detect a component if it is a meaningful fraction in >= pix_thr of pixels.

    The mean spectrum buries a minor compound; individual pixels where local
    coverage favours it still show it. Vote across the pixels."""
    Xp = preprocess(cube)
    votes = np.zeros(len(comps))
    for yv in Xp:
        B, _ = nnls(pures.T, yv)
        votes += (B / (B.sum() + 1e-12) >= frac_thr)
    return set(np.where(votes / len(Xp) >= pix_thr)[0])


def matched_significance(mean, pures, comps, k=9.0):
    """Keep component a if excluding it raises the NNLS residual by > noise (LOD)."""
    y = preprocess(mean[None, :])[0]
    B_full, _ = nnls(pures.T, y)
    resid = y - B_full @ pures
    noise = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
    keep = set()
    for a in range(len(comps)):
        others = [i for i in range(len(comps)) if i != a]
        Bo, _ = nnls(pures[others].T, y)
        r_wo = np.linalg.norm(y - Bo @ pures[others])
        r_full = np.linalg.norm(resid)
        if (r_wo - r_full) > k * noise:
            keep.add(a)
    return keep


def _score(true_lists, pred_sets, comps):
    """(recall, precision, f1, exact_composition) for index-set predictions."""
    n = len(comps)
    yt = np.array([[1 if comps[i] in t else 0 for i in range(n)]
                   for t in true_lists])
    yp = np.array([[1 if i in p else 0 for i in range(n)] for p in pred_sets])
    m = multilabel_metrics(yt, yp)
    exact = float(np.mean([set(np.where(a)[0]) == b for a, b in zip(yt, pred_sets)]))
    return m["micro_recall"], m["micro_precision"], m["micro_f1"], exact


@dataclass
class RealResult:
    cm4: np.ndarray
    acc4: float                      # spatial (honest) split accuracy
    acc4_random: float              # random pixel split (leaky, for comparison)
    classes4: list
    comps: list
    mix_names: list
    yt: np.ndarray
    yp: np.ndarray
    micro: dict
    combo_rows: list
    combo_cols: list
    combo_M: np.ndarray
    combo_exact: float
    strategies: list                 # (label, recall, precision, f1, exact)
    R: np.ndarray
    calib_rows: list                 # (name, present_idx, nom, raw, cal)
    err_raw: float
    err_cal: float
    pca_emb: np.ndarray = None       # (n_sample, 2)
    pca_lab: np.ndarray = None       # (n_sample,) class index 0-3
    pure_spectra: np.ndarray = None
    wn: np.ndarray = None


def _ratio_over(present_idx, vec, n):
    v = np.array([vec[i] if i in present_idx else 0.0 for i in range(n)])
    return v / (v.sum() + 1e-12)


def _response_factors(mix_names, B, mixes, comps):
    """Per-component response factors from equal-ratio (1:1) binary mixtures.

    For a binary mix of components i, j at equal nominal parts,
    log(B_i / B_j) = logR_i - logR_j. Solve least squares for logR (one component
    fixed as reference), then normalise by the median — so the result is
    independent of which component is the reference. Returns all-ones when there
    is not enough binary evidence (generalises the old 3-pesticide solver)."""
    n = len(comps)
    rows, rhs = [], []
    for k in mix_names:
        nominal = np.asarray(mixes[k], float)
        present = [i for i in range(n) if nominal[i] > 0]
        if len(present) != 2:
            continue
        i, j = present
        if nominal[i] != nominal[j]:
            continue
        bi, bj = B[k][i], B[k][j]
        if bi <= 0 or bj <= 0:
            continue
        row = np.zeros(n); row[i] = 1.0; row[j] = -1.0
        rows.append(row); rhs.append(np.log(bi / bj))
    if not rows:
        return np.ones(n)
    A = np.array(rows)
    logR, *_ = np.linalg.lstsq(A[:, 1:], np.array(rhs), rcond=None)  # col 0 = ref
    full = np.zeros(n); full[1:] = logR
    R = np.exp(full)
    return R / np.median(R)


def _resolve_dataset(pest_dir):
    """(classes, comps, blanks, path_of, mixes) for the data folder, grouping
    batches. Backwards-compatible: the pesticide example (DQ/THI/TBZ + blk, no
    mixtures.csv) yields comps in the canonical COMPS order with the built-in
    MIXES; any other set uses the discovered classes + a mixtures.csv manifest."""
    groups = discover_dataset(pest_dir)
    if len(groups) < 2:
        raise FileNotFoundError(
            f"need >= 2 reference classes in {pest_dir} (or its Reference/)."
            f"\nfound: {[c for c, _ in groups]}")
    path_of = {c: [p for _b, p, _r in maps] for c, maps in groups}
    classes = [c for c, _ in groups]
    comps_disc = [c for c in classes if not is_blank(c)]
    blanks = [c for c in classes if is_blank(c)]
    manifest = load_mixtures(pest_dir, comps_disc)
    if manifest is None and set(comps_disc) == set(COMPS):
        comps = list(COMPS)                      # canonical order + built-in mixes
        mixes = dict(MIXES)
        classes = comps + blanks
    else:
        comps = comps_disc
        mixes = manifest or {}
    return classes, comps, blanks, path_of, mixes


def compute_real(pest_dir=PEST_DEFAULT):
    classes, comps, blanks, path_of, mixes = _resolve_dataset(pest_dir)
    K = len(classes)

    # ---- load references (per-pixel cubes + coords), merging batch maps ----
    cubes, means, coords, wn = {}, {}, {}, None
    for c in classes:
        cbs, cds = [], []
        for p in path_of[c]:
            wn, cube, _m, coord = load_map(p)
            cbs.append(cube); cds.append(coord)
        cube = np.vstack(cbs); coord = np.vstack(cds)
        cubes[c] = cube; coords[c] = coord; means[c] = cube.mean(axis=0)
    pures = preprocess(np.vstack([means[c] for c in comps]))

    # ---- (1) single-component K-class confusion, two splits ----
    # Random pixel split is LEAKY (adjacent pixels are near-duplicates). A spatial
    # block split (train = left of each map, test = right) is the honest value.
    def _rf_cm(Xtr, ytr, Xte, yte):
        rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)
        rf.fit(Xtr, ytr); yp = rf.predict(Xte)
        return (confusion_matrix(yte, yp, labels=range(K)),
                float(accuracy_score(yte, yp)))

    Xall, yall = [], []
    Xtr_s, ytr_s, Xte_s, yte_s = [], [], [], []
    for i, c in enumerate(classes):
        Xp = preprocess(cubes[c])
        Xall.append(Xp); yall += [i] * len(Xp)
        xc = coords[c][:, 0]; left = xc < np.median(xc)      # split by X median
        if left.all() or (~left).all():
            left = np.arange(len(Xp)) % 2 == 0
        Xtr_s.append(Xp[left]); ytr_s += [i] * int(left.sum())
        Xte_s.append(Xp[~left]); yte_s += [i] * int((~left).sum())
    Xall = np.vstack(Xall); yall = np.array(yall)

    Xtr, Xte, ytr, yte = train_test_split(Xall, yall, test_size=0.3,
                                          stratify=yall, random_state=0)
    _, acc4_random = _rf_cm(Xtr, ytr, Xte, yte)
    cm4, acc4 = _rf_cm(np.vstack(Xtr_s), np.array(ytr_s),
                       np.vstack(Xte_s), np.array(yte_s))   # spatial = honest
    X, y = Xall, yall

    # PCA of the real per-pixel spectra (subsample per class for a clean plot)
    rng = np.random.default_rng(0)
    sel = np.concatenate([rng.choice(np.where(y == i)[0],
                          size=min(120, int((y == i).sum())), replace=False)
                          for i in range(K)])
    pca_emb = PCA(n_components=2, random_state=0).fit_transform(X[sel])
    pca_lab = y[sel]

    # ---- (2) mixture detection: three strategies ----
    clf = SERSMixtureClassifier(comps, prob_threshold=0.30,
                                max_components=len(comps),
                                augment=AugmentConfig(n_per_pure=200))
    clf.fit(pures)
    mix_names, mix_means, mix_cubes, true_lists = [], [], [], []
    for k, nominal in mixes.items():
        p = ratio_map_path(pest_dir, k)
        if p is None:
            continue
        _, cube, mean, _ = load_map(p)
        mix_names.append(k); mix_means.append(mean); mix_cubes.append(cube)
        true_lists.append([comps[i] for i, cc in enumerate(nominal) if cc > 0])
    if not mix_names:
        raise FileNotFoundError(
            "no mixture maps found (Ratio/ + mixtures.csv). The Real-data page "
            "needs mixtures; use the Model page for reference-only datasets.")
    mix_pp = preprocess(np.array(mix_means))

    # A) RandomForest on the mean   B) per-pixel NNLS vote   C) matched-filter
    pred_rf = [set(np.where(names_to_indicator([p], comps)[0])[0])
               for p in clf.predict(mix_pp)]
    pred_pp = [per_pixel_vote(c, pures, comps) for c in mix_cubes]
    pred_mf = [matched_significance(m, pures, comps) for m in mix_means]
    strategies = [
        ("RandomForest (mean)", *_score(true_lists, pred_rf, comps)),
        ("per-pixel NNLS vote", *_score(true_lists, pred_pp, comps)),
        ("matched-filter (mean)", *_score(true_lists, pred_mf, comps)),
    ]

    # per-pixel voting is the primary (best) detector for the grid + confusion
    pred_sets = pred_pp
    pred_lists = [[comps[i] for i in sorted(s)] for s in pred_sets]
    yt = names_to_indicator(true_lists, comps)
    yp = names_to_indicator(pred_lists, comps)
    micro = multilabel_metrics(yt, yp)

    # ---- (3) combination confusion (from the per-pixel detector) ----
    true_lbl = [_combo(t, comps) for t in true_lists]
    pred_lbl = [_combo(p, comps) for p in pred_lists]
    rows = sorted(set(true_lbl), key=lambda s: (s.count("+"), s))
    cols = rows + sorted(set(pred_lbl) - set(rows), key=lambda s: (s.count("+"), s))
    ri = {c: i for i, c in enumerate(rows)}; ci = {c: i for i, c in enumerate(cols)}
    M = np.zeros((len(rows), len(cols)), int)
    for t, p in zip(true_lbl, pred_lbl):
        M[ri[t], ci[p]] += 1
    exact = float(np.mean([t == p for t, p in zip(true_lbl, pred_lbl)]))

    # ---- (4) response-factor calibration ----
    B = {k: fit_B(m, pures)[0] for k, m in zip(mix_names, mix_pp)}
    R = _response_factors(mix_names, B, mixes, comps)

    n = len(comps)
    calib_rows, er, ec = [], [], []
    for k in mix_names:
        nominal = np.array(mixes[k], float)
        present = [i for i in range(n) if nominal[i] > 0]
        nom = _ratio_over(present, nominal, n)
        raw = _ratio_over(present, B[k], n)
        cal = _ratio_over(present, B[k] / R, n)
        calib_rows.append((k, present, nom, raw, cal))
        for i in present:
            er.append(abs(raw[i] - nom[i])); ec.append(abs(cal[i] - nom[i]))

    return RealResult(
        cm4=cm4, acc4=acc4, acc4_random=acc4_random,
        classes4=classes, comps=comps,
        mix_names=mix_names, yt=yt, yp=yp, micro=micro,
        combo_rows=rows, combo_cols=cols, combo_M=M, combo_exact=exact,
        strategies=strategies, R=R, calib_rows=calib_rows,
        err_raw=float(np.mean(er)) if er else 0.0,
        err_cal=float(np.mean(ec)) if ec else 0.0,
        pca_emb=pca_emb, pca_lab=pca_lab, pure_spectra=pures, wn=wn)


if __name__ == "__main__":
    r = compute_real()
    print("pure 4-class acc - spatial (honest):", round(r.acc4, 3),
          "| random (leaky):", round(r.acc4_random, 3))
    print("\ndetection strategies (recall / prec / F1 / exact):")
    for lbl, rec, prec, f1, ex in r.strategies:
        print(f"  {lbl:24s} {rec:.2f}  {prec:.2f}  {f1:.2f}  {ex:.2f}")
    print("\nR factors:", {c: round(v, 2) for c, v in zip(r.comps, r.R)})
    print("ratio err raw -> cal:", round(r.err_raw, 3), "->", round(r.err_cal, 3))
