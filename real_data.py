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
    """Return (wavenumbers, cube (n_pixels, n_wn), mean_spectrum)."""
    rows = list(csv.reader(open(path)))
    wn = np.array([float(v) for v in rows[2][2:] if v.strip() != ""])
    n = len(wn)
    data = []
    for r in rows[3:]:
        vals = [v for v in r[2:] if v.strip() != ""]
        if len(vals) < n:
            continue
        data.append([float(v) for v in vals[:n]])
    cube = np.asarray(data, float)
    return wn, cube, cube.mean(axis=0)


def _combo(names):
    order = [c for c in COMPS if c in names]
    return "+".join(order) if order else "(none)"


def per_pixel_vote(cube, pures, frac_thr=0.15, pix_thr=0.15):
    """Detect a component if it is a meaningful fraction in >= pix_thr of pixels.

    The mean spectrum buries a minor compound; individual pixels where local
    coverage favours it still show it. Vote across the 400 pixels."""
    Xp = preprocess(cube)
    votes = np.zeros(len(COMPS))
    for yv in Xp:
        B, _ = nnls(pures.T, yv)
        votes += (B / (B.sum() + 1e-12) >= frac_thr)
    return set(np.where(votes / len(Xp) >= pix_thr)[0])


def matched_significance(mean, pures, k=9.0):
    """Keep component a if excluding it raises the NNLS residual by > noise (LOD)."""
    y = preprocess(mean[None, :])[0]
    B_full, _ = nnls(pures.T, y)
    resid = y - B_full @ pures
    noise = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
    keep = set()
    for a in range(len(COMPS)):
        others = [i for i in range(len(COMPS)) if i != a]
        Bo, _ = nnls(pures[others].T, y)
        r_wo = np.linalg.norm(y - Bo @ pures[others])
        r_full = np.linalg.norm(resid)
        if (r_wo - r_full) > k * noise:
            keep.add(a)
    return keep


def _score(true_lists, pred_sets):
    """(recall, precision, f1, exact_composition) for index-set predictions."""
    yt = np.array([[1 if COMPS[i] in t else 0 for i in range(3)]
                   for t in true_lists])
    yp = np.array([[1 if i in p else 0 for i in range(3)] for p in pred_sets])
    m = multilabel_metrics(yt, yp)
    exact = float(np.mean([set(np.where(a)[0]) == b for a, b in zip(yt, pred_sets)]))
    return m["micro_recall"], m["micro_precision"], m["micro_f1"], exact


@dataclass
class RealResult:
    cm4: np.ndarray
    acc4: float
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


def _ratio_over(present_idx, vec):
    v = np.array([vec[i] if i in present_idx else 0.0 for i in range(3)])
    return v / (v.sum() + 1e-12)


def compute_real(pest_dir=PEST_DEFAULT):
    ref_dir = os.path.join(pest_dir, "Reference")
    ratio_dir = os.path.join(pest_dir, "Ratio")
    missing = [f for f in REF_FILES.values()
               if not os.path.exists(os.path.join(ref_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"reference CSVs not found in {ref_dir}\nmissing: {missing}")

    # ---- load references (per-pixel cubes + means) ----
    cubes, means, wn = {}, {}, None
    for lab, fn in REF_FILES.items():
        wn, cube, mean = load_map(os.path.join(ref_dir, fn))
        cubes[lab] = cube; means[lab] = mean
    pures = preprocess(np.vstack([means["DQ"], means["THI"], means["TBZ"]]))

    # ---- (1) single-component 4-class per-pixel confusion ----
    X, y = [], []
    for i, lab in enumerate(CLASSES4):
        Xp = preprocess(cubes[lab])
        X.append(Xp); y += [i] * len(Xp)
    X = np.vstack(X); y = np.array(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y,
                                          random_state=0)
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)
    rf.fit(Xtr, ytr); yp4 = rf.predict(Xte)
    cm4 = confusion_matrix(yte, yp4, labels=range(4))
    acc4 = float(accuracy_score(yte, yp4))

    # PCA of the real per-pixel spectra (subsample per class for a clean plot)
    rng = np.random.default_rng(0)
    sel = np.concatenate([rng.choice(np.where(y == i)[0],
                          size=min(120, int((y == i).sum())), replace=False)
                          for i in range(4)])
    pca_emb = PCA(n_components=2, random_state=0).fit_transform(X[sel])
    pca_lab = y[sel]

    # ---- (2) mixture detection: three strategies ----
    clf = SERSMixtureClassifier(COMPS, prob_threshold=0.30, max_components=3,
                                augment=AugmentConfig(n_per_pure=200))
    clf.fit(pures)
    mix_names, mix_means, mix_cubes, true_lists = [], [], [], []
    for k, nominal in MIXES.items():
        p = os.path.join(ratio_dir, k + "_corrected.csv")
        if not os.path.exists(p):
            continue
        _, cube, mean = load_map(p)
        mix_names.append(k); mix_means.append(mean); mix_cubes.append(cube)
        true_lists.append([COMPS[i] for i, c in enumerate(nominal) if c > 0])
    mix_pp = preprocess(np.array(mix_means))

    # A) RandomForest on the mean   B) per-pixel NNLS vote   C) matched-filter
    pred_rf = [set(np.where(names_to_indicator([p], COMPS)[0])[0])
               for p in clf.predict(mix_pp)]
    pred_pp = [per_pixel_vote(c, pures) for c in mix_cubes]
    pred_mf = [matched_significance(m, pures) for m in mix_means]
    strategies = [
        ("RandomForest (mean)", *_score(true_lists, pred_rf)),
        ("per-pixel NNLS vote", *_score(true_lists, pred_pp)),
        ("matched-filter (mean)", *_score(true_lists, pred_mf)),
    ]

    # per-pixel voting is the primary (best) detector for the grid + confusion
    pred_sets = pred_pp
    pred_lists = [[COMPS[i] for i in sorted(s)] for s in pred_sets]
    yt = names_to_indicator(true_lists, COMPS)
    yp = names_to_indicator(pred_lists, COMPS)
    micro = multilabel_metrics(yt, yp)

    # ---- (3) combination confusion (from the per-pixel detector) ----
    true_lbl = [_combo(t) for t in true_lists]
    pred_lbl = [_combo(p) for p in pred_lists]
    rows = sorted(set(true_lbl), key=lambda s: (s.count("+"), s))
    cols = rows + sorted(set(pred_lbl) - set(rows), key=lambda s: (s.count("+"), s))
    ri = {c: i for i, c in enumerate(rows)}; ci = {c: i for i, c in enumerate(cols)}
    M = np.zeros((len(rows), len(cols)), int)
    for t, p in zip(true_lbl, pred_lbl):
        M[ri[t], ci[p]] += 1
    exact = float(np.mean([t == p for t, p in zip(true_lbl, pred_lbl)]))

    # ---- (4) response-factor calibration ----
    B = {k: fit_B(m, pures)[0] for k, m in zip(mix_names, mix_pp)}
    A, yv = [], []
    A.append([1, 0]); yv.append(np.log(B["DQ1TH1"][0] / B["DQ1TH1"][1]))
    A.append([0, 1]); yv.append(np.log(B["TB1TH1"][2] / B["TB1TH1"][1]))
    A.append([-1, 1]); yv.append(np.log(B["TBZ1DQ1"][2] / B["TBZ1DQ1"][0]))
    x, *_ = np.linalg.lstsq(np.array(A, float), np.array(yv, float), rcond=None)
    R = np.array([np.exp(x[0]), 1.0, np.exp(x[1])]); R = R / np.median(R)

    calib_rows, er, ec = [], [], []
    for k in mix_names:
        nominal = np.array(MIXES[k], float)
        present = [i for i in range(3) if nominal[i] > 0]
        nom = _ratio_over(present, nominal)
        raw = _ratio_over(present, B[k])
        cal = _ratio_over(present, B[k] / R)
        calib_rows.append((k, present, nom, raw, cal))
        for i in present:
            er.append(abs(raw[i] - nom[i])); ec.append(abs(cal[i] - nom[i]))

    return RealResult(
        cm4=cm4, acc4=acc4, classes4=CLASSES4, comps=COMPS,
        mix_names=mix_names, yt=yt, yp=yp, micro=micro,
        combo_rows=rows, combo_cols=cols, combo_M=M, combo_exact=exact,
        strategies=strategies, R=R, calib_rows=calib_rows,
        err_raw=float(np.mean(er)), err_cal=float(np.mean(ec)),
        pca_emb=pca_emb, pca_lab=pca_lab, pure_spectra=pures, wn=wn)


if __name__ == "__main__":
    r = compute_real()
    print("pure 4-class acc:", round(r.acc4, 3))
    print("\ndetection strategies (recall / prec / F1 / exact):")
    for lbl, rec, prec, f1, ex in r.strategies:
        print(f"  {lbl:24s} {rec:.2f}  {prec:.2f}  {f1:.2f}  {ex:.2f}")
    print("\nR factors:", {c: round(v, 2) for c, v in zip(r.comps, r.R)})
    print("ratio err raw -> cal:", round(r.err_raw, 3), "->", round(r.err_cal, 3))
