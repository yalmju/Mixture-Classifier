"""model_training.py — train a single-component classifier on a set of reference
SERS maps, for the UNMIXR "Model" page.

The classes are whatever reference maps the data folder holds (one pure substance
per map) — discovered by `dataset.discover_references`, not hardcoded. The
DQ / THI / TBZ / BLK pesticides are just the example that ships.

Two selectable backends, both trained on the honest spatial (block) split — the
left half of each map trains, the right half tests (no adjacent-pixel leakage):

    "rf"      RandomForest on the per-pixel spectra  (scikit-learn, no torch).
              Learning curve = out-of-bag error as trees are added.
    "resnet"  ResNet1D deep classifier on the per-pixel spectra  (torch), with a
              per-epoch training-loss curve.

UI-agnostic: only numpy / scikit-learn are imported at module load; torch is
imported lazily inside the resnet path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from scipy.signal import savgol_filter

from real_data import load_map, PEST_DEFAULT
from dataset import discover_dataset, reference_dir, is_blank
from sers_mixture import als_baseline
from io_utils import load_calibration_csv


@dataclass
class TrainResult:
    backend: str                     # "rf" or "resnet"
    classes: list                    # discovered class names (blank last)
    comps: list                      # non-blank classes
    confusion: np.ndarray            # (K, K) int, spatial split, rows=true cols=pred
    acc: float                       # spatial-split test accuracy
    macro_f1: float                  # unweighted mean of per-class F1
    per_component: dict              # name -> (precision, recall, f1, support)
    curve_x: np.ndarray              # epoch (resnet) or #trees (rf)
    curve_y: np.ndarray              # training loss (resnet) or OOB error (rf)
    curve_label: str                 # y-axis label for the learning curve
    curve_xlabel: str                # x-axis label for the learning curve
    pca_emb: np.ndarray              # (n_sample, 2) PCA of real per-pixel spectra
    pca_lab: np.ndarray              # (n_sample,) class index
    n_train: int
    n_test: int
    wn: np.ndarray = None            # wavenumber axis
    split: str = "spatial"           # "spatial" (honest) or "random" (leaky)
    band_f: np.ndarray = None        # per-wavenumber ANOVA F (class discriminability)
    acc_std: float = 0.0             # cross-fold accuracy SD (batch-CV only)
    vip: np.ndarray = None           # per-wavenumber PLS-DA VIP (only for the PLS backend)
    model: object = None             # the fitted estimator (for saving / reuse)
    pca_var: np.ndarray = None       # (2,) PC1/PC2 explained-variance ratio
    box_wn: np.ndarray = None        # (k,) wavenumbers of the top discriminative bands
    box_vals: np.ndarray = None      # (n_sub, k) intensity at those bands (subsampled)
    box_lab: np.ndarray = None       # (n_sub,) class index for each box row


# --------------------------------------------------------------------------
# data — discovered reference maps, honest spatial (block) split
# --------------------------------------------------------------------------
def _featurize(cube, baseline=True, deriv=0, norm="l2"):
    """Per-pixel feature transform: (optional ALS baseline) → (optional Savitzky-
    Golay derivative) → normalization (l2 / snv / none)."""
    X = np.asarray(cube, float)
    if baseline:
        X = np.stack([y - als_baseline(y) for y in X])
        X = np.clip(X, 0, None)                        # SERS intensities >= 0
    if deriv in (1, 2):
        w = min(11, X.shape[1] // 2 * 2 - 1)           # odd window <= n_features
        X = savgol_filter(X, window_length=max(5, w), polyorder=3,
                          deriv=deriv, axis=1)
    if norm == "l2":
        n = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.where(n > 0, n, 1.0)
    elif norm == "snv":                                # standard normal variate
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        X = (X - mu) / np.where(sd > 0, sd, 1.0)
    return X


def _feat_map(path, baseline=True, trim=None, deriv=0, norm="l2"):
    """Load one map, optionally trim the wavenumber window, and featurize it.
    Returns (wn, X (n_pix, n_feat), coords)."""
    wn, cube, _mean, coord = load_map(path)
    if trim is not None:
        lo, hi = trim
        m = (wn >= lo) & (wn <= hi)
        if m.sum() >= 10:
            wn = wn[m]; cube = cube[:, m]
    return wn, _featurize(cube, baseline=baseline, deriv=deriv, norm=norm), coord


def _renorm(X, norm):
    if norm == "l2":
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(n > 0, n, 1.0)
    if norm == "snv":
        mu = X.mean(axis=1, keepdims=True); sd = X.std(axis=1, keepdims=True)
        return (X - mu) / np.where(sd > 0, sd, 1.0)
    return X


def _augment_lowconc(X, y, norm, seed, sigmas=(0.03, 0.06, 0.10)):
    """Add noisier copies of each training spectrum to mimic the lower SNR of low
    concentrations. The features are magnitude-normalised (peaks ~0.3–0.7 for L2),
    so noise on the order of the peak values is what degrades SNR — hence absolute
    sigmas (scaled by the feature spread for non-L2 norms), not a tiny fraction of
    the median. Makes the classifier transfer from high-conc references to trace."""
    rng = np.random.default_rng(seed)
    base = 1.0 if norm == "l2" else (float(X.std()) or 1.0)
    Xs, ys = [X], [y]
    for s in sigmas:
        Xn = _renorm(X + rng.normal(0.0, s * base, X.shape), norm)
        Xs.append(Xn); ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def _detectable(spec):
    """True if a spectrum has a peak clearly above its own noise (so sub-LOD
    dilution spectra, which are indistinguishable from a blank, are not mislabelled
    as the compound)."""
    br = np.asarray(spec, float) - als_baseline(np.asarray(spec, float))
    return br.max() > 5.0 * (np.std(br) + 1e-9)


def _calib_train_samples(calib_path, classes, wn, baseline, trim, deriv, norm):
    """Featurise a dilution-series CSV into extra training samples (one per standard)
    for the classes that exist in the dataset — real low→high concentration examples
    so the model sees the full range, not just the near-saturation references. Only
    spectra with a detectable peak are kept (sub-LOD standards would just teach the
    model that noise is the compound)."""
    axis_c, names_c, dils = load_calibration_csv(calib_path)
    Xc, yc = [], []
    for name, (Cg, specs) in zip(names_c, dils):
        if name not in classes:
            continue
        ci = classes.index(name)
        specs = np.asarray(specs, float)
        if trim is not None:
            lo, hi = trim; m = (axis_c >= lo) & (axis_c <= hi)
            if m.sum() >= 10:
                specs = specs[:, m]
        if wn is not None and specs.shape[1] != len(wn):
            continue                                        # axis mismatch → skip safely
        keep = np.array([_detectable(s) for s in specs])
        if not keep.any():
            continue
        Xf = _featurize(specs[keep], baseline=baseline, deriv=deriv, norm=norm)
        Xc.append(Xf); yc += [ci] * len(Xf)
    if not Xc:
        return None, None
    return np.vstack(Xc), np.array(yc)


def _load_split(data_dir, baseline=True, trim=None, deriv=0, norm="l2",
                split="spatial", seed=0, test_frac=0.5, augment=False,
                calib_path=None, progress=None):
    """Group the reference maps into classes (merging batches of the same
    substance), featurize each map, and split into train/test.

    ``split``:
      "spatial"  honest block split within each map — the top ``test_frac`` by X
                 tests, the rest trains
      "random"   leaky per-pixel shuffle, holding out ``test_frac`` (for comparison)
      "batch"    leave-one-batch-out — hold each class's last batch's map out as
                 test (single-map classes fall back to a spatial split)
      "manual"   honour per-map train/test roles from Samples (samples.csv)
    ``test_frac`` (0-1) sets the held-out fraction for the spatial / random splits
    (and the spatial fallback). ``trim`` is an optional (low, high) wavenumber
    window; ``baseline`` / ``deriv`` / ``norm`` control the feature transform.
    Returns per-pixel matrices + the ordered class names."""
    groups = discover_dataset(data_dir)
    if len(groups) < 2:
        raise FileNotFoundError(
            "need at least 2 substance classes (one or more maps each). Pick the "
            "data folder with your reference maps (a Reference/ subfolder is used "
            f"if present).\nlooked in: {reference_dir(data_dir)}"
            f"\nfound classes: {[c for c, _ in groups]}")

    classes = [c for c, _ in groups]
    frac = min(0.9, max(0.05, float(test_frac)))       # keep both sides non-empty

    def _spatial(X, coord):
        """train = the lower (1-frac) by X, test = the top `frac` (contiguous)."""
        xc = coord[:, 0]
        thr = np.quantile(xc, 1.0 - frac)
        train = xc < thr
        if train.all() or (~train).all():              # degenerate coords -> by count
            k = max(1, int(round(len(X) * (1.0 - frac))))
            train = np.arange(len(X)) < k
        return train

    rng = np.random.default_rng(seed)
    n_maps = sum(len(m) for _c, m in groups); done = 0
    Xtr, ytr, Xte, yte, wn = [], [], [], [], None
    for i, (_cls, maps) in enumerate(groups):
        feats = []
        for batch, path, role in maps:
            done += 1
            if progress:
                progress(f"loading & preprocessing '{_cls}'  "
                         f"(map {done}/{n_maps})")
            wn, X, coord = _feat_map(path, baseline, trim, deriv, norm)
            feats.append((batch, X, coord, role))

        if split == "random":                          # pool all pixels, shuffle
            allX = np.vstack([X for _b, X, _c, _r in feats])
            n = len(allX); n_te = min(n - 1, max(1, int(round(n * frac))))
            idx = rng.permutation(n)
            Xte.append(allX[idx[:n_te]]); yte += [i] * n_te
            Xtr.append(allX[idx[n_te:]]); ytr += [i] * (n - n_te)
        elif split == "manual" and any(r == "test" for _b, _X, _c, r in feats):
            for _batch, X, _c, role in feats:          # honour the Samples roles
                (Xte if role == "test" else Xtr).append(X)
                (yte if role == "test" else ytr).extend([i] * len(X))
        elif split == "batch" and len(feats) >= 2:     # leave the last batch out
            held = max(b for b, _X, _c, _r in feats)
            for batch, X, _c, _r in feats:
                (Xte if batch == held else Xtr).append(X)
                (yte if batch == held else ytr).extend([i] * len(X))
        else:                                          # spatial (also 1-map fallback)
            for _b, X, coord, _r in feats:
                left = _spatial(X, coord)
                Xtr.append(X[left]); ytr += [i] * int(left.sum())
                Xte.append(X[~left]); yte += [i] * int((~left).sum())
    Xtr = np.vstack(Xtr); ytr = np.array(ytr)
    Xte = np.vstack(Xte); yte = np.array(yte)

    if calib_path:                                          # add dilution-series examples
        if progress:
            progress("adding dilution-series spectra to training")
        Xc, yc = _calib_train_samples(calib_path, classes, wn, baseline, trim, deriv, norm)
        if Xc is not None:
            Xtr = np.vstack([Xtr, Xc]); ytr = np.concatenate([ytr, yc])
    if augment:                                             # low-conc noise augmentation
        if progress:
            progress("augmenting training set for low-concentration transfer")
        Xtr, ytr = _augment_lowconc(Xtr, ytr, norm, seed)
    return Xtr, ytr, Xte, yte, wn, classes


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------
def _train_rf(Xtr, ytr, Xte, yte, K, n_estimators=300, seed=0, progress=None):
    """RandomForest on per-pixel spectra; OOB-error learning curve as trees grow."""
    from sklearn.ensemble import RandomForestClassifier

    step = max(50, int(round(n_estimators / 6 / 10)) * 10)
    trees = list(range(step, n_estimators + 1, step))
    if not trees or trees[-1] != n_estimators:
        trees.append(n_estimators)

    rf = RandomForestClassifier(oob_score=True, warm_start=True, bootstrap=True,
                                n_jobs=-1, random_state=seed)
    xs, ys = [], []
    for n in trees:
        rf.n_estimators = n
        rf.fit(Xtr, ytr)
        xs.append(n)
        ys.append(1.0 - float(rf.oob_score_))         # OOB error
        if progress:
            progress(f"RandomForest — {n}/{n_estimators} trees  "
                     f"(OOB err {ys[-1]:.3f})")
    yp = rf.predict(Xte)
    cm = confusion_matrix(yte, yp, labels=range(K))
    acc = float(np.mean(yp == yte))
    return cm, acc, np.array(xs, float), np.array(ys, float), "OOB error", "trees", rf


def _train_resnet(Xtr, ytr, Xte, yte, K, epochs=25, batch_size=128, lr=1e-3,
                  base=16, seed=0, progress=None):
    """ResNet1D K-class classifier on per-pixel spectra; per-epoch loss curve."""
    import torch
    import torch.nn as nn
    from resnet1d import ResNet1D

    torch.manual_seed(seed)
    Xt = torch.tensor(np.asarray(Xtr, np.float32))
    yt = torch.tensor(np.asarray(ytr, np.int64))
    ds = torch.utils.data.TensorDataset(Xt, yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    net = ResNet1D(K, base=base)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    net.train()
    xs, ys = [], []
    for ep in range(epochs):
        tot = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(xb)
        xs.append(ep + 1)
        ys.append(tot / len(ds))
        if progress:
            progress(f"ResNet1D — epoch {ep + 1}/{epochs}  (loss {ys[-1]:.3f})")

    net.eval()
    with torch.no_grad():
        logits = net(torch.tensor(np.asarray(Xte, np.float32)))
        yp = logits.argmax(1).cpu().numpy()
    cm = confusion_matrix(yte, yp, labels=range(K))
    acc = float(np.mean(yp == yte))
    return (cm, acc, np.array(xs, float), np.array(ys, float),
            "training loss (cross-entropy)", "epoch", net)


class _PLSDA:
    """PLS-DA classifier — PLS regression onto one-hot class targets, predict =
    arg-max column. Keeps the fitted PLS so VIP scores can be read off it. This is
    the standard chemometrics discriminant model; VIP > 1 marks important bands."""

    def __init__(self, n_components=10, seed=0):
        self.n_components = n_components

    def fit(self, X, y):
        from sklearn.cross_decomposition import PLSRegression
        X = np.asarray(X, float)
        self.classes_ = np.unique(y)
        Y = np.zeros((len(y), len(self.classes_)))
        for k, c in enumerate(self.classes_):
            Y[np.asarray(y) == c, k] = 1.0
        nc = min(self.n_components, X.shape[1], max(1, X.shape[0] - 1))
        self.pls = PLSRegression(n_components=max(1, nc)).fit(X, Y)
        return self

    def predict(self, X):
        pred = self.pls.predict(np.asarray(X, float))
        return self.classes_[pred.argmax(1)]


def _vip(pls):
    """Variable Importance in Projection for a fitted sklearn PLSRegression.
    VIP_j = sqrt( p · Σ_a s_a (w_ja/‖w_a‖)² / Σ_a s_a ),  s_a = Y-variance of comp a."""
    t = np.asarray(pls.x_scores_)                          # (n, A)
    w = np.asarray(pls.x_weights_)                         # (p, A)
    q = np.asarray(pls.y_loadings_)                        # (m, A)
    p = w.shape[0]
    ssy = (q ** 2).sum(axis=0) * (t ** 2).sum(axis=0)      # (A,) variance explained in Y
    total = float(ssy.sum())
    if total <= 0:
        return np.zeros(p)
    wnorm = w / np.where(np.linalg.norm(w, axis=0, keepdims=True) > 0,
                         np.linalg.norm(w, axis=0, keepdims=True), 1.0)
    vip = np.sqrt(p * (wnorm ** 2 @ ssy) / total)
    return np.nan_to_num(vip, nan=0.0, posinf=0.0, neginf=0.0)


def _model_factory(algo, seed):
    """Return a no-arg callable that builds a fresh scikit-learn classifier."""
    if algo == "svm":
        from sklearn.svm import SVC
        return lambda: SVC(kernel="rbf", C=10.0, gamma="scale", random_state=seed)
    if algo == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return lambda: KNeighborsClassifier(n_neighbors=5)
    if algo == "logreg":
        from sklearn.linear_model import LogisticRegression
        return lambda: LogisticRegression(max_iter=1000)
    if algo == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return lambda: HistGradientBoostingClassifier(random_state=seed)
    if algo == "pls":
        return lambda: _PLSDA(seed=seed)
    raise ValueError(f"unknown algorithm: {algo}")


def _train_generic(make_model, Xtr, ytr, Xte, yte, K, seed=0, progress=None):
    """Any scikit-learn classifier: learning curve = test error vs training-set
    size (train on growing subsets, evaluate on the held-out spatial block),
    final model fit on the full training block for the confusion matrix."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(Xtr))
    xs, ys = [], []
    for fr in (0.2, 0.4, 0.6, 0.8, 1.0):
        m = max(K * 2, int(len(Xtr) * fr))
        idx = order[:m]
        if progress:
            progress(f"fitting on {int(fr * 100)}% of the training set")
        mdl = make_model(); mdl.fit(Xtr[idx], ytr[idx])
        yp = mdl.predict(Xte)
        xs.append(m); ys.append(1.0 - float(np.mean(yp == yte)))
    mdl = make_model(); mdl.fit(Xtr, ytr)
    yp = mdl.predict(Xte)
    cm = confusion_matrix(yte, yp, labels=range(K))
    acc = float(np.mean(yp == yte))
    return (cm, acc, np.array(xs, float), np.array(ys, float),
            "test error", "training-set size", mdl)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def _resnet_predict(Xtr, ytr, Xte, K, epochs=25, seed=0):
    """Train ResNet1D and return test predictions (no learning curve — for CV)."""
    import torch
    import torch.nn as nn
    from resnet1d import ResNet1D
    torch.manual_seed(seed)
    Xt = torch.tensor(np.asarray(Xtr, np.float32))
    yt = torch.tensor(np.asarray(ytr, np.int64))
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt), batch_size=128, shuffle=True)
    net = ResNet1D(K); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss(); net.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(np.asarray(Xte, np.float32))).argmax(1).cpu().numpy()


def _fit_predict(backend, Xtr, ytr, Xte, K, epochs=25, n_estimators=300, seed=0):
    """Fit the chosen backend and return test-set predictions (one CV fold)."""
    if backend == "resnet":
        return _resnet_predict(Xtr, ytr, Xte, K, epochs=epochs, seed=seed)
    if backend == "rf":
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1,
                                    random_state=seed)
        rf.fit(Xtr, ytr); return rf.predict(Xte)
    mdl = _model_factory(backend, seed)(); mdl.fit(Xtr, ytr)
    return mdl.predict(Xte)


def _train_batch_cv(data_dir, backend, epochs, n_estimators, seed,
                    baseline, trim, deriv, norm, progress) -> TrainResult:
    """Leave-one-batch-out cross-validation. Every class must share the same set
    of >=2 batches; each batch is held out as the test fold in turn. The confusion
    matrix is pooled (each map tested once) and accuracy is reported as mean ± SD
    across folds — the honest cross-spot number replicate maps enable."""
    groups = discover_dataset(data_dir)
    classes = [c for c, _ in groups]
    K = len(classes)
    if K < 2:
        raise FileNotFoundError("need at least 2 substance classes.")
    batch_sets = [set(b for b, _p, _r in maps) for _c, maps in groups]
    folds = sorted(batch_sets[0])
    if len(folds) < 2 or any(bs != set(folds) for bs in batch_sets):
        raise ValueError(
            "batch-CV needs the SAME set of >=2 batches for every class (got "
            f"{ {c: sorted(bs) for c, bs in zip(classes, batch_sets)} }). Assign "
            "batches in Samples, or use the 'batch' / 'manual' split.")

    per_maps, wn = [], None                             # (class idx, batch, X)
    for i, (cls, maps) in enumerate(groups):
        for batch, path, _role in maps:
            if progress:
                progress(f"loading '{cls}' batch {batch}")
            wn, X, _coord = _feat_map(path, baseline, trim, deriv, norm)
            per_maps.append((i, batch, X))

    cm = np.zeros((K, K), int); fold_acc = []
    for fi, f in enumerate(folds):
        if progress:
            progress(f"batch-CV — fold {fi + 1}/{len(folds)} (test = batch {f})")
        Xtr, ytr, Xte, yte = [], [], [], []
        for ci, b, X in per_maps:
            if b == f:
                Xte.append(X); yte.extend([ci] * len(X))
            else:
                Xtr.append(X); ytr.extend([ci] * len(X))
        yp = _fit_predict(backend, np.vstack(Xtr), np.array(ytr),
                          np.vstack(Xte), K, epochs, n_estimators, seed)
        yte = np.array(yte)
        cm += confusion_matrix(yte, yp, labels=range(K))
        fold_acc.append(float(np.mean(yp == yte)))

    acc = float(np.mean(fold_acc)); acc_std = float(np.std(fold_acc))
    per_c = _per_class_prf(cm, classes)
    macro_f1 = float(np.mean([per_c[c][2] for c in classes]))

    Xall = np.vstack([X for _i, _b, X in per_maps])
    yall = np.concatenate([[i] * len(X) for i, _b, X in per_maps])
    from sklearn.feature_selection import f_classif
    F, _p = f_classif(Xall, yall)
    band_f = np.nan_to_num(np.asarray(F, float), nan=0.0, posinf=0.0, neginf=0.0)
    vip = _vip(_PLSDA(seed=seed).fit(Xall, yall).pls) if backend == "pls" else None
    rng = np.random.default_rng(seed)
    sel = np.concatenate([
        rng.choice(np.where(yall == i)[0],
                   size=min(120, int((yall == i).sum())), replace=False)
        for i in range(K)])
    _pca = PCA(n_components=2, random_state=seed).fit(Xall[sel])
    pca_emb = _pca.transform(Xall[sel]); pca_lab = yall[sel]
    box_wn, box_vals, box_lab = _top_band_box(
        Xall, yall, vip if vip is not None else band_f, wn, K, seed)
    comps = [c for c in classes if not is_blank(c)]
    return TrainResult(
        backend=backend, classes=classes, comps=comps, confusion=cm, acc=acc,
        macro_f1=macro_f1, per_component=per_c,
        curve_x=np.array(folds, float), curve_y=np.array(fold_acc),
        curve_label="fold test accuracy", curve_xlabel="held-out batch",
        pca_emb=pca_emb, pca_lab=pca_lab, n_train=len(yall), n_test=int(cm.sum()),
        wn=wn, split="batch-cv", band_f=band_f, acc_std=acc_std, vip=vip,
        pca_var=_pca.explained_variance_ratio_,
        box_wn=box_wn, box_vals=box_vals, box_lab=box_lab)


def top_bands(importance, wn, k=6, min_sep=30.0):
    """Indices of the k most discriminative bands, but spaced ≥ ``min_sep`` cm⁻¹
    apart so the list is distinct PEAKS, not several adjacent points of the same
    peak (greedy non-maximum suppression by importance)."""
    picked = []
    for idx in np.argsort(importance)[::-1]:
        if all(abs(wn[idx] - wn[j]) >= min_sep for j in picked):
            picked.append(int(idx))
        if len(picked) >= k:
            break
    return np.array(picked, int)


def _top_band_box(Xall, yall, importance, wn, K, seed, k=6, per_class=150):
    """For the k most discriminative (and well-separated) bands, collect a per-class
    subsample of the intensity at each band, so the UI can show a box plot of how
    each substance's signal distributes at the best peaks.
    Returns (box_wn (k,), box_vals (n_sub, k), box_lab (n_sub,))."""
    if importance is None or wn is None or len(importance) != Xall.shape[1]:
        return None, None, None
    top = top_bands(importance, wn, k=k)              # distinct peaks, strongest first
    rng = np.random.default_rng(seed)
    sel = np.concatenate([
        rng.choice(np.where(yall == i)[0],
                   size=min(per_class, int((yall == i).sum())), replace=False)
        for i in range(K) if (yall == i).any()])
    return wn[top], Xall[sel][:, top], yall[sel]


def _per_class_prf(cm, classes):
    """Per-class (precision, recall, f1, support) from a confusion matrix."""
    per = {}
    for i, nm in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        per[nm] = (float(p), float(r), float(f), int(cm[i, :].sum()))
    return per


def train_model(pest_dir=PEST_DEFAULT, backend="rf", epochs=25,
                n_estimators=300, seed=0, baseline=True, trim=None,
                deriv=0, norm="l2", split="spatial", test_frac=0.5,
                augment=False, calib_path=None, progress=None) -> TrainResult:
    """Discover the reference maps in ``pest_dir`` and train the chosen algorithm.
    ``backend`` is one of rf / resnet / svm / knn / logreg / gbm. Feature options:
    ``baseline`` (ALS on/off), ``deriv`` (0/1/2 Savitzky-Golay), ``norm``
    (l2/snv/none), ``trim`` (low, high) wavenumber window. ``split`` is "spatial"
    (honest) or "random" (leaky). ``pest_dir`` is the data folder (or its
    Reference/ subfolder)."""
    if backend == "resnet":
        try:
            import torch  # noqa: F401  (fail early with a clear message)
        except Exception as exc:
            raise RuntimeError(
                "ResNet1D backend needs PyTorch — install it (pip install torch) "
                "or pick a scikit-learn algorithm (RandomForest, SVM, k-NN…).") from exc

    if split == "batch-cv":
        return _train_batch_cv(pest_dir, backend, epochs, n_estimators, seed,
                               baseline, trim, deriv, norm, progress)

    Xtr, ytr, Xte, yte, wn, classes = _load_split(
        pest_dir, baseline=baseline, trim=trim, deriv=deriv, norm=norm,
        split=split, seed=seed, test_frac=test_frac, augment=augment,
        calib_path=calib_path, progress=progress)
    K = len(classes)

    if backend == "resnet":
        cm, acc, cx, cy, ylab, xlab, model = _train_resnet(
            Xtr, ytr, Xte, yte, K, epochs=epochs, seed=seed, progress=progress)
    elif backend == "rf":
        cm, acc, cx, cy, ylab, xlab, model = _train_rf(
            Xtr, ytr, Xte, yte, K, n_estimators=n_estimators, seed=seed,
            progress=progress)
    else:
        cm, acc, cx, cy, ylab, xlab, model = _train_generic(
            _model_factory(backend, seed), Xtr, ytr, Xte, yte, K, seed=seed,
            progress=progress)

    if progress:
        progress("finalising — confusion, F1, PCA…")
    per = _per_class_prf(cm, classes)
    macro_f1 = float(np.mean([per[nm][2] for nm in classes]))

    # per-wavenumber ANOVA F across classes — which bands discriminate substances
    from sklearn.feature_selection import f_classif
    Xall = np.vstack([Xtr, Xte]); yall = np.concatenate([ytr, yte])
    F, _pval = f_classif(Xall, yall)
    band_f = np.nan_to_num(np.asarray(F, float), nan=0.0, posinf=0.0, neginf=0.0)

    # PLS-DA also yields VIP scores (the chemometrics band-importance measure)
    vip = _vip(model.pls) if (backend == "pls" and hasattr(model, "pls")) else None

    # PCA of the real per-pixel spectra (subsample per class for a clean plot)
    rng = np.random.default_rng(seed)
    sel = np.concatenate([
        rng.choice(np.where(yall == i)[0],
                   size=min(120, int((yall == i).sum())), replace=False)
        for i in range(K)])
    _pca = PCA(n_components=2, random_state=seed).fit(Xall[sel])
    pca_emb = _pca.transform(Xall[sel]); pca_lab = yall[sel]
    box_wn, box_vals, box_lab = _top_band_box(
        Xall, yall, vip if vip is not None else band_f, wn, K, seed)

    comps = [c for c in classes if not is_blank(c)]
    return TrainResult(
        backend=backend, classes=classes, comps=comps,
        confusion=cm, acc=acc, macro_f1=macro_f1, per_component=per,
        curve_x=cx, curve_y=cy, curve_label=ylab, curve_xlabel=xlab,
        pca_emb=pca_emb, pca_lab=pca_lab, pca_var=_pca.explained_variance_ratio_,
        box_wn=box_wn, box_vals=box_vals, box_lab=box_lab,
        n_train=len(ytr), n_test=len(yte), wn=wn, split=split, band_f=band_f,
        vip=vip, model=model)


if __name__ == "__main__":
    r = train_model(backend="rf")
    print(f"backend={r.backend}  classes={r.classes}")
    print(f"spatial acc={r.acc:.3f}  macro F1={r.macro_f1:.3f}")
    for nm in r.classes:
        p, rc, f, s = r.per_component[nm]
        print(f"  {nm:8s}  P={p:.2f} R={rc:.2f} F1={f:.2f}  (n={s})")
