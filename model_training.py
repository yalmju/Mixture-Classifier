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


def _load_split(data_dir, baseline=True, trim=None, deriv=0, norm="l2",
                split="spatial", seed=0, progress=None):
    """Group the reference maps into classes (merging batches of the same
    substance), featurize each map, and split into train/test.

    ``split``:
      "spatial"  honest block split — left half of each map trains, right tests
      "random"   leaky per-pixel shuffle (for comparison)
      "batch"    leave-one-batch-out — hold each class's last batch's map out as
                 test (classes with a single map fall back to a spatial split)
    ``trim`` is an optional (low, high) wavenumber window; ``baseline`` / ``deriv``
    / ``norm`` control the feature transform. Returns per-pixel matrices + the
    ordered class names."""
    groups = discover_dataset(data_dir)
    if len(groups) < 2:
        raise FileNotFoundError(
            "need at least 2 substance classes (one or more maps each). Pick the "
            "data folder with your reference maps (a Reference/ subfolder is used "
            f"if present).\nlooked in: {reference_dir(data_dir)}"
            f"\nfound classes: {[c for c, _ in groups]}")

    classes = [c for c, _ in groups]

    def _feat_map(path):
        wn_, cube, _mean, coord = load_map(path)
        if trim is not None:
            lo, hi = trim
            m = (wn_ >= lo) & (wn_ <= hi)
            if m.sum() >= 10:                          # ignore degenerate windows
                wn_ = wn_[m]; cube = cube[:, m]
        return wn_, _featurize(cube, baseline=baseline, deriv=deriv, norm=norm), coord

    def _spatial(X, coord):
        left = coord[:, 0] < np.median(coord[:, 0])
        if left.all() or (~left).all():                # degenerate map -> alternate
            left = np.arange(len(X)) % 2 == 0
        return left

    rng = np.random.default_rng(seed)
    Xtr, ytr, Xte, yte, wn = [], [], [], [], None
    for i, (_cls, maps) in enumerate(groups):
        if progress:
            progress(f"loading & preprocessing '{_cls}'  ({i + 1}/{len(groups)})")
        feats = []
        for batch, path, role in maps:
            wn, X, coord = _feat_map(path)
            feats.append((batch, X, coord, role))

        if split == "random":                          # pool all pixels, shuffle
            allX = np.vstack([X for _b, X, _c, _r in feats])
            idx = rng.permutation(len(allX)); cut = len(allX) // 2
            Xtr.append(allX[idx[:cut]]); ytr += [i] * cut
            Xte.append(allX[idx[cut:]]); yte += [i] * (len(allX) - cut)
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
    return (np.vstack(Xtr), np.array(ytr),
            np.vstack(Xte), np.array(yte), wn, classes)


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
    return cm, acc, np.array(xs, float), np.array(ys, float), "OOB error", "trees"


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
            "training loss (cross-entropy)", "epoch")


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
            "test error", "training-set size")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
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
                deriv=0, norm="l2", split="spatial", progress=None) -> TrainResult:
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

    Xtr, ytr, Xte, yte, wn, classes = _load_split(
        pest_dir, baseline=baseline, trim=trim, deriv=deriv, norm=norm,
        split=split, seed=seed, progress=progress)
    K = len(classes)

    if backend == "resnet":
        cm, acc, cx, cy, ylab, xlab = _train_resnet(
            Xtr, ytr, Xte, yte, K, epochs=epochs, seed=seed, progress=progress)
    elif backend == "rf":
        cm, acc, cx, cy, ylab, xlab = _train_rf(
            Xtr, ytr, Xte, yte, K, n_estimators=n_estimators, seed=seed,
            progress=progress)
    else:
        cm, acc, cx, cy, ylab, xlab = _train_generic(
            _model_factory(backend, seed), Xtr, ytr, Xte, yte, K, seed=seed,
            progress=progress)

    if progress:
        progress("finalising — confusion, F1, PCA…")
    per = _per_class_prf(cm, classes)
    macro_f1 = float(np.mean([per[nm][2] for nm in classes]))

    # PCA of the real per-pixel spectra (subsample per class for a clean plot)
    Xall = np.vstack([Xtr, Xte]); yall = np.concatenate([ytr, yte])
    rng = np.random.default_rng(seed)
    sel = np.concatenate([
        rng.choice(np.where(yall == i)[0],
                   size=min(120, int((yall == i).sum())), replace=False)
        for i in range(K)])
    pca_emb = PCA(n_components=2, random_state=seed).fit_transform(Xall[sel])
    pca_lab = yall[sel]

    comps = [c for c in classes if not is_blank(c)]
    return TrainResult(
        backend=backend, classes=classes, comps=comps,
        confusion=cm, acc=acc, macro_f1=macro_f1, per_component=per,
        curve_x=cx, curve_y=cy, curve_label=ylab, curve_xlabel=xlab,
        pca_emb=pca_emb, pca_lab=pca_lab,
        n_train=len(ytr), n_test=len(yte), wn=wn, split=split)


if __name__ == "__main__":
    r = train_model(backend="rf")
    print(f"backend={r.backend}  classes={r.classes}")
    print(f"spatial acc={r.acc:.3f}  macro F1={r.macro_f1:.3f}")
    for nm in r.classes:
        p, rc, f, s = r.per_component[nm]
        print(f"  {nm:8s}  P={p:.2f} R={rc:.2f} F1={f:.2f}  (n={s})")
