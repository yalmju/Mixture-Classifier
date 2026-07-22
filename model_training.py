"""model_training.py — train a single-component classifier on the REAL pest
reference maps (DQ / THI / TBZ / BLK), for the UNMIXR "Model" page.

The Model page used to train on *synthetic* pure spectra (a demo that overlapped
the Discriminator page). This turns it into a genuine model-training tool that
learns from the real pesticide reference maps, with two selectable backends:

    "rf"      RandomForest on the per-pixel spectra  (scikit-learn, no torch).
              Learning curve = out-of-bag error as trees are added.
    "resnet"  ResNet1D deep classifier on the per-pixel spectra  (torch), with a
              live per-epoch training-loss curve — the "model learning" view the
              GUI never had before.

Both backends train on the SAME honest spatial (block) split the Discriminator
page uses — train on the left half of each map (by X coordinate), test on the
right half — so per-pixel accuracy is not inflated by adjacent-pixel leakage.

UI-agnostic: only numpy / scikit-learn are imported at module load. torch is
imported lazily inside the resnet path, so the RF backend (and importing this
module from the Qt app) works even when torch is not installed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from real_data import load_map, REF_FILES, CLASSES4, COMPS, PEST_DEFAULT
from sers_mixture import preprocess


@dataclass
class TrainResult:
    backend: str                     # "rf" or "resnet"
    classes: list                    # CLASSES4  (DQ / THI / TBZ / BLK)
    comps: list                      # COMPS     (the 3 detectable pesticides)
    confusion: np.ndarray            # (K, K) int, spatial split, rows=true cols=pred
    acc: float                       # spatial-split test accuracy
    macro_f1: float                  # unweighted mean of per-class F1
    per_component: dict              # name -> (precision, recall, f1, support)
    curve_x: np.ndarray              # epoch (resnet) or #trees (rf)
    curve_y: np.ndarray              # training loss (resnet) or OOB error (rf)
    curve_label: str                 # y-axis label for the learning curve
    curve_xlabel: str                # x-axis label for the learning curve
    pca_emb: np.ndarray              # (n_sample, 2) PCA of real per-pixel spectra
    pca_lab: np.ndarray              # (n_sample,) class index 0-3
    n_train: int
    n_test: int
    wn: np.ndarray = None            # wavenumber axis


# --------------------------------------------------------------------------
# data — real pest references, honest spatial (block) split
# --------------------------------------------------------------------------
def _resolve_ref_dir(pest_dir):
    """Find the folder that actually holds the reference CSVs. Accept either the
    Pest_Discriminator root (maps live in .../Reference/) or the Reference folder
    itself — so 'Training data…' works whichever of the two the user picks."""
    candidates = [os.path.join(pest_dir, "Reference"), pest_dir]
    for d in candidates:
        if all(os.path.exists(os.path.join(d, f)) for f in REF_FILES.values()):
            return d
    missing = [f for f in REF_FILES.values()
               if not os.path.exists(os.path.join(candidates[0], f))]
    raise FileNotFoundError(
        "reference CSVs not found. Pick the Pest_Discriminator folder (or its "
        f"Reference/ subfolder).\nlooked in: {candidates[0]}  and  {candidates[1]}"
        f"\nmissing: {missing}")


def _load_split(pest_dir):
    """Load the 4 reference maps and split each by its X-median into a
    train (left) / test (right) block. Returns preprocessed per-pixel matrices."""
    ref_dir = _resolve_ref_dir(pest_dir)

    Xtr, ytr, Xte, yte, wn = [], [], [], [], None
    for i, lab in enumerate(CLASSES4):
        wn, cube, _mean, coord = load_map(os.path.join(ref_dir, REF_FILES[lab]))
        Xp = preprocess(cube)
        xc = coord[:, 0]
        left = xc < np.median(xc)                     # spatial block split
        if left.all() or (~left).all():               # degenerate map -> random
            left = np.arange(len(Xp)) % 2 == 0
        Xtr.append(Xp[left]); ytr += [i] * int(left.sum())
        Xte.append(Xp[~left]); yte += [i] * int((~left).sum())
    return (np.vstack(Xtr), np.array(ytr),
            np.vstack(Xte), np.array(yte), wn)


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------
def _train_rf(Xtr, ytr, Xte, yte, n_estimators=300, seed=0):
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
    yp = rf.predict(Xte)
    cm = confusion_matrix(yte, yp, labels=range(len(CLASSES4)))
    acc = float(np.mean(yp == yte))
    return cm, acc, np.array(xs, float), np.array(ys, float), "OOB error", "trees"


def _train_resnet(Xtr, ytr, Xte, yte, epochs=25, batch_size=128, lr=1e-3,
                  base=16, seed=0):
    """ResNet1D 4-class classifier on per-pixel spectra; per-epoch loss curve."""
    import torch
    import torch.nn as nn
    from resnet1d import ResNet1D

    torch.manual_seed(seed)
    K = len(CLASSES4)
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

    net.eval()
    with torch.no_grad():
        logits = net(torch.tensor(np.asarray(Xte, np.float32)))
        yp = logits.argmax(1).cpu().numpy()
    cm = confusion_matrix(yte, yp, labels=range(K))
    acc = float(np.mean(yp == yte))
    return (cm, acc, np.array(xs, float), np.array(ys, float),
            "training loss (cross-entropy)", "epoch")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def _per_class_prf(cm):
    """Per-class (precision, recall, f1, support) from a confusion matrix."""
    per = {}
    for i, nm in enumerate(CLASSES4):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        per[nm] = (float(p), float(r), float(f), int(cm[i, :].sum()))
    return per


def train_model(pest_dir=PEST_DEFAULT, backend="rf", epochs=25,
                n_estimators=300, seed=0) -> TrainResult:
    """Load the real pest references and train the chosen backend on the honest
    spatial split. ``backend`` is "rf" (RandomForest) or "resnet" (ResNet1D)."""
    if backend == "resnet":
        try:
            import torch  # noqa: F401  (fail early with a clear message)
        except Exception as exc:
            raise RuntimeError(
                "ResNet1D backend needs PyTorch — install it (pip install torch) "
                "or pick the RandomForest backend.") from exc

    Xtr, ytr, Xte, yte, wn = _load_split(pest_dir)

    if backend == "resnet":
        cm, acc, cx, cy, ylab, xlab = _train_resnet(
            Xtr, ytr, Xte, yte, epochs=epochs, seed=seed)
    else:
        cm, acc, cx, cy, ylab, xlab = _train_rf(
            Xtr, ytr, Xte, yte, n_estimators=n_estimators, seed=seed)

    per = _per_class_prf(cm)
    macro_f1 = float(np.mean([per[nm][2] for nm in CLASSES4]))

    # PCA of the real per-pixel spectra (subsample per class for a clean plot)
    Xall = np.vstack([Xtr, Xte]); yall = np.concatenate([ytr, yte])
    rng = np.random.default_rng(seed)
    sel = np.concatenate([
        rng.choice(np.where(yall == i)[0],
                   size=min(120, int((yall == i).sum())), replace=False)
        for i in range(len(CLASSES4))])
    pca_emb = PCA(n_components=2, random_state=seed).fit_transform(Xall[sel])
    pca_lab = yall[sel]

    return TrainResult(
        backend=backend, classes=CLASSES4, comps=COMPS,
        confusion=cm, acc=acc, macro_f1=macro_f1, per_component=per,
        curve_x=cx, curve_y=cy, curve_label=ylab, curve_xlabel=xlab,
        pca_emb=pca_emb, pca_lab=pca_lab,
        n_train=len(ytr), n_test=len(yte), wn=wn)


if __name__ == "__main__":
    r = train_model(backend="rf")
    print(f"backend={r.backend}  spatial acc={r.acc:.3f}  macro F1={r.macro_f1:.3f}")
    print("per-class P/R/F1:")
    for nm in r.classes:
        p, rc, f, s = r.per_component[nm]
        print(f"  {nm:4s}  P={p:.2f} R={rc:.2f} F1={f:.2f}  (n={s})")
