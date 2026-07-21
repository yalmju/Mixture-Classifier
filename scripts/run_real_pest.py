"""run_real_pest.py — run the mixture pipeline on the REAL pesticide SERS data.

Loads the DQ / THI / TBZ pure references and the binary ratio mixtures from the
Pest_Discriminator project (20x20 hyperspectral maps, 2001-pt axis, baseline-
corrected), trains the mixture classifier on the pure means, and reports:

    - PCA of the real per-pixel spectra (pures + blank), coloured by compound
    - detection: which pesticides the classifier calls in each ratio mixture
    - multilabel precision / recall / F1 on the mixture set
    - NNLS signal-composition of each binary mixture vs its nominal ratio
      (deviation = competitive adsorption + differing SERS response)

    python run_real_pest.py

Writes real_pest_report.png next to this file and prints a text report.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from sers_mixture import (preprocess, SERSMixtureClassifier, AugmentConfig,
                          multilabel_metrics, names_to_indicator)
from competitive import fit_B

PEST = r"S:\Google Drive\내 드라이브\github\Pest_Discriminator"
COMPS = ["DQ", "THI", "TBZ"]
SERIES = {"DQ": "#1a73e8", "THI": "#0f9d6b", "TBZ": "#d8542a", "blank": "#98a1ac"}

# ratio maps -> nominal concentration over [DQ, THI, TBZ]
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


def main():
    # ---- load pure references + blank ----
    wn, _, dq = load_map(os.path.join(PEST, "Reference", "DQ_corrected.csv"))
    _, _, thi = load_map(os.path.join(PEST, "Reference", "THI_corrected.csv"))
    _, _, tbz = load_map(os.path.join(PEST, "Reference", "TBZ_corrected.csv"))
    _, blk_cube, blk = load_map(os.path.join(PEST, "Reference", "blk_corrected.csv"))
    pure_raw = np.vstack([dq, thi, tbz])
    pures = preprocess(pure_raw)
    print(f"loaded refs: axis {wn[0]:.0f}-{wn[-1]:.0f} cm^-1, {len(wn)} points")

    # ---- train on the 3 pure means ----
    clf = SERSMixtureClassifier(COMPS, prob_threshold=0.30, max_components=3,
                                augment=AugmentConfig(n_per_pure=200))
    clf.fit(pures)

    # ---- load mixtures ----
    mix_names, mix_mean, mix_true, mix_cubes = [], [], [], []
    for key, nominal in MIXES.items():
        p = os.path.join(PEST, "Ratio", key + "_corrected.csv")
        if not os.path.exists(p):
            print("  (missing)", key); continue
        _, cube, mean = load_map(p)
        mix_names.append(key); mix_mean.append(mean); mix_cubes.append(cube)
        mix_true.append([COMPS[i] for i, c in enumerate(nominal) if c > 0])
    mix_mean = preprocess(np.array(mix_mean))

    # ---- detection + metrics ----
    pred = clf.predict(mix_mean)
    y_true = names_to_indicator(mix_true, COMPS)
    y_pred = names_to_indicator(pred, COMPS)
    micro = multilabel_metrics(y_true, y_pred)

    print("\n=== detection on real ratio mixtures ===")
    print(f"{'mixture':10s} {'nominal':16s} {'detected':16s}  ok")
    for nm, t, p in zip(mix_names, mix_true, pred):
        ok = "OK" if set(t) == set(p) else "x"
        print(f"{nm:10s} {'+'.join(t):16s} {'+'.join(p):16s}  {ok}")
    print(f"\nmicro  P={micro['micro_precision']:.2f} R={micro['micro_recall']:.2f} "
          f"F1={micro['micro_f1']:.2f}  exact={micro['exact_match_ratio']:.2f}")

    # ---- NNLS signal composition vs nominal (binary mixtures) ----
    print("\n=== recovered signal ratio vs nominal (binary mixtures) ===")
    ratio_rows = []
    for nm, mean, nominal in zip(mix_names, mix_mean, [MIXES[k] for k in mix_names]):
        B, _ = fit_B(mean, pures)
        comp = B / (B.sum() + 1e-12)
        nom = np.array(nominal, float); nom = nom / nom.sum()
        ratio_rows.append((nm, comp, nom))
        pieces = "  ".join(f"{COMPS[i]} {comp[i]*100:4.0f}%/{nom[i]*100:3.0f}%"
                           for i in range(3) if nom[i] > 0)
        print(f"{nm:10s} (recovered/nominal)  {pieces}")

    # ---- PCA of real per-pixel spectra ----
    fig = Figure(figsize=(12, 8), dpi=110); fig.patch.set_facecolor("white")

    ax1 = fig.add_subplot(2, 2, 1)
    for i, nm in enumerate(COMPS):
        ax1.plot(wn, pures[i] + i * 0.5, color=SERIES[nm], lw=1.0, label=nm)
    ax1.set_xlim(400, 1800); ax1.set_yticks([])
    ax1.set_xlabel("wavenumber (cm$^{-1}$)"); ax1.set_title("real pure references")
    ax1.legend(fontsize=8)

    # PCA on a subsample of pixels per compound
    ax2 = fig.add_subplot(2, 2, 2)
    pix_X, pix_y = [], []
    maps = {"DQ": os.path.join(PEST, "Reference", "DQ_corrected.csv"),
            "THI": os.path.join(PEST, "Reference", "THI_corrected.csv"),
            "TBZ": os.path.join(PEST, "Reference", "TBZ_corrected.csv"),
            "blank": os.path.join(PEST, "Reference", "blk_corrected.csv")}
    rng = np.random.default_rng(0)
    for nm, path in maps.items():
        _, cube, _ = load_map(path)
        idx = rng.choice(len(cube), size=min(120, len(cube)), replace=False)
        Xp = preprocess(cube[idx])
        pix_X.append(Xp); pix_y += [nm] * len(idx)
    pix_X = np.vstack(pix_X)
    emb = PCA(n_components=2, random_state=0).fit_transform(pix_X)
    for nm in maps:
        m = np.array([y == nm for y in pix_y])
        ax2.scatter(emb[m, 0], emb[m, 1], s=10, color=SERIES[nm], alpha=0.6, label=nm)
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
    ax2.set_title("PCA of real per-pixel spectra"); ax2.legend(fontsize=8)

    # confusion-ish detection grid
    ax3 = fig.add_subplot(2, 2, 3)
    M = np.zeros((len(mix_names), 3))
    for r, p in enumerate(pred):
        for c, nm in enumerate(COMPS):
            M[r, c] = 1 if nm in p else 0
    ax3.imshow(M, cmap="Greens", aspect="auto", vmin=0, vmax=1)
    ax3.set_xticks(range(3)); ax3.set_xticklabels(COMPS)
    ax3.set_yticks(range(len(mix_names))); ax3.set_yticklabels(mix_names, fontsize=7)
    ax3.set_title(f"detected components  (F1={micro['micro_f1']:.2f})")
    for r, t in enumerate(mix_true):
        for c, nm in enumerate(COMPS):
            if nm in t:
                ax3.add_patch(matplotlib.patches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="#d8542a", lw=1.6))

    # recovered vs nominal ratio (binary)
    ax4 = fig.add_subplot(2, 2, 4)
    xs, ys, cs = [], [], []
    for nm, comp, nom in ratio_rows:
        for i in range(3):
            if nom[i] > 0:
                xs.append(nom[i]); ys.append(comp[i]); cs.append(SERIES[COMPS[i]])
    ax4.scatter(xs, ys, c=cs, s=40)
    ax4.plot([0, 1], [0, 1], ls="--", color="#98a1ac", lw=1)
    ax4.set_xlabel("nominal fraction"); ax4.set_ylabel("recovered signal fraction")
    ax4.set_title("ratio: recovered vs nominal")

    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "real_pest_report.png")
    fig.savefig(out, dpi=130, facecolor="white")
    print("\nsaved figure ->", out)


if __name__ == "__main__":
    import matplotlib.patches  # noqa (used in ax3)
    main()
