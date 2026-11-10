"""Full-Grid64 ternary pair drawn from the LOCKED workbook prediction table
(260824_Compoosition_data.xlsx, sheet 'grid64': true uM in B-D, NNLS % in J-L,
MLP % in N-P) — the same numbers behind the manuscript's reference panels, so
this figure is consistent with the (b)(iii) recovery table by construction.

Two versions: arrows+points over the interpolated accuracy surface, and the
surface alone.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labfig
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy.interpolate import RBFInterpolator
import openpyxl

labfig.setup()
CO = labfig.CO
INK = "#20262e"
MUTE = "#8a919b"

XLSX = r"S:/Google Drive/내 드라이브/ACF_PEST_DB/260824/260824_Compoosition_data.xlsx"
wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
ws = wb["grid64"]
TRUE, NNLS, MLP = [], [], []
for row in ws.iter_rows(min_row=2, values_only=True):
    try:
        t = np.array([float(row[1]), float(row[2]), float(row[3])])
        nn = np.array([float(row[9]), float(row[10]), float(row[11])])
    except (TypeError, ValueError):
        continue
    TRUE.append(t / t.sum())
    NNLS.append(np.clip(nn, 0, None) / (np.clip(nn, 0, None).sum() + 1e-12))
    MLP.append(tuple(int(v) for v in (row[1], row[2], row[3])))
TRUE = np.stack(TRUE); PRED = {"nnls": np.stack(NNLS)}
# MLP side: HELD-OUT predictions of the deployed model (the sheet's own MLP
# columns are in-sample — mean deviation 1.4 pp — and must not be shown).
import re
zc = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "results", "ternary_grid64_cache.npz"),
             allow_pickle=True)
def key_from_name(nm):
    m = re.match(r"DQ(\d+)-TB(\d+)-THI?(\d+)", str(nm))
    return tuple(int(g) for g in m.groups())
cache = {key_from_name(nm): p_ for nm, p_ in zip(zc["names"], zc["mlp"])}
keys = []
for row in MLP:
    keys.append(row)
PRED["mlp"] = np.stack([cache[k] for k in keys])
print(f"{len(TRUE)} grid conditions · NNLS from workbook sheet, "
      f"MLP held-out from deployed dlm")

SUBS = ["DQ", "TBZ", "THI"]
V = {"THI": np.array([0.5, np.sqrt(3) / 2]),
     "TBZ": np.array([0.0, 0.0]),
     "DQ": np.array([1.0, 0.0])}
tri = np.array([V["TBZ"], V["DQ"], V["THI"], V["TBZ"]])


def bary(c):
    c = np.clip(np.asarray(c, float), 0, None)
    c = c / (c.sum() + 1e-12)
    return c[0] * V["DQ"] + c[1] * V["TBZ"] + c[2] * V["THI"]


norm = Normalize(0.0, 1.0)
cmap = cm.RdYlGn


def draw(points, arrows=True):
    tag = ("" if arrows else "_points") if points else "_surface"
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9))
    for ax, (m, title) in zip(axes, [("nnls", "NNLS unmixing"),
                                     ("mlp", "MLP unmixing")]):
        ax.set_axis_off(); ax.set_aspect("equal")
        # reference convention: accuracy = 1 - total L1 composition error
        acc = np.clip(1.0 - np.abs(PRED[m] - TRUE).sum(1), 0, 1)
        pts = np.array([bary(t) for t in TRUE])
        gx, gy = np.meshgrid(np.linspace(0, 1, 320),
                             np.linspace(0, np.sqrt(3) / 2, 280))
        rbf = RBFInterpolator(pts, acc, kernel="thin_plate_spline",
                              smoothing=0.015)
        surf = np.clip(rbf(np.column_stack([gx.ravel(), gy.ravel()])
                           ).reshape(gx.shape), 0, 1)
        im = ax.imshow(surf, extent=(0, 1, 0, np.sqrt(3) / 2), origin="lower",
                       cmap=cmap, norm=norm, alpha=0.80,
                       zorder=0.5, interpolation="bicubic")
        im.set_clip_path(PathPatch(Path(tri[:3]), transform=ax.transData))
        ax.plot(tri[:, 0], tri[:, 1], color=INK, lw=1.0, zorder=2)
        for f in (0.25, 0.5, 0.75):
            for a, b, c in ((V["TBZ"], V["DQ"], V["THI"]),
                            (V["DQ"], V["THI"], V["TBZ"]),
                            (V["THI"], V["TBZ"], V["DQ"])):
                p0 = a + f * (b - a); p1 = a + f * (c - a)
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="white",
                        lw=0.6, alpha=0.7, zorder=1)
        ax.text(*(V["THI"] + [0, 0.05]), "THI", color=CO["THI"], fontsize=10,
                weight="bold", ha="center")
        ax.text(*(V["TBZ"] + [-0.04, -0.05]), "TBZ", color=CO["TBZ"],
                fontsize=10, weight="bold", ha="right")
        ax.text(*(V["DQ"] + [0.04, -0.05]), "DQ", color=CO["DQ"], fontsize=10,
                weight="bold", ha="left")
        if points:
            for t, p, a_ in zip(TRUE, PRED[m], acc):
                pt, pp = bary(t), bary(p)
                if arrows:
                    ax.annotate("", pp, pt, arrowprops=dict(
                        arrowstyle="-|>", color="#5b636d", lw=0.7,
                        mutation_scale=6, shrinkA=2.5, shrinkB=2.5), zorder=3)
                ax.scatter(*pt, s=24, facecolors="white",
                           edgecolors="#6a7178", linewidths=0.8, zorder=4)
                ax.scatter(*pp, s=28, facecolors=[cmap(norm(a_))],
                           edgecolors=INK, linewidths=0.5, zorder=5)
        ax.set_xlim(-0.14, 1.14); ax.set_ylim(-0.12, 1.02)
        ax.set_title(title, fontsize=10, weight="bold", pad=6)
        dev = 0.5 * np.abs(PRED[m] - TRUE).sum(1)
        ax.text(0.5, -0.105, f"mean deviation {np.mean(dev)*100:.1f} %p",
                fontsize=8, color=MUTE, ha="center")
    cax = fig.add_axes([0.435, 0.90, 0.13, 0.025])
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal", ticks=[0, 1])
    cb.ax.set_xticklabels(["0", "1"], fontsize=7)
    cb.outline.set_linewidth(0.5)
    cax.set_title(r"accuracy = 1 $-$ $\Sigma$|pred$-$true|", fontsize=6.2, pad=2)
    note = ("NNLS: apparent surface composition (locked workbook table) · "
            "MLP: condition-held-out, deployed model · all 64 grid conditions")
    if points:
        note += " · open ○ = prepared, filled ● = predicted"
    fig.text(0.5, 0.015, note, fontsize=7.5, color=MUTE, ha="center")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "figures")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_ternary_grid64{tag}.{ext}"),
                    dpi=600, bbox_inches="tight", pad_inches=0.02,
                    facecolor="white")
    plt.close(fig)


draw(points=True)
draw(points=True, arrows=False)
draw(points=False)
print("saved fig_ternary_grid64 (+_surface)")
