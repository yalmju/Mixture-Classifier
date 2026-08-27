"""Preview of the sanctioned high-concentration ternary pair (Origin data 06b):
NNLS apparent surface composition vs held-out MLP (DL05) on the 25 conditions
with any component > 100 uM (>=100-fold imbalance excluded upstream).
Final figure is plotted in Origin from the same CSV; this is a check render.
"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labfig
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy.interpolate import griddata

labfig.setup()
CO = labfig.CO
INK = "#20262e"
MUTE = "#8a919b"

CSV = (r"S:/Google Drive/내 드라이브/github/Mixture Classifier/documentation/"
       r"results/origin_ready_20260825/06b_high_concentration_nnls_vs_mlp_ternary.csv")
rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
def vec(r, pre):
    v = np.array([float(r[pre + "_DQ_pct"]), float(r[pre + "_TBZ_pct"]),
                  float(r[pre + "_THI_pct"])])
    return np.clip(v, 0, None) / (np.clip(v, 0, None).sum() + 1e-12)
TRUE = np.stack([vec(r, "true") for r in rows])
PRED = {"nnls": np.stack([vec(r, "nnls") for r in rows]),
        "mlp": np.stack([vec(r, "mlp") for r in rows])}
print(len(TRUE), "conditions")

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


def draw(points):
    tag = "" if points else "_surface"
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9))
    for ax, (m, title) in zip(axes, [("nnls", "NNLS unmixing"),
                                     ("mlp", "MLP unmixing")]):
        ax.set_axis_off(); ax.set_aspect("equal")
        acc = 1.0 - 0.5 * np.abs(PRED[m] - TRUE).sum(1)
        pts = np.array([bary(t) for t in TRUE])
        gx, gy = np.meshgrid(np.linspace(0, 1, 320),
                             np.linspace(0, np.sqrt(3) / 2, 280))
        lin = griddata(pts, acc, (gx, gy), method="linear")
        near = griddata(pts, acc, (gx, gy), method="nearest")
        surf = np.where(np.isnan(lin), near, lin)
        im = ax.imshow(surf, extent=(0, 1, 0, np.sqrt(3) / 2), origin="lower",
                       cmap=cmap, norm=norm, alpha=0.55 if points else 0.85,
                       zorder=0.5, interpolation="bilinear")
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
                ax.annotate("", pp, pt, arrowprops=dict(
                    arrowstyle="-|>", color="#5b636d", lw=0.7,
                    mutation_scale=6, shrinkA=2.5, shrinkB=2.5), zorder=3)
                ax.scatter(*pt, s=24, facecolors="white",
                           edgecolors="#6a7178", linewidths=0.8, zorder=4)
                ax.scatter(*pp, s=28, facecolors=[cmap(norm(a_))],
                           edgecolors=INK, linewidths=0.5, zorder=5)
        ax.set_xlim(-0.14, 1.14); ax.set_ylim(-0.12, 1.02)
        ax.set_title(title, fontsize=10, weight="bold", pad=6)
        ax.text(0.5, -0.105, f"mean deviation {np.mean(1-acc)*100:.1f} %p",
                fontsize=8, color=MUTE, ha="center")
    cax = fig.add_axes([0.435, 0.90, 0.13, 0.025])
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal", ticks=[0, 1])
    cb.ax.set_xticklabels(["0", "1"], fontsize=7)
    cb.outline.set_linewidth(0.5)
    cax.set_title("accuracy", fontsize=7, pad=2)
    note = ("25 conditions with any component > 100 uM · NNLS: apparent surface "
            "composition · MLP: held-out (DL05, origin package 06)")
    if points:
        note += " · open ○ = prepared, filled ● = predicted"
    fig.text(0.5, 0.015, note, fontsize=7.5, color=MUTE, ha="center")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "figures")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_ternary_highconc25{tag}.{ext}"),
                    dpi=600, bbox_inches="tight", pad_inches=0.02,
                    facecolor="white")
    plt.close(fig)


draw(points=True)
draw(points=False)
print("saved fig_ternary_highconc25 (+_surface)")
