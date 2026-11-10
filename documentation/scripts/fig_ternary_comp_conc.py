"""Combined ternary — composition AND concentration in one panel pair.

Geometry carries composition recovery (open = prepared ratio, filled = held-out
predicted ratio); fill colour carries concentration accuracy (mean min/max fold
of the held-out µM prediction, 1 = exact, 0.5 = two-fold). Grid64 conditions;
conditions sharing the same ratio (e.g. 3:3:3 and 24:24:24) are averaged and
the marker annotated implicitly by the caption. PLS vs MLP panels — the
composition geometry is similar, the concentration colour separates them.
"""
import os
import sys
import csv
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")))
import labfig
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

labfig.setup()
CO = labfig.CO
INK = "#20262e"
MUTE = "#8a919b"

RES = (r"S:/Google Drive/내 드라이브/github/Mixture Classifier/documentation/"
       r"results/origin_ready_20260825")
CONC = {(r["DQ_uM"], r["TBZ_uM"], r["THI_uM"]): r
        for r in csv.DictReader(open(os.path.join(
            RES, "09_grid64_concentration_accuracy_long.csv"),
            encoding="utf-8-sig"))
        for r in [{k: (int(v) if k.endswith("_uM") else v)
                   for k, v in r.items()}]}

PAT = re.compile(r"DQ(\d+)-TB[Z]?(\d+)-TH[I]?(\d+)")


def comp_held(name):
    from dl_model import load_model
    m = load_model(rf"S:/Google Drive/내 드라이브/ACF_PEST_DB/260826_Model/"
                   rf"{name}_composition_260826.dlm")
    cols = [m["subs"].index(s) for s in ("DQ", "TBZ", "THI")]
    out = {}
    for ev in (m.get("loo_eval") or {}, m.get("test_eval") or {}):
        for pth, pred in zip(ev.get("paths", []), ev.get("pred", [])):
            g = PAT.search(os.path.basename(pth))
            if not g:
                continue
            k = tuple(int(x) for x in g.groups())
            p = np.clip(np.asarray(pred, float)[cols], 0, None)
            out.setdefault(k, []).append(p / (p.sum() + 1e-12))
    return {k: np.mean(v, axis=0) for k, v in out.items()}


HELD = {m: comp_held(m) for m in ("pls", "mlp")}
GRID = [(d, t, h) for d in (3, 6, 12, 24) for t in (3, 6, 12, 24)
        for h in (3, 6, 12, 24)]

# group by ratio class (conditions that share a composition point)
groups = {}
for k in GRID:
    frac = tuple(np.round(np.array(k) / sum(k), 4))
    groups.setdefault(frac, []).append(k)
print(f"{len(GRID)} conditions → {len(groups)} unique composition positions")

V = {"THI": np.array([0.5, np.sqrt(3) / 2]),
     "TBZ": np.array([0.0, 0.0]),
     "DQ": np.array([1.0, 0.0])}
tri = np.array([V["TBZ"], V["DQ"], V["THI"], V["TBZ"]])
bary = lambda c: c[0] * V["DQ"] + c[1] * V["TBZ"] + c[2] * V["THI"]

norm = Normalize(0.4, 1.0)
cmap = cm.RdYlGn
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9))
for ax, (m, title) in zip(axes, [("pls", "PLS"), ("mlp", "MLP")]):
    ax.set_axis_off(); ax.set_aspect("equal")
    ax.plot(tri[:, 0], tri[:, 1], color=INK, lw=1.0, zorder=2)
    for f in (0.25, 0.5, 0.75):
        for a, b, c in ((V["TBZ"], V["DQ"], V["THI"]),
                        (V["DQ"], V["THI"], V["TBZ"]),
                        (V["THI"], V["TBZ"], V["DQ"])):
            p0 = a + f * (b - a); p1 = a + f * (c - a)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#e2e6ea", lw=0.5,
                    zorder=1)
    ax.text(*(V["THI"] + [0, 0.05]), "THI", color=CO["THI"], fontsize=10,
            weight="bold", ha="center")
    ax.text(*(V["TBZ"] + [-0.04, -0.05]), "TBZ", color=CO["TBZ"], fontsize=10,
            weight="bold", ha="right")
    ax.text(*(V["DQ"] + [0.04, -0.05]), "DQ", color=CO["DQ"], fontsize=10,
            weight="bold", ha="left")
    accs, devs = [], []
    for frac, ks in groups.items():
        t = np.array(frac)
        p = np.mean([HELD[m][k] for k in ks], axis=0)
        cacc = np.mean([float(CONC[k][f"conc_acc_{m}"]) for k in ks])
        accs.append(cacc); devs.append(50 * np.abs(p - t).sum())
        pt, pp = bary(t), bary(p)
        ax.plot([pt[0], pp[0]], [pt[1], pp[1]], color="#9aa3ad", lw=0.7,
                zorder=3)
        ax.scatter(*pt, s=24, facecolors="white", edgecolors="#6a7178",
                   linewidths=0.8, zorder=4)
        ax.scatter(*pp, s=34, facecolors=[cmap(norm(cacc))], edgecolors="white",
                   linewidths=0.7, zorder=5)
    ax.set_xlim(-0.14, 1.14); ax.set_ylim(-0.12, 1.02)
    ax.set_title(title, fontsize=10, weight="bold", pad=6)
    ax.text(0.5, -0.105,
            f"composition {np.mean(devs):.1f} %p · concentration accuracy "
            f"{np.mean(accs):.2f}", fontsize=8, color=MUTE, ha="center")

cax = fig.add_axes([0.435, 0.90, 0.13, 0.025])
cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                  orientation="horizontal", ticks=[0.4, 1.0])
cb.ax.set_xticklabels(["0.4", "1"], fontsize=7)
cb.outline.set_linewidth(0.5)
cax.set_title("concentration accuracy", fontsize=6.4, pad=2)
fig.text(0.5, 0.015,
         "grid64 held-out · geometry: prepared ○ → predicted ● composition · "
         "colour: µM accuracy (min/max fold, 1 = exact, 0.5 = two-fold) · "
         "conditions sharing a ratio averaged",
         fontsize=7.0, color=MUTE, ha="center")
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_ternary_comp_conc.{ext}"), dpi=600,
                bbox_inches="tight", pad_inches=0.02, facecolor="white")
print("saved fig_ternary_comp_conc")
