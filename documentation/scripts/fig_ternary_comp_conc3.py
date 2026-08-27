"""Three-way combined ternary — NNLS | PLS | MLP on Grid64.

Each panel: geometry = composition recovery (open prepared → filled predicted),
colour (points + TPS background) = concentration accuracy (mean min/max fold,
1 = exact, 0.5 = two-fold), shared 0–1 scale.

Provenance per panel:
  NNLS — composition: apparent surface composition (locked workbook table);
         concentration: single-band calibration inversion (integrated tables;
         untrained, so no split is needed).
  PLS / MLP — composition and concentration from the deployed dlms' held-out
         records (LOO + independent test batch).
"""
import os
import sys
import csv
import re
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")))
import labfig
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy.interpolate import RBFInterpolator

labfig.setup()
CO = labfig.CO
INK = "#20262e"
MUTE = "#8a919b"

RES = (r"S:/Google Drive/내 드라이브/github/Mixture Classifier/documentation/"
       r"results")
PAT = re.compile(r"DQ(\d+)-TB[Z]?(\d+)-TH[I]?(\d+)")
trip = lambda nm: (lambda g: tuple(int(x) for x in g.groups()) if g else None)(
    PAT.search(nm))
GRID = [(d, t, h) for d in (3, 6, 12, 24) for t in (3, 6, 12, 24)
        for h in (3, 6, 12, 24)]

# ---- compositions ---------------------------------------------------------------
COMP = {}
wb = openpyxl.load_workbook(
    r"S:/Google Drive/내 드라이브/ACF_PEST_DB/260824/260824_Compoosition_data.xlsx",
    read_only=True, data_only=True)
nn = {}
for row in wb["grid64"].iter_rows(min_row=2, values_only=True):
    try:
        k = (int(row[1]), int(row[2]), int(row[3]))
        v = np.clip(np.array([float(row[9]), float(row[10]), float(row[11])]),
                    0, None)
    except (TypeError, ValueError):
        continue
    nn.setdefault(k, v / (v.sum() + 1e-12))
COMP["nnls"] = nn


def comp_held(name):
    from dl_model import load_model
    m = load_model(rf"S:/Google Drive/내 드라이브/ACF_PEST_DB/260826_Model/"
                   rf"{name}_composition_260826.dlm")
    cols = [m["subs"].index(s) for s in ("DQ", "TBZ", "THI")]
    out = {}
    for ev in (m.get("loo_eval") or {}, m.get("test_eval") or {}):
        for pth, pred in zip(ev.get("paths", []), ev.get("pred", [])):
            k = trip(os.path.basename(pth))
            if k:
                p = np.clip(np.asarray(pred, float)[cols], 0, None)
                out.setdefault(k, []).append(p / (p.sum() + 1e-12))
    return {k: np.mean(v, axis=0) for k, v in out.items()}


COMP["pls"] = comp_held("pls")
COMP["mlp"] = comp_held("mlp")

# ---- concentration accuracy per condition ---------------------------------------
RMSE = {}
for r in csv.DictReader(open(os.path.join(
        RES, "origin_ready_20260825", "09c_grid64_concentration_rmse_summary.csv"),
        encoding="utf-8-sig")):
    RMSE[r["method"]] = (float(r["RMSE_uM"]), float(r["within_2x_pct"]))

CACC = {"pls": {}, "mlp": {}}
for r in csv.DictReader(open(os.path.join(
        RES, "origin_ready_20260825", "09_grid64_concentration_accuracy_long.csv"),
        encoding="utf-8-sig")):
    k = (int(r["DQ_uM"]), int(r["TBZ_uM"]), int(r["THI_uM"]))
    CACC["pls"][k] = float(r["conc_acc_pls"])
    CACC["mlp"][k] = float(r["conc_acc_mlp"])
# NNLS = pure single-band calibration inversion from the integrated tables
conc = json.loads(open(os.path.join(
    RES, "integrated_final_20260824", "full", "concentration_results.json"),
    encoding="utf-8").read())
CACC["nnls"] = {}
for rec in conc["records"]:
    k = trip(rec["map"])
    if k in set(GRID):
        t = np.asarray(rec["true"], float)
        p = np.clip(np.asarray(rec["Ccal"], float), 1e-6, None)
        ok = t > 0
        fold = p[ok] / t[ok]
        CACC["nnls"][k] = float(np.minimum(fold, 1 / fold).mean())
print("conditions with NNLS conc:", len(CACC["nnls"]))

# ---- grouping by unique composition position ------------------------------------
groups = {}
for k in GRID:
    frac = tuple(np.round(np.array(k) / sum(k), 4))
    groups.setdefault(frac, []).append(k)

V = {"THI": np.array([0.5, np.sqrt(3) / 2]),
     "TBZ": np.array([0.0, 0.0]),
     "DQ": np.array([1.0, 0.0])}
tri = np.array([V["TBZ"], V["DQ"], V["THI"], V["TBZ"]])
bary = lambda c: c[0] * V["DQ"] + c[1] * V["TBZ"] + c[2] * V["THI"]

norm = Normalize(0.0, 1.0)
cmap = cm.RdYlGn
fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.75))
for ax, (m, title) in zip(axes, [("nnls", "NNLS"), ("pls", "PLS"),
                                 ("mlp", "MLP")]):
    ax.set_axis_off(); ax.set_aspect("equal")
    pos, accs, devs, pairs = [], [], [], []
    for frac, ks in groups.items():
        t = np.array(frac)
        ks_c = [k for k in ks if k in CACC[m]]
        if not ks_c:
            continue
        p = np.mean([COMP[m][k] for k in ks], axis=0)
        cacc = float(np.mean([CACC[m][k] for k in ks_c]))
        pos.append(bary(t)); accs.append(cacc)
        devs.append(50 * np.abs(p - t).sum())
        pairs.append((bary(t), bary(p)))
    pos = np.array(pos); accs = np.array(accs)
    ax.plot(tri[:, 0], tri[:, 1], color=INK, lw=1.0, zorder=2)
    for f in (0.25, 0.5, 0.75):
        for a, b, c in ((V["TBZ"], V["DQ"], V["THI"]),
                        (V["DQ"], V["THI"], V["TBZ"]),
                        (V["THI"], V["TBZ"], V["DQ"])):
            p0 = a + f * (b - a); p1 = a + f * (c - a)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#e2e6ea", lw=0.5,
                    zorder=1)
    ax.text(*(V["THI"] + [0, 0.05]), "THI", color=CO["THI"], fontsize=9.5,
            weight="bold", ha="center")
    ax.text(*(V["TBZ"] + [-0.04, -0.05]), "TBZ", color=CO["TBZ"], fontsize=9.5,
            weight="bold", ha="right")
    ax.text(*(V["DQ"] + [0.04, -0.05]), "DQ", color=CO["DQ"], fontsize=9.5,
            weight="bold", ha="left")
    for (pt, pp), a_ in zip(pairs, accs):
        ax.plot([pt[0], pp[0]], [pt[1], pp[1]], color="#6b737d", lw=0.7,
                zorder=3)
        ax.scatter(*pt, s=20, facecolors="white", edgecolors="#6a7178",
                   linewidths=0.7, zorder=4)
        ax.scatter(*pp, s=30, facecolors=[cmap(norm(a_))], edgecolors="white",
                   linewidths=0.6, zorder=5)
    ax.set_xlim(-0.14, 1.14); ax.set_ylim(-0.12, 1.02)
    ax.set_title(title, fontsize=10, weight="bold", pad=5)
    ax.text(0.5, -0.108,
            f"composition {np.mean(devs):.1f} %p · µM RMSE "
            f"{RMSE[m][0]:.1f} · within-2× {RMSE[m][1]:.0f}%",
            fontsize=7.6, color=MUTE, ha="center")

cax = fig.add_axes([0.645, 0.915, 0.085, 0.028])
cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                  orientation="horizontal", ticks=[0, 1])
cb.ax.set_xticklabels(["0", "1"], fontsize=6.5)
cb.outline.set_linewidth(0.5)
cax.set_title("concentration accuracy", fontsize=6.2, pad=2)
fig.text(0.5, 0.012,
         "grid64 held-out · geometry: prepared ○ → predicted ● composition · "
         "colour: µM accuracy (min/max fold) · NNLS concentration = "
         "single-band calibration inversion · stacked totals averaged",
         fontsize=6.8, color=MUTE, ha="center")
fig.tight_layout(rect=(0, 0.03, 1, 0.96))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_ternary_comp_conc3.{ext}"), dpi=600,
                bbox_inches="tight", pad_inches=0.02, facecolor="white")
print("saved fig_ternary_comp_conc3")
