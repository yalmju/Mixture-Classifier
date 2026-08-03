# -*- coding: utf-8 -*-
"""Empirical composition-response surfaces from the same 34 held-out predictions.

Triangle coordinates are nominal compositions. Background RGB is the composition that
the method predicts at that nominal location, interpolated from LOO samples. White
curves mark changes in predicted dominant substance. This visualizes distortion; the
held-out composition error remains the independent numerical summary.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import RBFInterpolator

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "docs")
d = np.load(os.path.join(HERE, "tri_preds.npz"))
Y, P_NNLS, P_DL = d["Y"], d["Pn"], d["Pd"]       # columns: DQ, TBZ, THI
TRUE_DOM = np.argmax(Y, axis=1)
GROUP_HEX = ["#1a73e8", "#4a9e2a", "#d6336c"]
GROUP_RGB = np.array([[0.10, 0.45, 0.91], [0.29, 0.62, 0.16], [0.84, 0.20, 0.42]])
GROUP_NAME = ["DQ-dominant", "TBZ-dominant", "THI-dominant"]
H = np.sqrt(3.0) / 2.0
TOP, BL, BR = np.array([0.5, H]), np.array([0.0, 0.0]), np.array([1.0, 0.0])


def normalize(frac):
    f = np.clip(np.asarray(frac, float), 0, None)
    return f / (f.sum(axis=-1, keepdims=True) + 1e-12)


def bary(frac):
    f = normalize(frac)
    return f[..., 2, None] * TOP + f[..., 1, None] * BL + f[..., 0, None] * BR


def simplex_grid(res=560):
    gx = np.linspace(-0.02, 1.02, res); gy = np.linspace(-0.02, H + 0.02, res)
    xx, yy = np.meshgrid(gx, gy)
    thi = yy / H; dq = xx - 0.5 * thi; tbz = 1.0 - thi - dq
    inside = (dq >= 0) & (tbz >= 0) & (thi >= 0)
    nominal = np.stack([dq, tbz, thi], axis=-1)
    return gx, gy, xx, yy, inside, nominal


def response_surface(pred, nominal_grid, inside, smoothing=0.005):
    """Nominal composition -> held-out predicted composition via smooth RBF."""
    anchors = np.eye(3)                  # pure references must remain pure at the vertices
    x_train = bary(np.vstack([Y, anchors]))
    y_train = np.vstack([normalize(pred), anchors])
    rbf = RBFInterpolator(x_train, y_train, kernel="thin_plate_spline",
                          smoothing=float(smoothing))
    out = np.zeros(nominal_grid.shape, float)
    out[inside] = normalize(rbf(bary(nominal_grid[inside])))
    return out



def draw(method, pred, filename):
    pred = normalize(pred)
    gx, gy, xx, yy, inside, nominal = simplex_grid()
    surface = response_surface(pred, nominal, inside)
    rgb = surface @ GROUP_RGB
    rgb = 0.86 * rgb + 0.14                         # slight white lift for legibility
    image = np.dstack([rgb, np.where(inside, 0.94, 0.0)])

    fig, ax = plt.subplots(figsize=(7.2, 6.9))
    ax.imshow(image, extent=[gx[0], gx[-1], gy[0], gy[-1]], origin="lower",
              interpolation="bilinear", zorder=0)
    for t in (0.25, 0.5, 0.75):
        for p, q, r in ((BR, TOP, BL), (TOP, BR, BL), (BL, BR, TOP)):
            ax.plot([p[0] + t * (q[0] - p[0]), p[0] + t * (r[0] - p[0])],
                    [p[1] + t * (q[1] - p[1]), p[1] + t * (r[1] - p[1])],
                    color="white", lw=0.55, alpha=0.38, zorder=1)
    ax.plot([TOP[0], BL[0], BR[0], TOP[0]], [TOP[1], BL[1], BR[1], TOP[1]],
            color="#1c2430", lw=1.3, zorder=3)

    # Identical nominal sample positions in both figures; point colour gives true group.
    nominal_xy = bary(Y)
    for g in range(3):
        pts = nominal_xy[TRUE_DOM == g]
        ax.scatter(pts[:, 0], pts[:, 1], s=74, color=GROUP_HEX[g], alpha=0.74,
                   edgecolors="white", linewidths=0.9, zorder=4)
    for frac, text, dy, colour in (([0, 0, 1], "THI", 0.055, GROUP_HEX[2]),
                                   ([0, 1, 0], "TBZ", -0.060, GROUP_HEX[1]),
                                   ([1, 0, 0], "DQ", -0.060, GROUP_HEX[0])):
        p = bary(frac)
        ax.text(p[0], p[1] + dy, text, ha="center",
                va="bottom" if text == "THI" else "top", fontsize=16,
                fontweight="bold", color=colour)

    err = float(np.mean(0.5 * np.abs(pred - Y).sum(axis=1)))
    dominant = np.argmax(surface[inside], axis=1)
    areas = np.bincount(dominant, minlength=3) / len(dominant)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.14, 1.14); ax.set_ylim(-0.12, H + 0.10)
    handles = [Line2D([], [], marker="o", mfc=GROUP_HEX[g], mec="white", ls="",
                      ms=10, label=GROUP_NAME[g]) for g in range(3)]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10.5,
               framealpha=0, bbox_to_anchor=(0.5, 0.018))
    fig.text(0.5, 0.002, f"{method}  -  held-out mean composition error {err:.1%}",
             ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#1c2430")
    fig.savefig(os.path.join(OUT, filename), dpi=160, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    return err, areas


if __name__ == "__main__":
    e1, a1 = draw("NNLS", P_NNLS, "triangle_response_nnls.png")
    e2, a2 = draw("DL", P_DL, "triangle_response_dl.png")
    print(f"NNLS error {e1:.1%}, dominant areas DQ/TBZ/THI {a1.round(3)}")
    print(f"DL error {e2:.1%}, dominant areas DQ/TBZ/THI {a2.round(3)}")