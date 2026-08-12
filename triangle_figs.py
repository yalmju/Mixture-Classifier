"""triangle_figs.py — publication-style ternary figures, shared by the app's exports and
offline experiment scripts so both produce the SAME figures.

Two views of the same (true, predicted) composition pairs:

  accuracy_triangle  interior shaded by prediction accuracy (IDW over the samples),
                     true (open) -> predicted (filled) drift arrows, per-corner recovery ± SE
  rgb_triangle       interior coloured by composition itself (each substance an RGB
                     primary), so a balanced mixture sits on a neutral centre and the
                     drift towards a dominant corner is visible as a colour shift

Both take ``rows`` = list of (name, true_fracs, pred_fracs) with fractions ordered like
``subs``, and return a matplotlib Figure. UI-agnostic (no Qt).
"""
from __future__ import annotations

import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

INK = "#1c2430"
MUTE = "#5b6673"
FAINT = "#c7ccd2"
EMAX = 0.6                                    # composition error mapped to "fully wrong"


# Text size multiplier. These figures are also pulled out as standalone panels and
# dropped into a layout at a fraction of their natural size, where point-sized text
# comes out oversized. The app never sets this, so its figures are unchanged.
SCALE = 1.0


def _fs(pt):
    """Point-sized text/lines."""
    return pt * SCALE


def _ms(area):
    """Scatter areas are points SQUARED, so they take the square of the scale — using
    the linear factor left the markers swamping a shrunken triangle."""
    return area * SCALE * SCALE


def cns(ax):
    """The app's cnsplots form (same as ui_common.Canvas.style, without the Qt import):
    despine + thin 0.5pt black spines + tight black ticks. Kept here so exported figures
    look identical to the on-screen ones."""
    for name, s in ax.spines.items():
        if name in ("top", "right"):
            s.set_visible(False)
        else:
            s.set_color("black"); s.set_linewidth(0.5)
    ax.tick_params(colors="black", labelsize=_fs(9), length=2, width=0.6, pad=1)
    ax.xaxis.label.set_color("black"); ax.yaxis.label.set_color("black")
    ax.title.set_color("black")
    return ax


def _geom(n):
    """Corner positions: first substance bottom-right, second bottom-left, third top."""
    h = 3 ** 0.5 / 2
    return np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([0.5, h])


def _bary(frac, BR, BL, TOP):
    f = np.asarray(frac, float)
    s = f.sum()
    f = f / s if s > 0 else f
    return f[0] * BR + f[1] * BL + (f[2] * TOP if len(f) > 2 else 0.0)


def _frame(ax, BR, BL, TOP, subs, colors, corner_labels=None):
    for t in (0.25, 0.5, 0.75):
        for P, Q, R in [(BR, TOP, BL), (TOP, BR, BL), (BL, BR, TOP)]:
            ax.plot([P[0] + t * (Q[0] - P[0]), P[0] + t * (R[0] - P[0])],
                    [P[1] + t * (Q[1] - P[1]), P[1] + t * (R[1] - P[1])],
                    color="white", lw=0.7, alpha=0.6, zorder=1)
    ax.plot([TOP[0], BL[0], BR[0], TOP[0]], [TOP[1], BL[1], BR[1], TOP[1]],
            color=INK, lw=1.2, zorder=2)
    # labels pushed OUTSIDE the simplex so they never collide with the lines or points
    spots = [(BR, "left", "top", 0.03, -0.05), (BL, "right", "top", -0.03, -0.05),
             (TOP, "center", "bottom", 0.0, 0.05)]
    for i, (p, ha, va, dx, dy) in enumerate(spots[:len(subs)]):
        txt = subs[i] if not corner_labels else f"{subs[i]}\n{corner_labels[i]}"
        ax.text(p[0] + dx, p[1] + dy, txt, ha=ha, va=va, fontsize=_fs(15), fontweight="bold",
                color=colors[i], linespacing=1.3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.20, 1.20); ax.set_ylim(-0.26, 3 ** 0.5 / 2 + 0.18)


def _recovery(rows, i):
    """Mean recovery % (predicted/true) for substance i, with its standard error."""
    v = [r[2][i] / r[1][i] * 100 for r in rows if r[1][i] > 0.02]
    if not v:
        return float("nan"), 0.0
    v = np.asarray(v, float)
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return float(v.mean()), float(se)


def accuracy_triangle(rows, subs, colors, fig=None):
    """Interior shaded by prediction accuracy; arrows true -> predicted; corner recovery."""
    fig = fig or Figure(figsize=(7.2, 6.8))
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111)
    BR, BL, TOP = _geom(len(subs))
    cmap = matplotlib.colormaps["RdYlGn"]

    pts = np.array([_bary(r[1], BR, BL, TOP) for r in rows]) if rows else np.zeros((0, 2))
    acc = np.array([1 - min(0.5 * np.abs(np.asarray(r[2]) - np.asarray(r[1])).sum() / EMAX, 1.0)
                    for r in rows])
    if len(pts) >= 3:                                   # smooth accuracy field (IDW + blur)
        res = 260
        gx = np.linspace(-0.02, 1.02, res); gy = np.linspace(-0.02, TOP[1] + 0.02, res)
        GX, GY = np.meshgrid(gx, gy)
        det = (BL[1] - BR[1]) * (TOP[0] - BR[0]) + (BR[0] - BL[0]) * (TOP[1] - BR[1])
        a = ((BL[1] - BR[1]) * (GX - BR[0]) + (BR[0] - BL[0]) * (GY - BR[1])) / det
        b = ((BR[1] - TOP[1]) * (GX - BR[0]) + (TOP[0] - BR[0]) * (GY - BR[1])) / det
        inside = (a >= -1e-9) & (b >= -1e-9) & ((1 - a - b) >= -1e-9)
        gi = np.stack([GX.ravel(), GY.ravel()], 1)
        d2 = ((gi[:, None, 0] - pts[None, :, 0]) ** 2
              + (gi[:, None, 1] - pts[None, :, 1]) ** 2) + 3e-3
        w = 1.0 / d2 ** 0.9
        Z = ((w * acc[None, :]).sum(1) / w.sum(1)).reshape(GX.shape)
        try:
            from scipy.ndimage import gaussian_filter
            Z = gaussian_filter(Z, sigma=7)
        except Exception:
            pass
        ax.pcolormesh(GX, GY, np.ma.masked_where(~inside, Z), cmap=cmap, vmin=0, vmax=1,
                      alpha=0.42, shading="gouraud", zorder=0)

    corner = []
    for i in range(len(subs)):
        rec, se = _recovery(rows, i)
        corner.append("—" if rec != rec else f"{rec:.0f}±{se:.0f}%")
    _frame(ax, BR, BL, TOP, subs, colors, corner_labels=corner)

    for r in rows:
        p0 = _bary(r[1], BR, BL, TOP); p1 = _bary(r[2], BR, BL, TOP)
        e = 0.5 * np.abs(np.asarray(r[2]) - np.asarray(r[1])).sum()
        if np.hypot(*(p1 - p0)) > 4e-3:                 # skip only true zero-length
            ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=_fs(16),
                                         color="#3f4650", lw=1.8, alpha=0.9, zorder=3,
                                         shrinkA=3, shrinkB=6))
        ax.scatter(*p0, s=_ms(44), facecolors="white", edgecolors=MUTE, linewidths=_fs(1.2), zorder=4)
        ax.scatter(*p1, s=_ms(76), color=cmap(1 - min(e / EMAX, 1.0)), edgecolors="white",
                   linewidths=_fs(0.7), zorder=5)

    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), ax=ax,
                      fraction=0.03, pad=0.02, shrink=0.5, aspect=22)
    cb.set_label("accuracy (green = exact, red = poor)", fontsize=_fs(10))
    cb.set_ticks([0, 0.5, 1.0]); cb.ax.tick_params(labelsize=_fs(9), colors="black",
                                                   length=2, width=0.6)
    cb.outline.set_linewidth(0.5); cb.outline.set_edgecolor("black")
    fig.legend(handles=[Line2D([], [], marker="o", mfc="white", mec=MUTE, mew=1.2, ls="",
                               ms=9, label="true composition"),
                        Line2D([], [], marker="o", mfc="#4a9e2a", mec="white", ls="",
                               ms=10, label="predicted (colour = accuracy)")],
               loc="lower center", ncol=2, fontsize=_fs(11), framealpha=0,
               bbox_to_anchor=(0.5, -0.06))
    return fig


def rgb_triangle(rows, subs, colors, fig=None):
    """Interior coloured by composition itself; arrows true -> predicted."""
    fig = fig or Figure(figsize=(7.0, 6.8))
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111)
    BR, BL, TOP = _geom(len(subs))
    prim = [np.array(matplotlib.colors.to_rgb(c)) for c in colors]

    res = 460
    gx = np.linspace(-0.02, 1.02, res); gy = np.linspace(-0.02, TOP[1] + 0.02, res)
    GX, GY = np.meshgrid(gx, gy)
    det = (BL[1] - BR[1]) * (TOP[0] - BR[0]) + (BR[0] - BL[0]) * (TOP[1] - BR[1])
    wt = ((BL[1] - BR[1]) * (GX - BR[0]) + (BR[0] - BL[0]) * (GY - BR[1])) / det   # -> BR
    wl = ((BR[1] - TOP[1]) * (GX - BR[0]) + (TOP[0] - BR[0]) * (GY - BR[1])) / det  # -> BL
    wtop = 1 - wt - wl
    inside = (wt >= -1e-9) & (wl >= -1e-9) & (wtop >= -1e-9)
    img = np.zeros((res, res, 4))
    for w, col in ((wt, prim[0]), (wl, prim[1]), (wtop, prim[2] if len(prim) > 2 else prim[0])):
        for k in range(3):
            img[:, :, k] += np.clip(w, 0, None) * col[k]
    img[:, :, :3] = np.clip(img[:, :, :3], 0.0, 1.0)          # blends can exceed 1
    img[:, :, 3] = np.where(inside, 0.62, 0.0)
    ax.imshow(img, extent=[gx[0], gx[-1], gy[0], gy[-1]], origin="lower", zorder=0,
              interpolation="bilinear")

    _frame(ax, BR, BL, TOP, subs, colors)
    for r in rows:
        p0 = _bary(r[1], BR, BL, TOP); p1 = _bary(r[2], BR, BL, TOP)
        if np.hypot(*(p1 - p0)) > 4e-3:
            ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=_fs(16),
                                         color="#1c2430", lw=1.7, alpha=0.9, zorder=3,
                                         shrinkA=3, shrinkB=6))
        ax.scatter(*p0, s=_ms(48), facecolors="white", edgecolors=INK, linewidths=_fs(1.3), zorder=4)
        ax.scatter(*p1, s=_ms(72), facecolors=INK, edgecolors="white", linewidths=_fs(0.9), zorder=5)
    fig.legend(handles=[Line2D([], [], marker="o", mfc="white", mec=INK, mew=1.3, ls="",
                               ms=9, label="true (solution) composition"),
                        Line2D([], [], marker="o", mfc=INK, mec="white", ls="", ms=9,
                               label="predicted (measured) composition")],
               loc="lower center", ncol=2, fontsize=_fs(11), framealpha=0,
               bbox_to_anchor=(0.5, -0.06))
    return fig


def compare_triangles(rows_a, rows_b, subs, colors, labels=("NNLS", "model"), fig=None):
    """Two accuracy ternaries side by side on ONE figure — the classical unmixing next to
    the trained model, drawn from the same mixtures so the drift arrows are comparable."""
    fig = fig or Figure(figsize=(13.0, 6.6))
    fig.patch.set_alpha(0.0)
    axes = fig.subplots(1, 2)
    cmap = matplotlib.colormaps["RdYlGn"]
    for ax, rows, lab in zip(axes, (rows_a, rows_b), labels):
        BR, BL, TOP = _geom(len(subs))
        pts = np.array([_bary(r[1], BR, BL, TOP) for r in rows]) if rows else np.zeros((0, 2))
        acc = np.array([1 - min(0.5 * np.abs(np.asarray(r[2]) - np.asarray(r[1])).sum() / EMAX, 1.0)
                        for r in rows])
        if len(pts) >= 3:
            res = 220
            gx = np.linspace(-0.02, 1.02, res); gy = np.linspace(-0.02, TOP[1] + 0.02, res)
            GX, GY = np.meshgrid(gx, gy)
            det = (BL[1] - BR[1]) * (TOP[0] - BR[0]) + (BR[0] - BL[0]) * (TOP[1] - BR[1])
            a = ((BL[1] - BR[1]) * (GX - BR[0]) + (BR[0] - BL[0]) * (GY - BR[1])) / det
            b = ((BR[1] - TOP[1]) * (GX - BR[0]) + (TOP[0] - BR[0]) * (GY - BR[1])) / det
            inside = (a >= -1e-9) & (b >= -1e-9) & ((1 - a - b) >= -1e-9)
            gi = np.stack([GX.ravel(), GY.ravel()], 1)
            d2 = ((gi[:, None, 0] - pts[None, :, 0]) ** 2
                  + (gi[:, None, 1] - pts[None, :, 1]) ** 2) + 3e-3
            w = 1.0 / d2 ** 0.9
            Z = ((w * acc[None, :]).sum(1) / w.sum(1)).reshape(GX.shape)
            try:
                from scipy.ndimage import gaussian_filter
                Z = gaussian_filter(Z, sigma=6)
            except Exception:
                pass
            ax.pcolormesh(GX, GY, np.ma.masked_where(~inside, Z), cmap=cmap, vmin=0, vmax=1,
                          alpha=0.42, shading="gouraud", zorder=0)
        corner = []
        for i in range(len(subs)):
            rec, se = _recovery(rows, i)
            corner.append("—" if rec != rec else f"{rec:.0f}±{se:.0f}%")
        _frame(ax, BR, BL, TOP, subs, colors, corner_labels=corner)
        for r in rows:
            p0 = _bary(r[1], BR, BL, TOP); p1 = _bary(r[2], BR, BL, TOP)
            e = 0.5 * np.abs(np.asarray(r[2]) - np.asarray(r[1])).sum()
            if np.hypot(*(p1 - p0)) > 4e-3:
                ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=_fs(15),
                                             color="#3f4650", lw=1.6, alpha=0.9, zorder=3,
                                             shrinkA=3, shrinkB=6))
            ax.scatter(*p0, s=_ms(40), facecolors="white", edgecolors=MUTE, linewidths=_fs(1.1), zorder=4)
            ax.scatter(*p1, s=_ms(70), color=cmap(1 - min(e / EMAX, 1.0)), edgecolors="white",
                       linewidths=_fs(0.7), zorder=5)
        err = float(np.mean([0.5 * np.abs(np.asarray(r[2]) - np.asarray(r[1])).sum()
                             for r in rows])) if rows else float("nan")
        ax.text(0.5, -0.20, f"{lab}   —   mean composition error {err:.0%}",
                transform=ax.transAxes, ha="center", va="top", fontsize=_fs(13),
                fontweight="bold", color=INK)
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), ax=axes,
                      fraction=0.016, pad=0.02, shrink=0.5, aspect=22)
    cb.set_label("accuracy (green = exact, red = poor)", fontsize=_fs(10))
    cb.set_ticks([0, 0.5, 1.0]); cb.ax.tick_params(labelsize=_fs(9), colors="black",
                                                   length=2, width=0.6)
    cb.outline.set_linewidth(0.5); cb.outline.set_edgecolor("black")
    fig.legend(handles=[Line2D([], [], marker="o", mfc="white", mec=MUTE, mew=1.2, ls="",
                               ms=9, label="true composition"),
                        Line2D([], [], marker="o", mfc="#4a9e2a", mec="white", ls="",
                               ms=10, label="predicted (colour = accuracy)")],
               loc="lower center", ncol=2, fontsize=_fs(11), framealpha=0,
               bbox_to_anchor=(0.5, -0.06))
    return fig
