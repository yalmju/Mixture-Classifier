"""grid_figs.py — condition-grid figures for a whole experiment at once.

The Compose panel scores every mixture leave-one-condition-out but could only ever
show one number per run. These draw the whole set:

    composition_pies    one predicted pie per condition
    grid_truepred       the design grid, true vs predicted bars in every cell
    grid_error          the design grid as an error heat-map
    pixel_maps          one 10x10 per-pixel composition map per condition

Same conventions as triangle_figs: pyplot is never imported (the app draws on its own
Figure), colours are passed in rather than hardcoded, and the background stays
transparent so an export drops onto any slide.

ERROR COLOUR SCALE — the ends mean something.
    green end   the repeat-scatter floor (``floor``): the spread between preparations
                of the SAME mixture. At or below it the model is as right as the
                measurement can show, so it is fully green.
    red  end    the uninformative baseline (``null_error``): the error you get by
                ignoring the spectrum and answering "equal parts" every time. Red
                therefore means "no better than guessing", not "30%".
A fixed 0-30% ramp was used before and it painted honest results red: 0% is
unreachable (the floor is ~10%) and 30% is not a threshold of anything.
"""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Wedge, Patch
from matplotlib.collections import PatchCollection

INK = "#1c2430"
MUTE = "#5b6673"
BG_GREY = "#c9ced6"
RED = "#c0392f"
AMBER = "#c8791f"

DEFAULT_FLOOR = 9.6          # %; preparation-to-preparation scatter measured on replicates


# ----------------------------------------------------------------- helpers
def comp_error(true, pred):
    """Composition distance in % — half the L1 gap, the number used everywhere else."""
    return float(np.abs(np.asarray(pred, float) - np.asarray(true, float)).sum() / 2)


def null_error(trues):
    """Mean error of the uninformative answer: equal parts, every time.

    This is the red end of the scale. It is computed from the true compositions of the
    experiment being drawn, so a design that is mostly near-equal mixtures (where
    'equal parts' is nearly right) gets a tighter scale than a design that spans the
    corners — which is correct: informativeness is relative to the design."""
    T = np.asarray(trues, float)
    if not len(T):
        return 25.0
    flat = np.full(T.shape[1], 100.0 / T.shape[1])
    return float(np.mean([comp_error(t, flat) for t in T]))


def group_by_condition(rows, concs, fmt="{}"):
    """Collapse per-MAP rows to one row per condition, averaging repeats.

    Held-out scoring reports a row per map, so a mixture prepared three times appears
    three times. A grid cell holds one condition, and averaging the repeats is what the
    reported per-condition error already means — drawing whichever repeat happened to
    come last would silently pick one preparation out of three."""
    order, bucket = [], {}
    for r, c in zip(rows, concs):
        k = tuple(round(float(v), 6) for v in c)
        if k not in bucket:
            bucket[k] = []; order.append(k)
        bucket[k].append((np.asarray(r[1], float), np.asarray(r[2], float)))
    out_rows, out_concs = [], []
    for k in order:
        T = np.mean([b[0] for b in bucket[k]], axis=0)
        P = np.mean([b[1] for b in bucket[k]], axis=0)
        P = P / (P.sum() or 1.0) * T.sum()
        out_rows.append((fmt.format("/".join(_fmt(v) for v in k)), T, P))
        out_concs.append(k)
    return out_rows, out_concs


def _err_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "err", ["#2f9e5e", "#a9c94b", "#e8c445", "#e08b3c", RED])


def err_color(err, floor, null):
    """Text colour for one error value, on the same scale as the heat-map."""
    if err <= floor:
        return MUTE
    return AMBER if err < null else RED


def _levels(concs, i, min_frac=0.10):
    """Levels of substance ``i`` that the design actually spans.

    A level held by one or two conditions is not an axis of the grid — it is an extra
    mixture that happens to sit outside it. Keeping it opened an empty column in every
    block and quietly folded that stray condition into a block mean, so levels carried
    by fewer than ``min_frac`` of the conditions are dropped from the layout (the
    conditions themselves are reported as excluded, never silently binned)."""
    from collections import Counter
    cnt = Counter(round(float(c[i]), 6) for c in concs)
    keep = max(2, int(round(min_frac * len(concs))))
    return sorted(v for v, n in cnt.items() if n >= keep)


def _grid_split(concs):
    """(levels per substance, index mask of on-grid conditions) or (None, None)."""
    if len(concs) < 4 or len(concs[0]) != 3:
        return None, None
    L = [_levels(concs, i) for i in range(3)]
    if any(not lv or len(lv) > 6 for lv in L):
        return None, None
    on = [all(round(float(c[i]), 6) in L[i] for i in range(3)) for c in concs]
    if sum(on) < 4:
        return None, None
    return L, on


def _fmt(v):
    return f"{v:g}"


# ----------------------------------------------------------------- pies
def composition_pies(rows, subs, colors, fig=None, ncol=9, title=None):
    """One predicted pie per condition, titled with the true composition.

    ``rows`` = [(label, true_pct, pred_pct)]. Good for spotting a compound that
    collapses; poor for judging agreement — use grid_truepred for that."""
    rows = list(rows)
    n = len(rows)
    ncol = max(1, min(ncol, n))
    nrow = int(np.ceil(n / ncol))
    fig = fig or Figure(figsize=(1.62 * ncol, 1.92 * nrow))
    fig.patch.set_alpha(0.0)
    axes = fig.subplots(nrow, ncol, squeeze=False)
    for ax in np.ravel(axes):
        ax.axis("off")
    for idx, (name, t, p) in enumerate(rows):
        ax = np.ravel(axes)[idx]
        t = np.asarray(t, float); p = np.asarray(p, float)
        keep = [i for i in range(len(subs)) if p[i] >= 1.0] or [int(np.argmax(p))]
        ax.pie([p[i] for i in keep], labels=[subs[i] for i in keep],
               colors=[colors[i] for i in keep], autopct="%1.0f%%",
               textprops={"fontsize": 6.0, "color": INK})
        ax.set_aspect("equal")
        ax.set_title(f"{name}\ntrue {'/'.join(f'{v:.0f}' for v in t)}  ·  "
                     f"err {comp_error(t, p):.0f}%", fontsize=6.6, pad=1.5)
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0.005, 1, 0.965])
    else:
        fig.tight_layout()
    return fig


# ----------------------------------------------------------------- true vs predicted grid
def _dominant(t, margin=2.0):
    """Index of the dominant compound, or None when the top two are within ``margin``
    percentage points — 33/33/33 has no dominant compound to get wrong, and marking it
    as an error measures rounding rather than the model."""
    s = np.sort(np.asarray(t, float))[::-1]
    return None if s[0] - s[1] < margin else int(np.argmax(t))


def _cell(ax, t, p, colors, floor, null, first=False):
    ax.set_xlim(-2, 102); ax.set_ylim(0, 1.42)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for y, v in ((0.72, t), (0.24, p)):
        x = 0.0
        for i in range(len(t)):
            ax.add_patch(Rectangle((x, y), v[i], 0.30, facecolor=colors[i], lw=0, zorder=2))
            x += v[i]
    # drop lines from the TRUE boundaries: where the lower bar meets them, it is right
    b = 0.0
    for i in range(len(t) - 1):
        b += t[i]
        ax.plot([b, b], [0.18, 1.08], ls=(0, (2.2, 1.8)), lw=1.0, color=INK, zorder=4)
    e = comp_error(t, p)
    ax.text(0, 1.20, f"{e:.0f}%", ha="left", va="bottom", fontsize=7.6,
            color=err_color(e, floor, null), fontweight="bold")
    dt = _dominant(t)
    if dt is None:
        ax.text(100, 1.20, "tie", ha="right", va="bottom", fontsize=6.4, color=MUTE)
    elif int(np.argmax(p)) != dt:
        ax.text(100, 1.20, "dominant X", ha="right", va="bottom", fontsize=6.4,
                color=RED, fontweight="bold")
    if first:
        ax.text(-3.5, 0.87, "true", ha="right", va="center", fontsize=6.2, color=MUTE)
        ax.text(-3.5, 0.39, "pred", ha="right", va="center", fontsize=6.2, color=MUTE)


def grid_truepred(rows, concs, subs, colors, fig=None, floor=DEFAULT_FLOOR, title=None):
    """The design grid with true (top) and predicted (bottom) bars in every cell.

    ``concs`` = [(c0, c1, c2)] in the same order as ``rows``. With a factorial design
    the layout is blocks of subs[0] x (subs[1] rows x subs[2] columns), so where the
    model struggles is read off the position. Anything else falls back to a wrapped
    flat grid in concentration order."""
    rows = list(rows); concs = [tuple(float(v) for v in c) for c in concs]
    trues = [np.asarray(r[1], float) for r in rows]
    null = null_error(trues)
    by_c = {c: (np.asarray(r[1], float), np.asarray(r[2], float))
            for c, r in zip(concs, rows)}

    L, on = _grid_split(concs)
    if L is None:
        n = len(rows); ncol = min(8, n); nrow = int(np.ceil(n / ncol))
        fig = fig or Figure(figsize=(2.0 * ncol, 1.30 * nrow))
        fig.patch.set_alpha(0.0)
        axes = fig.subplots(nrow, ncol, squeeze=False)
        for ax in np.ravel(axes):
            ax.axis("off")
        for i, (name, t, p) in enumerate(sorted(rows, key=lambda r: str(r[0]))):
            ax = np.ravel(axes)[i]; ax.set_axis_on()
            _cell(ax, np.asarray(t, float), np.asarray(p, float), colors, floor, null,
                  first=(i == 0))
            ax.set_title(str(name), fontsize=7, pad=3)
        _legend(fig, subs, colors, floor, null, rows, trues)
        fig.tight_layout(rect=[0, 0.075, 1, 0.94 if title else 1.0])
        if title:
            fig.suptitle(title, fontsize=11.5, fontweight="bold")
        return fig

    off = [r for r, ok in zip(rows, on) if not ok]
    rows = [r for r, ok in zip(rows, on) if ok]
    trues = [np.asarray(r[1], float) for r in rows]
    nb = len(L[0])
    bc = int(np.ceil(nb / 2)) if nb > 2 else nb
    br = int(np.ceil(nb / bc))
    fig = fig or Figure(figsize=(3.5 * len(L[2]) * bc / 2.0, 2.95 * len(L[1]) * br / 2.0))
    fig.patch.set_alpha(0.0)
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    outer = GridSpec(br, bc, figure=fig, hspace=0.24, wspace=0.14,
                     left=0.065, right=0.985, top=0.878, bottom=0.098)
    for bi, b in enumerate(L[0]):
        inner = GridSpecFromSubplotSpec(len(L[1]), len(L[2]),
                                        subplot_spec=outer[bi // bc, bi % bc],
                                        hspace=0.34, wspace=0.16)
        be = []
        for r, rv in enumerate(L[1]):
            for c, cv in enumerate(L[2]):
                ax = fig.add_subplot(inner[r, c])
                k = (b, rv, cv)
                if k not in by_c:
                    ax.axis("off"); continue
                t, p = by_c[k]
                be.append(comp_error(t, p))
                _cell(ax, t, p, colors, floor, null, first=(bi == 0 and r == 0 and c == 0))
                if r == 0:
                    ax.set_title(f"{subs[2]} {_fmt(cv)}" if c == 0 else _fmt(cv),
                                 fontsize=8, color=colors[2], pad=4)
                if c == 0:
                    ax.set_ylabel(f"{subs[1]} {_fmt(rv)}" if r == 0 else _fmt(rv),
                                  fontsize=8, color=colors[1], rotation=0,
                                  ha="right", va="center", labelpad=22)
        bb = fig.add_subplot(outer[bi // bc, bi % bc], frameon=False)
        bb.set_xticks([]); bb.set_yticks([]); bb.patch.set_alpha(0)
        n_ok = sum(1 for e in be if e <= floor)
        bb.set_title(f"{subs[0]} {_fmt(b)}        mean {np.mean(be) if be else float('nan'):.1f}%"
                     f"   ·   {n_ok}/{len(be)} at the repeat floor",
                     fontsize=10.5, color=colors[0], pad=30)
    fig.suptitle(title or ("Composition, condition by condition — top bar = true, "
                           "bottom bar = predicted\nthe dashed lines drop from the TRUE "
                           "boundaries: where the lower bar meets them, the model got it right"),
                 fontsize=11.5, fontweight="bold", y=0.975)
    _legend(fig, subs, colors, floor, null, rows, trues, off=off)
    return fig


def _legend(fig, subs, colors, floor, null, rows, trues, off=()):
    preds = [np.asarray(r[2], float) for r in rows]
    errs = np.array([comp_error(t, p) for t, p in zip(trues, preds)])
    judged = [(_dominant(t), int(np.argmax(p))) for t, p in zip(trues, preds)]
    nj = [x for x in judged if x[0] is not None]
    n_dom = sum(1 for d, a in nj if d == a)
    fig.legend(handles=[Patch(facecolor=colors[i], label=subs[i]) for i in range(len(subs))],
               loc="lower center", ncol=len(subs), fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, 0.052))
    fig.text(0.5, 0.036,
             f"error %:  grey ≤ {floor:.1f} (repeat floor — as right as the measurement shows)"
             f"  ·  red ≥ {null:.1f} (no better than answering equal parts)        "
             f"'dominant X' = wrong largest compound  ·  'tie' = true top two within 2%p",
             ha="center", fontsize=8.5, color=MUTE)
    fig.text(0.5, 0.012,
             f"{len(rows)} conditions        "
             f"dominant compound right {n_dom}/{len(nj)} (ties excluded)        "
             f"at the repeat floor {int((errs <= floor).sum())}/{len(errs)}        "
             f"better than guessing {int((errs < null).sum())}/{len(errs)}        "
             f"mean {errs.mean():.1f}% · median {np.median(errs):.1f}%"
             + (f"        off-grid, not drawn: {', '.join(str(r[0]) for r in off)}"
                if len(off) else ""),
             ha="center", fontsize=9, color=MUTE)


# ----------------------------------------------------------------- error heat-map
def grid_error(rows, concs, subs, fig=None, floor=DEFAULT_FLOOR):
    """The design grid as an error heat-map — where the hard conditions sit.

    Scale runs from the repeat floor to the uninformative baseline (see module note),
    not 0-30%: green means 'at the measurement limit', red means 'no better than
    answering equal parts'."""
    rows = list(rows); concs = [tuple(float(v) for v in c) for c in concs]
    trues = [np.asarray(r[1], float) for r in rows]
    null = null_error(trues)
    err = {c: comp_error(r[1], r[2]) for c, r in zip(concs, rows)}
    L, on = _grid_split(concs)
    if L is None:
        return None
    n_off = sum(1 for ok in on if not ok)
    fig = fig or Figure(figsize=(3.9 * len(L[0]), 4.4))
    fig.patch.set_alpha(0.0)
    axes = fig.subplots(1, len(L[0]), squeeze=False)[0]
    cmap = _err_cmap()
    im = None
    for bi, b in enumerate(L[0]):
        M = np.full((len(L[1]), len(L[2])), np.nan)
        for r, rv in enumerate(L[1]):
            for c, cv in enumerate(L[2]):
                if (b, rv, cv) in err:
                    M[r, c] = err[(b, rv, cv)]
        ax = axes[bi]
        im = ax.imshow(M, cmap=cmap, vmin=floor, vmax=null, origin="upper")
        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                if np.isfinite(M[r, c]):
                    ax.text(c, r, f"{M[r, c]:.0f}", ha="center", va="center", fontsize=10,
                            color="white" if M[r, c] > (floor + null) / 2 else INK,
                            fontweight="bold")
        ax.set_xticks(range(len(L[2]))); ax.set_xticklabels([_fmt(v) for v in L[2]], fontsize=9.5)
        ax.set_yticks(range(len(L[1]))); ax.set_yticklabels([_fmt(v) for v in L[1]], fontsize=9.5)
        ax.set_xlabel(subs[2])
        if bi == 0:
            ax.set_ylabel(subs[1])
        ax.set_title(f"{subs[0]} {_fmt(b)}\nmean {np.nanmean(M):.1f}%", fontsize=10)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, extend="both")
    cb.set_label("composition error (%)")
    cb.outline.set_visible(False)
    cb.ax.text(1.0, floor, f"  {floor:.1f} repeat floor", transform=cb.ax.transData,
               ha="left", va="center", fontsize=8, color=MUTE)
    cb.ax.text(1.0, null, f"  {null:.1f} no better\n  than guessing",
               transform=cb.ax.transData, ha="left", va="center", fontsize=8, color=MUTE)
    fig.text(0.5, 0.005, "green = at the repeat floor (as right as the measurement shows)"
                         "   ·   red = no better than answering equal parts"
                         + (f"   ·   {n_off} off-grid condition(s) not drawn" if n_off else ""),
             ha="center", fontsize=8.5, color=MUTE)
    return fig


# ----------------------------------------------------------------- tables (redraw elsewhere)
# Every figure here has a companion table, because these figures get rebuilt in Origin
# and a PNG is not data. Long form for the per-condition and per-pixel tables (one
# observation per line, the shape a plotting package wants); the error grid also comes
# out as one matrix per block, since a heat-map needs a matrix rather than a column.
def condition_table(rows, concs, subs, floor=DEFAULT_FLOOR):
    """(header, rows) — one line per condition, everything the grid/pie figures draw."""
    trues = [np.asarray(r[1], float) for r in rows]
    null = null_error(trues)
    head = (["condition"] + [f"{s}_uM" for s in subs]
            + [f"true_{s}_pct" for s in subs] + [f"pred_{s}_pct" for s in subs]
            + ["composition_error_pct", "at_repeat_floor", "better_than_guessing",
               "dominant_true", "dominant_pred", "dominant_correct"])
    out = []
    for (name, t, p), c in zip(rows, concs):
        t = np.asarray(t, float); p = np.asarray(p, float)
        e = comp_error(t, p)
        dt = _dominant(t)
        out.append([str(name)] + [f"{v:g}" for v in c]
                   + [f"{v:.3f}" for v in t] + [f"{v:.3f}" for v in p]
                   + [f"{e:.3f}", str(int(e <= floor)), str(int(e < null)),
                      "tie" if dt is None else subs[dt], subs[int(np.argmax(p))],
                      "" if dt is None else str(int(dt == int(np.argmax(p))))])
    return head, out


def error_matrix_tables(rows, concs, subs):
    """[(name, header, rows)] — the error heat-map as one matrix per block, rows =
    subs[1] levels, columns = subs[2] levels. Empty list when the design is not a grid."""
    concs = [tuple(float(v) for v in c) for c in concs]
    L, on = _grid_split(concs)
    if L is None:
        return []
    err = {c: comp_error(r[1], r[2]) for c, r in zip(concs, rows)}
    tabs = []
    for b in L[0]:
        head = [f"{subs[1]}_uM \\ {subs[2]}_uM"] + [_fmt(v) for v in L[2]]
        body = []
        for rv in L[1]:
            line = [_fmt(rv)]
            for cv in L[2]:
                v = err.get((b, rv, cv))
                line.append("" if v is None else f"{v:.3f}")
            body.append(line)
        tabs.append((f"{subs[0]}_{_fmt(b)}", head, body))
    return tabs


def pixel_table(entries, subs):
    """(header, rows) — one line per pixel per map, the per-pixel composition maps."""
    head = ["map", "x", "y", "hit"] + [f"{s}_pct" for s in subs]
    out = []
    for name, coords, ratio, hit in entries:
        coords = np.asarray(coords, float); ratio = np.asarray(ratio, float)
        hit = np.asarray(hit, bool)
        for i in range(len(coords)):
            out.append([str(name), f"{coords[i, 0]:g}", f"{coords[i, 1]:g}", str(int(hit[i]))]
                       + [f"{100.0 * ratio[i, k]:.3f}" for k in range(ratio.shape[1])])
    return head, out


# ----------------------------------------------------------------- per-pixel maps
def pixel_maps(entries, subs, colors, fig=None, ncol=9, title=None):
    """One per-pixel composition map per condition.

    ``entries`` = [(label, coords (n,2), ratio (n,k), hit (n,))] — the same arrays the
    Real-data tab draws for a single map, tiled so a whole experiment reads at once."""
    entries = list(entries)
    n = len(entries)
    ncol = max(1, min(ncol, n))
    nrow = int(np.ceil(n / ncol))
    fig = fig or Figure(figsize=(1.58 * ncol, 1.80 * nrow))
    fig.patch.set_alpha(0.0)
    axes = fig.subplots(nrow, ncol, squeeze=False)
    for ax in np.ravel(axes):
        ax.axis("off")
    for idx, (name, coords, ratio, hit) in enumerate(entries):
        ax = np.ravel(axes)[idx]; ax.set_axis_on()
        coords = np.asarray(coords, float); ratio = np.asarray(ratio, float)
        hit = np.asarray(hit, bool)
        x, y = coords[:, 0], coords[:, 1]
        ux = np.unique(x)
        rad = (np.median(np.diff(ux)) * 0.46) if len(ux) > 1 else 0.46
        if (~hit).any():
            ax.scatter(x[~hit], y[~hit], c=BG_GREY, marker="s", s=9, edgecolors="none")
        wedges, wcols = [], []
        for i in np.where(hit)[0]:
            a0 = 90.0
            for k in range(ratio.shape[1]):
                f = ratio[i, k]
                if f <= 0.002:
                    continue
                a1 = a0 - f * 360.0
                wedges.append(Wedge((x[i], y[i]), rad, a1, a0)); wcols.append(colors[k])
                a0 = a1
        if wedges:
            ax.add_collection(PatchCollection(wedges, facecolors=wcols, edgecolors="none"))
        ax.set_xlim(x.min() - 1, x.max() + 1)
        ax.set_ylim(y.max() + 1, y.min() - 1)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{name}  ·  hit {100 * hit.mean():.0f}%", fontsize=6.6, pad=2)
    fig.legend(handles=[Patch(facecolor=colors[i], label=subs[i]) for i in range(len(subs))]
                       + [Patch(facecolor=BG_GREY, label="background")],
               loc="lower center", ncol=len(subs) + 1, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.002))
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0.028, 1, 0.965 if title else 1.0])
    return fig
