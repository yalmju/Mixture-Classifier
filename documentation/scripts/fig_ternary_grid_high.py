"""Ternary composition recovery on the hard subset — every binary condition plus
high-concentration ternary (total >= 99 uM) — NNLS vs the DEPLOYED MLP.

MLP side: held-out predictions carried inside mlp_composition_260826.dlm
(leave-one-condition-out + the independent 20-map test batch; BLK dropped and
renormalised). NNLS side: computed fresh on the same maps with the same
preprocessing and pixel sampling (untrained, so no split needed).

Two versions: (1) arrows+points over the interpolated accuracy surface,
(2) the surface alone.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")))
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

DATA = r"S:/Google Drive/내 드라이브/ACF_PEST_DB/Pure"
DLM = r"S:/Google Drive/내 드라이브/ACF_PEST_DB/260826_Model/mlp_composition_260826.dlm"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "results", "ternary_grid_high_cache.npz")

SUBS = ["DQ", "TBZ", "THI"]
V = {"THI": np.array([0.5, np.sqrt(3) / 2]),
     "TBZ": np.array([0.0, 0.0]),
     "DQ": np.array([1.0, 0.0])}


def bary(c):
    c = np.clip(np.asarray(c, float), 0, None)
    c = c / (c.sum() + 1e-12)
    return c[0] * V["DQ"] + c[1] * V["TBZ"] + c[2] * V["THI"]


def build_cache():
    from dl_model import load_model, _refs, _map_spectra, _composition_features
    from dl_quantify import surface_composition
    from dataset import load_mixture_list
    from real_data import load_map

    model = load_model(DLM)
    subs_all = list(model["subs"])
    cols = [subs_all.index(s) for s in SUBS]
    held = {}
    for ev in (model.get("loo_eval") or {}, model.get("test_eval") or {}):
        for pth, pred in zip(ev.get("paths", []), ev.get("pred", [])):
            held[os.path.normcase(os.path.normpath(pth))] = np.asarray(pred, float)

    items = load_mixture_list(DATA, filename_truth=True)
    subs_ref, wn, mask, P, lo, hi = _refs(DATA, False, (500, 2500))
    rows = []
    for it in items:
        path, ratio = it[0], it[1]
        conc = it[2] if len(it) > 2 else None
        key = os.path.normcase(os.path.normpath(path))
        if key not in held or conc is None:
            continue
        t = np.array([float(ratio.get(s, 0.0)) for s in SUBS], float)
        t = t / (t.sum() + 1e-12)
        tot_uM = sum(float(v) for v in conc.values()) * 1e6
        k = int((t > 0).sum())
        lv = sorted(set(round(float(v) * 1e6, 3) for v in conc.values()))
        grid = all(round(float(v) * 1e6, 3) in (3.0, 6.0, 12.0, 24.0)
                   for v in conc.values())
        if not (k == 3 and grid and tot_uM >= 36.0):
            continue
        mlp = np.clip(held[key][cols], 0, None)
        mlp = mlp / (mlp.sum() + 1e-12)
        _w, cube, _m, _c = load_map(path)
        specs = _map_spectra(cube, mask, 100, baseline_correct=True,
                             sampling="representative", sampling_seed=0,
                             map_id=path)
        feats = _composition_features(np.stack(specs), "legacy_l2")
        nn = surface_composition(feats, P).mean(0)
        nn = nn / (nn.sum() + 1e-12)
        rows.append((os.path.basename(path), t, nn, mlp, tot_uM, k))
        print(f"  {os.path.basename(path)}  nnls dev "
              f"{50*np.abs(nn-t).sum():.1f}  mlp dev {50*np.abs(mlp-t).sum():.1f}",
              flush=True)
    names = np.array([r[0] for r in rows], object)
    np.savez(CACHE, names=names,
             true=np.stack([r[1] for r in rows]),
             nnls=np.stack([r[2] for r in rows]),
             mlp=np.stack([r[3] for r in rows]),
             tot=np.array([r[4] for r in rows]),
             k=np.array([r[5] for r in rows]))
    return CACHE


if not os.path.exists(CACHE) or os.environ.get("REBUILD"):
    build_cache()
z = np.load(CACHE, allow_pickle=True)
TRUE, PRED = z["true"], {"nnls": z["nnls"], "mlp": z["mlp"]}
n_b = 0; n_h = int((z["k"] == 3).sum())

norm = Normalize(0.0, 1.0)
cmap = cm.RdYlGn
tri = np.array([V["TBZ"], V["DQ"], V["THI"], V["TBZ"]])


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
                ax.scatter(*pt, s=26, facecolors="white",
                           edgecolors="#6a7178", linewidths=0.8, zorder=4)
                ax.scatter(*pp, s=30, facecolors=[cmap(norm(a_))],
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
    note = (f"held-out (deployed model) · factorial grid conditions with "
            f"total ≥ 36 µM (n={n_h} of 64)")
    if points:
        note += " · open ○ = prepared, filled ● = predicted"
    fig.text(0.5, 0.015, note, fontsize=7.5, color=MUTE, ha="center")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "figures")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_ternary_grid_high{tag}.{ext}"),
                    dpi=600, bbox_inches="tight", pad_inches=0.02,
                    facecolor="white")
    plt.close(fig)


draw(points=True)
draw(points=False)
print(f"saved both versions: binary {n_b} + high-conc ternary {n_h}")
