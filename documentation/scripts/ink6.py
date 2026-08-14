"""Ink-coefficient readout for the 6-map composition check (RUN_ink-check_6maps.md §6).

Templates: all classes incl. INK, 300-1800 cm-1, ALS, L2.
Map: mean spectrum -> ALS -> clip -> NNLS on the templates -> take the INK coefficient.
"""
import os as _os, sys as _sys                 # paths 부트스트랩 — 기계마다 마운트가 다르다
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import sys, math, csv
import numpy as np
from scipy.optimize import nnls

REPO = paths.REPO
DB = paths.PURE
sys.path.insert(0, REPO)

import unmix
from real_data import load_map
from sers_mixture import als_baseline

LO, HI = 300.0, 1800.0


def despike(y, thr=7.0, win=5):
    """Median-filter despike: replace points whose residual exceeds thr*MAD."""
    y = np.asarray(y, float)
    k = win // 2
    pad = np.pad(y, k, mode="edge")
    med = np.array([np.median(pad[i:i + win]) for i in range(len(y))])
    r = y - med
    mad = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
    out = y.copy()
    bad = r > thr * mad
    out[bad] = med[bad]
    return out


def templates():
    names, wn, means = unmix._templates(DB, True, None)
    wn = np.asarray(wn, float)
    m = (wn >= LO) & (wn <= HI)
    P = unmix._l2(unmix._baseline_removed(means[:, m], True))
    return names, wn[m], P


def ink_of(path, names, wn_t, P, mode="mean-then-als", spike=False):
    wn, cube, mean, _c = load_map(path)
    wn = np.asarray(wn, float)
    m = (wn >= LO) & (wn <= HI)
    wn_m, cube = wn[m], np.asarray(cube, float)[:, m]
    if spike:
        cube = np.stack([despike(y) for y in cube])
    if mode == "mean-then-als":
        y = cube.mean(axis=0)
        y = np.clip(y - als_baseline(y), 0.0, None)
    else:                                   # als each pixel, then average
        y = np.stack([np.clip(v - als_baseline(v), 0.0, None) for v in cube]).mean(axis=0)
    if len(wn_m) != len(wn_t) or not np.allclose(wn_m, wn_t):
        y = np.interp(wn_t, wn_m, y)
    a, _ = nnls(P.T, y)
    return a, y


def report(rows, names, wn_t, P, **kw):
    out = []
    for label, path in rows:
        a, y = ink_of(path, names, wn_t, P, **kw)
        d = dict(zip(names, a))
        out.append((label, d))
    return out


if __name__ == "__main__":
    names, wn_t, P = templates()
    print("templates:", names, "| n_feat", len(wn_t))
    mode = sys.argv[1] if len(sys.argv) > 1 else "mean-then-als"
    spike = len(sys.argv) > 2 and sys.argv[2] == "spike"
    print(f"mode={mode} despike={spike}\n")

    base = "/Volumes/Seungki/260805"
    prior = []
    for s in (1, 2):
        for tag, c in (("1mM", 333.0), ("03mM", 111.0), ("01mM", 37.0), ("003mM", 12.3)):
            for suf in ("", "_corrected"):
                p = f"{base}/tert-{s}/tert_{tag}_{s}{suf}.csv"
                prior.append((f"set{s} {c:>5.1f}uM {suf or 'raw':>10}", p, c, s, suf))

    print("=== 2026-08-05 dilution series (reproduce the published calibration) ===")
    fits = {}
    for suf in ("", "_corrected"):
        print(f"\n-- source: {suf or 'raw'} --")
        xs, ys = [], []
        for label, p, c, s, sf in prior:
            if sf != suf:
                continue
            a, _ = ink_of(p, names, wn_t, P, mode=mode, spike=spike)
            ink = a[names.index("INK")]
            print(f"  {label}: INK={ink:12.1f}   " +
                  "  ".join(f"{n}={v:9.1f}" for n, v in zip(names, a) if n != "INK"))
            if ink > 0:
                xs.append(math.log10(c)); ys.append(math.log10(ink))
        if len(xs) >= 3:
            sl, ic = np.polyfit(xs, ys, 1)
            r2 = 1 - np.sum((np.array(ys) - (sl * np.array(xs) + ic)) ** 2) / \
                 np.sum((np.array(ys) - np.mean(ys)) ** 2)
            print(f"  FIT log10(INK) = {sl:.3f}*log10(C) + {ic:.3f}   R2={r2:.3f}  (n={len(xs)})")
            fits[suf] = (sl, ic)

    print("\n=== 6-map composition check (tert-new) ===")
    # actual compositions (des.txt is T:B:D; listed here as DQ/TBZ/THI)
    maps = [(1, 6, 6, 24), (2, 12, 12, 12), (3, 24, 6, 6),
            (4, 6, 6, 6), (5, 24, 24, 24), (6, 12, 12, 48)]
    K = {"DQ": 1.0, "TBZ": 11.6, "THI": 16.9}
    res = {}
    for n, dq, tbz, thi in maps:
        a, _ = ink_of(f"{base}/tert-new/{n}-1.csv", names, wn_t, P, mode=mode, spike=spike)
        d = dict(zip(names, a))
        res[n] = (d, dq, tbz, thi)
        tot = dq + tbz + thi
        print(f"  #{n} {dq:2d}/{tbz:2d}/{thi:2d} tot={tot:3d}  INK={d['INK']:11.1f}  " +
              "  ".join(f"{k}={d[k]:9.1f}" for k in names if k != "INK"))

    print("\n--- (1) composition at total 36 uM:  #3 (DQ-heavy) : #2 (even) : #1 (THI-heavy) ---")
    i3, i2, i1 = res[3][0]["INK"], res[2][0]["INK"], res[1][0]["INK"]
    print(f"  #3={i3:.1f}  #2={i2:.1f}  #1={i1:.1f}")
    if min(i1, i2, i3) > 0:
        print(f"  spread max/min = {max(i1,i2,i3)/min(i1,i2,i3):.2f}x   "
              f"#3/#1 = {i3/i1:.2f}x  (A predicts 1.0x, B predicts 6.1x with #3 largest)")

    print("\n--- (2) slope, 1:1:1 series #4 (18) -> #2 (36) -> #5 (72) ---")
    xs = [math.log10(t / 3) for t in (18, 36, 72)]
    ys = [res[n][0]["INK"] for n in (4, 2, 5)]
    print("  per-component C =", [f"{10**x:.0f}" for x in xs], " INK =", [f"{v:.1f}" for v in ys])
    if min(ys) > 0:
        sl, ic = np.polyfit(xs, np.log10(ys), 1)
        print(f"  slope={sl:.2f}  intercept={ic:.2f}   (2026-08-05: -2.01 / 6.32)")

    print("\n--- (3) same composition 1:1:4, two totals: #1 (36) -> #6 (72) ---")
    i6 = res[6][0]["INK"]
    if i6 > 0 and i1 > 0:
        sl = (math.log10(i6) - math.log10(i1)) / (math.log10(72 / 3) - math.log10(36 / 3))
        print(f"  #1={i1:.1f}  #6={i6:.1f}  ratio={i1/i6:.2f}x  slope={sl:.2f}")
