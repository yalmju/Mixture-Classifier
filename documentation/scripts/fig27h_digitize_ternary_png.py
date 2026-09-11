# -*- coding: utf-8 -*-
"""27h — 삼각도 PNG(사용자 Origin 그림)를 데이터로 읽어 배경 필드를 만든다.

입력: 삼각도 PNG (검은 외곽 삼각형, 예측점은 초록/노랑/주황/빨강, 진실은 흰 원).
처리: 1) 검은 삼각형 꼭짓점 검출(THI 위, TBZ 좌하, DQ 우하)
      2) 색 점 검출(HSV 군집) → 픽셀 좌표 → 무게중심 좌표(DQ, TBZ, THI)
      3) 점의 색 = 밴드(0 초록 … 3 빨강); 배경 = 국소 정답률 또는 밴드 점수 평균
출력: <png>_bg.png (삼각형 영역만, 투명 배경 — Origin 레이어 아래에 깔기),
      <png>_points.csv (x_px, y_px, DQ, TBZ, THI, band),
      <png>_field_XYZZ.csv (Origin ternary contour 용)

실행:  python -u fig27h_digitize_ternary_png.py figure.png [--sigma 20] [--palette rb|po]
       [--field fraction|score] [--correct-band 1.25|1.5|2] [--crop x0 y0 x1 y1]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
from scipy import ndimage  # noqa: E402

RB = LinearSegmentedColormap.from_list("rb", ["#c8322b", "#f6f1ec", "#2b64b5"])
PO = LinearSegmentedColormap.from_list("po", ["#5b4a8a", "#f7f5ed", "#e8a13a"])
# 밴드 기준색 (RGB 0-255): 초록 / 노랑 / 주황 / 빨강 — Origin 그림 팔레트
REF = {0: (31, 157, 85), 1: (232, 195, 58), 2: (238, 127, 45), 3: (214, 39, 40)}


def rgb2hsv(a):
    a = a.astype(float) / 255.0
    mx, mn = a.max(-1), a.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    m = d > 1e-6
    h[m & (mx == r)] = ((g - b) / np.where(d == 0, 1, d))[m & (mx == r)] % 6
    h[m & (mx == g)] = ((b - r) / np.where(d == 0, 1, d))[m & (mx == g)] + 2
    h[m & (mx == b)] = ((r - g) / np.where(d == 0, 1, d))[m & (mx == b)] + 4
    return h * 60, np.where(mx > 0, d / np.where(mx == 0, 1, mx), 0), mx


def find_triangle(img):
    """검은(어두운 무채색) 선의 픽셀에서 꼭짓점 3개."""
    h, s, v = rgb2hsv(img)
    dark = (v < 0.35) & (s < 0.35)
    # 가장 큰 어두운 연결 성분 = 삼각형 외곽선 (범례 마커·글자는 작다)
    lab, n = ndimage.label(ndimage.binary_dilation(dark, iterations=2))
    if n == 0:
        raise SystemExit("검은 삼각형 선을 찾지 못했습니다 (--crop 으로 삼각형만 잘라 주세요)")
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    objs = ndimage.find_objects(lab)
    areas = [(o[0].stop - o[0].start) * (o[1].stop - o[1].start) for o in objs]
    big = int(np.argmax(areas)) + 1
    ys, xs = np.nonzero((lab == big) & dark)
    top = (xs[ys.argmin()], ys.min())
    ybase = ys.max()
    base = ys > ybase - 0.02 * (ybase - top[1])
    left = (xs[base].min(), ybase)
    right = (xs[base].max(), ybase)
    return np.array(top, float), np.array(left, float), np.array(right, float)


def to_bary(pts, top, left, right):
    """픽셀 → (DQ, TBZ, THI) %: THI = top, TBZ = left, DQ = right."""
    A = np.array([[right[0] - left[0], top[0] - left[0]],
                  [right[1] - left[1], top[1] - left[1]]], float)
    out = []
    for p in pts:
        u, w = np.linalg.solve(A, np.asarray(p, float) - left)   # u→DQ, w→THI
        out.append([u, 1 - u - w, w])
    return np.clip(np.array(out), 0, 1) * 100


def find_dots(img, top, left, right, min_px=12):
    h, s, v = rgb2hsv(img)
    colored = (s > 0.45) & (v > 0.5)
    lab, n = ndimage.label(colored)
    dots = []
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < min_px:
            continue
        cy, cx = ndimage.center_of_mass(m)
        rgb = img[m].reshape(-1, 3).astype(float).mean(0)
        band = min(REF, key=lambda k: np.sum((np.array(REF[k]) - rgb) ** 2))
        # 삼각형 밖(범례 등)은 제외
        d, t, th = to_bary([(cx, cy)], top, left, right)[0]
        inside = (d + t + th > 99.0) and min(d, t, th) > -0.5
        u, w = np.linalg.solve(np.array([[right[0] - left[0], top[0] - left[0]],
                                         [right[1] - left[1], top[1] - left[1]]]),
                               np.array([cx, cy]) - left)
        if u < -0.02 or w < -0.02 or u + w > 1.02:
            continue
        dots.append((cx, cy, d, t, th, band))
    return dots


def field(bary, C, sigma):
    grid = [(d, t, 100 - d - t) for d in np.arange(0, 101, 1.0)
            for t in np.arange(0, 101 - d, 1.0)]
    G = np.array(grid, float)
    d2 = ((G[:, None, :] - bary[None, :, :]) ** 2).sum(-1)
    w = np.exp(-0.5 * d2 / sigma ** 2)
    dens = w.sum(1)
    return G, (w * C[None, :]).sum(1) / np.maximum(dens, 1e-12), dens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("png")
    ap.add_argument("--sigma", type=float, default=20.0)
    ap.add_argument("--palette", choices=["rb", "po"], default="rb")
    ap.add_argument("--field", choices=["fraction", "score"], default="fraction",
                    help="fraction: local share of points within --correct-band; "
                         "score: local mean of band score (green 1 … red 0)")
    ap.add_argument("--correct-band", type=float, default=2.0, choices=[1.25, 1.5, 2.0])
    ap.add_argument("--vrange", nargs=2, type=float, default=[0, 1])
    ap.add_argument("--crop", nargs=4, type=int, default=None, help="x0 y0 x1 y1 (px)")
    ap.add_argument("--alpha", type=float, default=0.85)
    a = ap.parse_args()
    img = np.asarray(Image.open(a.png).convert("RGB"))
    if a.crop:
        x0, y0, x1, y1 = a.crop
        img = img[y0:y1, x0:x1]
    top, left, right = find_triangle(img)
    dots = find_dots(img, top, left, right)
    if not dots:
        raise SystemExit("색 점을 찾지 못했습니다")
    bary = np.array([[d, t, th] for _, _, d, t, th, _ in dots])
    bands = np.array([b for *_, b in dots])
    cb_idx = {1.25: 0, 1.5: 1, 2.0: 2}[a.correct_band]
    C = ((bands <= cb_idx).astype(float) if a.field == "fraction"
         else np.array([1.0, 0.67, 0.33, 0.0])[bands])
    G, frac, dens = field(bary, C, a.sigma)
    stem = os.path.splitext(a.png)[0]
    with open(stem + "_points.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["x_px", "y_px", "DQ_pct", "TBZ_pct", "THI_pct", "band"])
        for cx, cy, d, t, th, b in dots:
            w.writerow([f"{cx:.1f}", f"{cy:.1f}", f"{d:.1f}", f"{t:.1f}", f"{th:.1f}",
                        ["within 1.25-fold", "within 1.5-fold", "within 2-fold", "Beyond 2-fold"][b]])
    with open(stem + "_field_XYZZ.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["DQ_pct_X", "TBZ_pct_Y", "THI_pct_Z", "Z2", "density"])
        for g, fr, dn in zip(G, frac, dens / dens.max()):
            w.writerow([f"{g[0]:.0f}", f"{g[1]:.0f}", f"{g[2]:.0f}", f"{fr:.4f}", f"{dn:.4f}"])
    # 배경 PNG: 원본과 같은 픽셀 크기, 삼각형 안만 칠하고 나머지 투명
    H, W = img.shape[:2]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_axis_off()
    gx = left[0] + (right[0] - left[0]) * G[:, 0] / 100 + (top[0] - left[0]) * G[:, 2] / 100
    gy = left[1] + (right[1] - left[1]) * G[:, 0] / 100 + (top[1] - left[1]) * G[:, 2] / 100
    tri = mtri.Triangulation(gx, gy)
    ax.tripcolor(tri, frac, cmap=RB if a.palette == "rb" else PO, vmin=a.vrange[0],
                 vmax=a.vrange[1], shading="gouraud", alpha=a.alpha, rasterized=True)
    fig.savefig(stem + "_bg.png", dpi=100, transparent=True)
    n = [int((bands == k).sum()) for k in range(4)]
    print(f"triangle top {top} left {left} right {right}")
    print(f"dots: green {n[0]} yellow {n[1]} orange {n[2]} red {n[3]}  → field mean "
          f"{frac[dens > 0.05 * dens.max()].mean():.2f}")
    print("saved", stem + "_bg.png", stem + "_points.csv", stem + "_field_XYZZ.csv")


if __name__ == "__main__":
    main()
