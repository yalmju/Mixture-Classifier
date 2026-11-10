# -*- coding: utf-8 -*-
"""LEAF 배경 참조 생성 — 잎맵의 '잎 바깥(아무것도 없는)' 픽셀을 Pure/LEAF-*.csv로.

260902_leaf Sample1–3에서, 사용자가 확인한 대로 @~1000 cm⁻¹가 밝은 영역이
잎 바깥/맨 기판이다(잎맥도 보이지만 어두운 쪽). 픽셀 선정:
  1) 1000±8 cm⁻¹ 밴드 상위 클러스터(Otsu 임계) = 밝은 영역
  2) 그중 분석물 마커(1270/1368/1572±8)가 상위 10%인 픽셀은 제외
     (혹시 모를 분석물 오염 방지)
선택 픽셀 행을 원본 형식 그대로 LEAF-1..3.csv로 저장 → BLANK_ALIASES에
"leaf"가 있으므로 NNLS 게이트가 INK처럼 배경 처리한다. 재학습 불필요.

실행:  python -u make_leaf_bg_refs.py
"""
import io
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")
SRC = r"S:\Google Drive\내 드라이브\ACF_PEST_DB\260902_leaf"
PURE = r"S:\Google Drive\내 드라이브\ACF_PEST_DB\Pure"
ANALYTE_BANDS = (1270.0, 1368.0, 1572.0)


def otsu(v):
    v = np.asarray(v, float)
    hist, edges = np.histogram(v, bins=128)
    mids = (edges[:-1] + edges[1:]) / 2
    w = hist.astype(float)
    best, thr = -1.0, mids[len(mids) // 2]
    csum = np.cumsum(w)
    cmean = np.cumsum(w * mids)
    tot, totm = csum[-1], cmean[-1]
    for i in range(1, len(mids) - 1):
        w0, w1 = csum[i], tot - csum[i]
        if w0 <= 0 or w1 <= 0:
            continue
        m0, m1 = cmean[i] / w0, (totm - cmean[i]) / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best:
            best, thr = var, mids[i]
    return thr


def band_val(spec, wn, centre, hw=8.0):
    m = np.abs(wn - centre) <= hw
    return spec[:, m].max(axis=1)


for k in (1, 2, 3):
    path = os.path.join(SRC, f"Sample{k}_corrected.csv")
    lines = open(path, encoding="utf-8-sig").read().splitlines()
    head, axis_line, rows = lines[:2], lines[2], lines[3:]
    wn = np.array([float(x) for x in axis_line.split(",")[2:] if x.strip()])
    spec = np.array([[float(x) if x.strip() else 0.0
                      for x in ln.split(",")[2:2 + len(wn)]] for ln in rows])
    b1000 = band_val(spec, wn, 1000.0)
    thr = otsu(b1000)
    sel = b1000 > thr
    for c in ANALYTE_BANDS:
        bv = band_val(spec, wn, c)
        sel &= bv < np.quantile(bv, 0.90)
    idx = np.where(sel)[0]
    print(f"Sample{k}: off-leaf {idx.size}/{len(rows)} px "
          f"(thr@1000={thr:.0f})")
    out = os.path.join(PURE, f"LEAF-{k}.csv")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"X num,{idx.size}\nY num,1\n")
        f.write(axis_line + "\n")
        for j in idx:
            f.write(rows[j] + "\n")
    print("  wrote", out)
print("done — 게이트가 LEAF를 배경으로 쓰려면 앱 재시작 후 Unmix만 하면 됨")
