# -*- coding: utf-8 -*-
"""µM 판독의 k-NN(라이브러리 조회) 대체안 — 외삽 구조적 불가 버전.

사용자 지시: "검량선(≤144)과 학습 맵 92개, 이것만으로 반환하라."
각 학습 혼합물 맵의 residual-피처 z벡터를 라이브러리로 만들고, 판독은
거리가중 k-NN으로 이웃 맵들의 실측 농도를 반환한다. LOO(자기 제외)로
현행 residual 헤드와 같은 92셋에서 채점한다.

실행:  python -u run_knn_um_readout.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from dataset import load_preprocess              # noqa: E402
from dl_model import (load_model, apply_model_pixels,                  # noqa: E402
                      _residual_context)
from unmix import unmix_map                       # noqa: E402

DLM = r"S:\Google Drive\내 드라이브\ACF_PEST_DB\260831_Model_FINAL\mlp_composition_260831_final.dlm"
PURE = r"S:\Google Drive\내 드라이브\ACF_PEST_DB\Pure"
OUT = os.path.join(ROOT, "documentation", "results", "knn_um_readout_20260901")
os.makedirs(OUT, exist_ok=True)

m = load_model(DLM)
u = m["uM"]
ab = np.asarray(u["loglinear_ab"], float)
cal_rng = np.asarray(u["cal_range_uM"], float)
bands = np.asarray(u["bands_cm"], float)
mu = np.asarray(u["mu"], float)
sd = np.where(np.asarray(u["sd"], float) > 0, np.asarray(u["sd"], float), 1.0)
cfg = load_preprocess(PURE)
SUBS = list(u["subs"])                            # DQ, TBZ, THI


def features(path):
    r = unmix_map(data_dir=PURE, test_path=path, method="dlpx",
                  baseline=bool(m.get("baseline", cfg["baseline"])),
                  trim=cfg["trim"], min_frac=0.15, hit_mode="threshold",
                  dl_model=m)
    wn = np.asarray(r.wn, float)
    X = np.clip(np.asarray(r.spectra, float), 0, None)
    mask = (wn >= m["lo"]) & (wn <= m["hi"])
    Xm = X[:, mask] if X.shape[1] == len(wn) else X
    wnm = wn[mask] if X.shape[1] == len(wn) else wn
    pk = np.clip(np.asarray(apply_model_pixels(m, wn, r.spectra), float), 0, None)
    cols = [list(m["subs"]).index(s) for s in SUBS]
    R = pk[:, cols] / (pk[:, cols].sum(1, keepdims=True) + 1e-12)
    sel = np.where((pk[:, cols].sum(1) >= 0.5) & r.hit)[0]
    if not len(sel):
        sel = np.where(pk[:, cols].sum(1) >= 0.5)[0]
    _, F, _ = _residual_context(Xm[sel], R[sel], np.zeros(len(sel), int),
                                wnm, bands, ab, cal_rng)
    return (np.asarray(F[0], float) - mu) / sd


def main():
    t0 = time.time()
    entries = json.load(open(os.path.join(PURE, "mixtures.json"), encoding="utf-8"))
    lib = []
    for i, e in enumerate(entries):
        conc = e.get("conc") or {}
        y = np.array([float(conc.get(s, 0.0)) * 1e6 for s in SUBS])
        imb = (y.max() / max(y[y > 0].min(), 1e-9)) >= 99 if (y > 0).any() else False
        try:
            z = features(e["path"])
        except Exception as ex:
            print(f"[{i}] skip {os.path.basename(e['path'])}: {ex}", flush=True)
            continue
        lib.append({"name": os.path.basename(e["path"]), "y": y.tolist(),
                    "z": z.tolist(), "imb100": bool(imb)})
        if i % 10 == 0:
            print(f"[{i}/{len(entries)}] {os.path.basename(e['path'])} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    json.dump(lib, open(os.path.join(OUT, "library.json"), "w"), indent=1)
    Z = np.array([r["z"] for r in lib]); Y = np.array([r["y"] for r in lib])
    keep = ~np.array([r["imb100"] for r in lib])
    print(f"library {len(lib)} maps ({keep.sum()} after 100:1 exclusion)", flush=True)

    def knn_predict(zq, exclude=-1, k=3):
        d = np.linalg.norm(Z - zq[None, :], axis=1)
        if exclude >= 0:
            d[exclude] = np.inf
        d[~keep] = np.inf
        idx = np.argsort(d)[:k]
        w = 1.0 / np.maximum(d[idx], 1e-9)
        w = w / w.sum()
        return (Y[idx] * w[:, None]).sum(0), float(d[idx][0]), [lib[j]["name"] for j in idx]

    # ── LOO 채점 (100:1 제외 셋, 진리 있는 맵) ──
    rows = []
    for i, r in enumerate(lib):
        if not keep[i] or not np.any(np.array(r["y"]) > 0):
            continue
        p, dmin, nn = knn_predict(np.array(r["z"]), exclude=i)
        rows.append({"map": r["name"], "true": r["y"], "pred": p.tolist(),
                     "dmin": dmin, "nn": nn})
    json.dump(rows, open(os.path.join(OUT, "loo.json"), "w"), indent=1)
    T = np.array([r["true"] for r in rows]); P = np.array([r["pred"] for r in rows])
    mmask = T > 0
    for tag, sel in (("전체", np.ones(len(rows), bool)),
                     ("검증창(전성분<=24)", np.array([max(r["true"]) <= 24 for r in rows]))):
        Ts, Ps = T[sel], P[sel]
        ms = Ts > 0
        rmse = float(np.sqrt(np.mean((Ps - Ts) ** 2)))
        fold = np.maximum(Ps[ms] / Ts[ms], Ts[ms] / np.maximum(Ps[ms], 1e-9))
        print(f"LOO {tag}: n={sel.sum()}맵  RMSE {rmse:.2f} µM  "
              f"within-2x {np.mean(fold <= 2)*100:.1f}%", flush=True)
    # ── 글씨맵 조회 ──
    zL = features(r"S:\Google Drive\내 드라이브\ACF_PEST_DB\Pest\260812_12 trio_THI_TBZ_DQ.csv")
    p, dmin, nn = knn_predict(zL)
    print(f"글씨맵: 최근접거리 {dmin:.2f}  이웃 {nn}  k-NN 반환 {np.round(p,1).tolist()}",
          flush=True)
    print(f"done {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
