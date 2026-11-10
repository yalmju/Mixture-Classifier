# -*- coding: utf-8 -*-
"""Presence 헤드 프로토타입 — 성분별 존재/부재를 명시적 분류 문제로.

배경: 회귀(softmax 조성·µM)는 0을 출력하지 못하고, 고정 NNLS 문턱은 부재
바닥값이 파트너 맥락에 따라 움직여(14_absence_floor_curves) 전역으로 성립하지
않는다. 여기서는 배경-인지 NNLS 분율 + 마커밴드 세기 + 파트너 맥락을 feature 로
하는 성분별 분류기를 **기존 데이터만으로** 학습해 held-out 으로 채점한다.

데이터 (맵 단위, (맵, 성분) = 1 샘플):
  - 89맵 번들 (grid64 64 + binary 등): 양성 ~247 / 음성(부재) 27
  - 검량 단일성분 35맵: 양성 35 / 음성 70  ← "파트너 있는 저농도 0"의 유일한 실측
누출 방지: 학습형 조성모델 출력은 feature 로 쓰지 않는다 (NNLS·밴드만).
CV: 조건(절대농도 키) 단위 grouped 5-fold.

실행:  python -u run_presence_head_prototype.py
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
MAIN = Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(MAIN / "documentation" / "scripts"))
import run_integrated_reevaluation as R  # noqa: E402
import dl_model  # noqa: E402
from scipy.optimize import nnls as scinnls  # noqa: E402

CAL = Path(r"S:\Google Drive\내 드라이브\ACF_PEST_DB\260821_Calib_144-9")
DLM = Path(r"S:\Google Drive\내 드라이브\ACF_PEST_DB\260826_Model\mlp_composition_260826.dlm")
OUT = Path(__file__).resolve().parents[1] / "results" / "presence_head_20260831"
SUBS_ORD = ["THI", "TBZ", "DQ"]                    # 보고 순서
BAND = {"DQ": 1570.0, "TBZ": 1270.0, "THI": 1367.0}


def pixel_features(raw_full, mask_r, PA, n_an, axis):
    """픽셀 배열 → 맵 feature dict 재료. raw_full: (n_px, 2001) baseline-removed >=0."""
    fr_an, shares = [], []
    for y in raw_full[:, mask_r]:
        yn = y / (np.linalg.norm(y) + 1e-12)
        w, _ = scinnls(PA.T, yn)
        a = w[:n_an]
        shares.append(a.sum() / (w.sum() + 1e-12))
        fr_an.append(a / (a.sum() + 1e-12))
    fr_an = np.asarray(fr_an); shares = np.asarray(shares)
    hit = shares >= 0.15
    use = hit if hit.any() else np.ones(len(shares), bool)
    band_sig = {}
    for s, c in BAND.items():
        m = np.abs(axis - c) <= 10.0
        band_sig[s] = float(np.median(raw_full[use][:, m].max(1)))
    return {
        "frac": fr_an[use].mean(0),                       # 배경-인지 NNLS 조성 (n_an,)
        "analyte_share": float(shares[use].mean()),
        "hit_fraction": float(hit.mean()),
        "band": band_sig,
        "total": float(np.median(raw_full[use].sum(1))),
    }


def build_rows():
    model = dl_model.load_model(str(DLM))
    subs_r, wn_r, mask_r, P, lo, hi = dl_model._refs(
        model.get("data_dir"), model.get("baseline", True), model.get("trim"))
    NU = dl_model._nuisance_refs(model.get("data_dir"), model.get("baseline", True), mask_r)
    PA = np.vstack([P, NU]) if NU is not None else P
    axis = R.AXIS_CM
    rows = []

    z = np.load(R.BUNDLE, allow_pickle=True)
    X = z["X"].astype(np.float32); names = z["names"].astype(object)
    C = z["concentration"].astype(float)
    raw_all = np.expm1(np.clip(X, 0, None))
    for name, idx in R.map_rows(names):
        f = pixel_features(raw_all[idx], mask_r, PA, len(subs_r), axis)
        conc = C[idx[0]]
        gkey = "c:" + ",".join(f"{v:.8g}" for v in conc)
        rows.append({"map": name, "group": gkey, "src": "bundle",
                     "true": {s: float(conc[j]) for j, s in enumerate(subs_r)}, **f})

    pat = re.compile(r"^(DQ|TBZ|THI)[-]?(\d+)")
    for fn in sorted(os.listdir(CAL)):
        m = pat.match(fn)
        if not fn.endswith(".csv") or not m:
            continue
        comp, conc = m.group(1), int(m.group(2))
        dat = np.genfromtxt(CAL / fn, delimiter=",", skip_header=1)
        cube = np.clip(dat[:, 1:6].T, 0, None)
        f = pixel_features(cube, mask_r, PA, len(subs_r), axis)
        rows.append({"map": fn, "group": f"cal:{comp}{conc}", "src": "calib",
                     "true": {s: (float(conc) if s == comp else 0.0) for s in subs_r}, **f})
    return rows, subs_r


def feature_vector(r, s, subs_r):
    j = subs_r.index(s)
    others = [k for k in range(len(subs_r)) if k != j]
    return [
        float(r["frac"][j]),                                   # 자기 NNLS 분율
        float(max(r["frac"][k] for k in others)),              # 최대 파트너 분율
        float(np.log1p(r["band"][s])),                         # 자기 마커밴드 세기
        float(np.log1p(max(r["band"][subs_r[k]] for k in others))),
        float(np.log1p(r["total"])),                           # 총 신호
        float(r["analyte_share"]),                             # 분석물 대 배경 비
    ]


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, subs_r = build_rows()
    print(f"{len(rows)} maps featurised ({time.time()-t0:.0f}s)")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score

    results, preds_out = {}, []
    for s in SUBS_ORD:
        F = np.array([feature_vector(r, s, subs_r) for r in rows])
        y = np.array([1 if r["true"][s] > 0 else 0 for r in rows])
        groups = np.array([r["group"] for r in rows], object)
        folds = R.grouped_folds(groups, 5, 701)
        prob = np.full(len(rows), np.nan)
        for held in folds:
            te = np.array([g in held for g in groups], bool)
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
            clf.fit(F[~te], y[~te])
            prob[te] = clf.predict_proba(F[te])[:, 1]
        auc = roc_auc_score(y, prob)
        # 3-상태 문턱: ND < 0.2 <= indeterminate <= 0.8 < detected
        det = prob > 0.8; nd = prob < 0.2
        fp = np.sum(det & (y == 0)); fn = np.sum(nd & (y == 1))
        ind = np.sum(~det & ~nd)
        results[s] = {"auc": float(auc), "n_present": int(y.sum()), "n_absent": int((1 - y).sum()),
                      "absent_called_detected": int(fp), "present_called_ND": int(fn),
                      "indeterminate": int(ind)}
        print(f"{s:<4} AUC {auc:.3f}  | absent->Detected {fp}/{(1-y).sum()}  "
              f"present->ND {fn}/{y.sum()}  indeterminate {ind}/{len(y)}")
        for r, p, yy in zip(rows, prob, y):
            preds_out.append({"component": s, "map": r["map"], "src": r["src"],
                              "true_uM": r["true"][s], "present": int(yy),
                              "prob": round(float(p), 4),
                              "state": "Detected" if p > 0.8 else ("ND" if p < 0.2 else "Indeterminate")})

    # 관심 사례: 8:1 소수 성분 (true 3 µM, 파트너 24 µM) 이 어떻게 분류됐나
    print("\n8:1 소수 성분(3 µM under 24 µM)의 판정 분포:")
    for s in SUBS_ORD:
        sel = [p for p in preds_out if p["component"] == s and p["src"] == "bundle"
               and p["true_uM"] == 3.0]
        st = {"Detected": 0, "Indeterminate": 0, "ND": 0}
        for p in sel:
            st[p["state"]] += 1
        print(f"  {s:<4} (n={len(sel)}): {st}")
    # 부재 사례 판정 분포 (소스별)
    print("\n부재 성분 판정 분포:")
    for src in ("bundle", "calib"):
        sel = [p for p in preds_out if p["present"] == 0 and p["src"] == src]
        st = {"Detected": 0, "Indeterminate": 0, "ND": 0}
        for p in sel:
            st[p["state"]] += 1
        print(f"  {src:<7} (n={len(sel)}): {st}")

    with (OUT / "presence_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(preds_out[0])); w.writeheader(); w.writerows(preds_out)
    R.dump_json(OUT / "presence_summary.json", {"per_component": results,
                "protocol": "grouped 5-fold (seed 701) · features: bg-aware NNLS + marker bands + intensity · logistic"})
    print(f"\ndone {(time.time()-t0):.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
