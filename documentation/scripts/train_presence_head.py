# -*- coding: utf-8 -*-
"""Presence 헤드 최종 적합 → 배포 사이드카(<dlm>.presence.json) 생성.

프로토타입(run_presence_head_prototype.py)의 grouped 5-fold 검증을 통과한 구성
그대로, 124맵 전체(89맵 번들 + 검량 단일맵 35장)에 최종 적합해 저장한다.
feature 정의는 dl_model.presence_feature_vector 와 공유 — 여기서 따로 만들지
않는다 (불일치가 곧 무효).

실행:  python -u train_presence_head.py
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
WT = Path(__file__).resolve().parents[2]          # 워크트리 루트 (수정된 dl_model)
sys.path.insert(0, str(WT))
MAIN = Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
sys.path.insert(1, str(MAIN / "documentation" / "scripts"))
import dl_model  # noqa: E402  (worktree 버전 — presence 함수 포함)
import run_integrated_reevaluation as R  # noqa: E402

CAL = Path(r"S:\Google Drive\내 드라이브\ACF_PEST_DB\260821_Calib_144-9")
DLM = Path(r"S:\Google Drive\내 드라이브\ACF_PEST_DB\260826_Model\mlp_composition_260826.dlm")
SUBS_ORD = ["THI", "TBZ", "DQ"]


def main():
    t0 = time.time()
    assert Path(dl_model.__file__).resolve().is_relative_to(WT), "worktree dl_model 이 아님"
    model = dl_model.load_model(str(DLM))
    axis = R.AXIS_CM
    rows = []

    z = np.load(R.BUNDLE, allow_pickle=True)
    X = z["X"].astype(np.float32); names = z["names"].astype(object)
    C = z["concentration"].astype(float)
    raw_all = np.expm1(np.clip(X, 0, None))
    subs_r = None
    for name, idx in R.map_rows(names):
        ctx = dl_model.presence_map_context(model, raw_all[idx], axis)
        subs_r = ctx["subs"]
        conc = C[idx[0]]
        rows.append({"ctx": ctx, "true": {s: float(conc[j]) for j, s in enumerate(subs_r)}})

    pat = re.compile(r"^(DQ|TBZ|THI)[-]?(\d+)")
    for fn in sorted(os.listdir(CAL)):
        m = pat.match(fn)
        if not fn.endswith(".csv") or not m:
            continue
        comp, conc = m.group(1), int(m.group(2))
        dat = np.genfromtxt(CAL / fn, delimiter=",", skip_header=1)
        ctx = dl_model.presence_map_context(model, np.clip(dat[:, 1:6].T, 0, None), dat[:, 0])
        rows.append({"ctx": ctx, "true": {s: (float(conc) if s == comp else 0.0) for s in subs_r}})
    print(f"{len(rows)} maps featurised ({time.time()-t0:.0f}s)")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    head = {"version": "presence_v1_20260831",
            "feature_names": ["nnls_self_frac", "nnls_partner_max", "log1p_band_self",
                              "log1p_band_partner_max", "log1p_total", "analyte_share"],
            "thresholds": {"nd": 0.2, "detected": 0.8},
            "training": {"n_maps": len(rows),
                         "sources": "bundle_89maps_24px + 260821_Calib_144-9 (35)",
                         "validation": "grouped5fold seed701 — AUC THI .997 / TBZ .940 / DQ .910",
                         "doc": "documentation/PRESENCE_HEAD_2026-08-31.md"},
            "components": {}}
    for s in SUBS_ORD:
        F = np.array([dl_model.presence_feature_vector(r["ctx"], s) for r in rows])
        y = np.array([1 if r["true"][s] > 0 else 0 for r in rows])
        sc = StandardScaler().fit(F)
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(sc.transform(F), y)
        head["components"][s] = {"mu": sc.mean_.tolist(), "sd": sc.scale_.tolist(),
                                 "coef": clf.coef_[0].tolist(),
                                 "intercept": float(clf.intercept_[0])}
        p = clf.predict_proba(sc.transform(F))[:, 1]
        print(f"{s:<4} fit n={len(y)} (+{y.sum()}/-{(1-y).sum()})  "
              f"in-sample: absent p<0.2 {np.mean(p[y==0]<0.2)*100:.0f}%  present p>0.8 {np.mean(p[y==1]>0.8)*100:.0f}%")

    side = os.path.splitext(str(DLM))[0] + ".presence.json"
    with open(side, "w", encoding="utf-8") as f:
        json.dump(head, f, indent=1)
    print("saved sidecar:", side)


if __name__ == "__main__":
    main()
