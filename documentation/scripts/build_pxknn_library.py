# -*- coding: utf-8 -*-
"""픽셀 k-NN 라이브러리(<dlm>.pxknn.npz) 재구축 — hit 픽셀 전부.

이전 npz는 맵당 24픽셀(검량 10픽셀) 서브샘플 + 4개 맵 누락이라, 라이브러리에
있는 조건을 열어도 자기 픽셀을 못 찾았다(33:33:33 → 12). 여기서는
  · Pure/mixtures.json의 모든 맵(100:1 불균형 맵은 map k-NN과 같은 규칙으로 제외)
    → 게이트 통과(hit) 픽셀 전부
  · 260821_Calib_144-9의 단일성분 검량 스펙트럼(파일당 5 반복; CAL-<sub>-<µM>)
    → 표준 맵 형식으로 임시 변환해 같은 unmix 경로로 처리, 5개 전부
를 담는다. 피처 7개는 page_real._pxknn_lookup과 동일:
log1p(마커 밴드 신호 3) · log1p(총합) · 모델 조성 3, 전체 z-score.
npz에는 원 피처 F도 함께 저장(재구축·추가 시 재-unmix 불필요).

실행:  python -u build_pxknn_library.py            # 전체 재계산(≈17분)
       python -u build_pxknn_library.py --reuse    # 혼합물 피처는 기존 npz 재사용,
                                                   # 검량만 다시 계산
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import sys
import tempfile
import time

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from dataset import load_preprocess                  # noqa: E402
from dl_model import load_model, _band_signal         # noqa: E402
from unmix import unmix_map                           # noqa: E402

DB = r"S:\Google Drive\내 드라이브\ACF_PEST_DB"
DLM = os.path.join(DB, "260831_Model_FINAL", "mlp_composition_260831_final.dlm")
PURE = os.path.join(DB, "Pure")
CAL = os.path.join(DB, "260821_Calib_144-9")
NPZ = os.path.splitext(DLM)[0] + ".pxknn.npz"
SUBS = ["DQ", "TBZ", "THI"]

m = load_model(DLM)
cfg = load_preprocess(PURE)
bands = np.asarray(m["uM"]["bands_cm"], float)
assert [s for s in m["subs"] if s in SUBS] == SUBS, m["subs"]


def pixel_features(path, all_px=False):
    r = unmix_map(data_dir=PURE, test_path=path, method="dlpx",
                  baseline=bool(m.get("baseline", cfg["baseline"])),
                  trim=cfg["trim"], min_frac=0.15, hit_mode="threshold",
                  dl_model=m)
    hit = np.ones(r.n_pixels, bool) if all_px else np.asarray(r.hit, bool)
    if not hit.any():
        return None
    wn = np.asarray(r.wn, float)
    X = np.clip(np.asarray(r.spectra, float), 0, None)[hit]
    sig = np.log1p(np.clip(_band_signal(X, wn, bands), 0, None))
    tot = np.log1p(X.sum(1))[:, None]
    Rq = np.clip(np.asarray(r.ratio_nb, float), 0, None)[hit]
    assert Rq.shape[1] == 3, (path, Rq.shape, list(r.nonbg))
    return np.hstack([sig, tot, Rq])


def cal_as_map(src, dst):
    """wavenumber × spec_01..05(+mean,sd) → 'X num,5 / Y num,1 / X,Y,wn… / rows'."""
    rows = [ln.split(",") for ln in open(src, encoding="utf-8-sig")
            .read().splitlines() if ln.strip()]
    head = rows[0]
    cols = [j for j, h in enumerate(head) if h.strip().startswith("spec_")]
    wn = [r_[0] for r_ in rows[1:]]
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"X num,{len(cols)}\nY num,1\n")
        f.write("X,Y," + ",".join(wn) + "\n")
        for i, j in enumerate(cols):
            f.write(f"{i},0," + ",".join(r_[j] for r_ in rows[1:]) + "\n")
    return len(cols)


def cal_jobs(tmp):
    out = []
    for fn in sorted(os.listdir(CAL)):
        mm = re.match(r"^(DQ|TBZ|THI)-?(\d+)", fn)
        if not mm or not fn.endswith(".csv"):
            continue
        s, c = mm.group(1), float(mm.group(2))
        y = np.zeros(3)
        y[SUBS.index(s)] = c
        dst = os.path.join(tmp, f"CAL-{s}-{int(c)}__{len(out)}.csv")
        cal_as_map(os.path.join(CAL, fn), dst)
        out.append((dst, f"CAL-{s}-{int(c)}", y, True))
    return out


def mixture_jobs():
    out = []
    for e in json.load(open(os.path.join(PURE, "mixtures.json"),
                            encoding="utf-8")):
        conc = e.get("conc") or {}
        y = np.array([float(conc.get(s, 0.0)) * 1e6 for s in SUBS])
        imb = ((y.max() / max(y[y > 0].min(), 1e-9)) >= 99
               if (y > 0).any() else False)
        if imb:
            print("skip 100:1", os.path.basename(e["path"]))
            continue
        out.append((e["path"], os.path.basename(e["path"]), y, False))
    return out


def run_jobs(jobs, t0):
    F_all, Y_all, C_all = [], [], []
    for i, (path, name, y, all_px) in enumerate(jobs):
        try:
            F = pixel_features(path, all_px)
        except Exception as ex:
            print(f"[{i}] FAIL {name}: {ex}", flush=True)
            continue
        if F is None:
            print(f"[{i}] no hit pixels: {name}", flush=True)
            continue
        F_all.append(F)
        Y_all.append(np.tile(y, (len(F), 1)))
        C_all += [name] * len(F)
        print(f"[{i}/{len(jobs)}] {name}: {len(F)} px ({time.time()-t0:.0f}s)",
              flush=True)
    return F_all, Y_all, C_all


def main():
    t0 = time.time()
    reuse = "--reuse" in sys.argv
    if reuse:
        old = np.load(NPZ, allow_pickle=True)
        cond = old["cond"].astype(str)
        F0 = (np.asarray(old["F"], float) if "F" in old
              else np.asarray(old["Fz"], float) * old["sd"] + old["mu"])
        keep = ~np.char.startswith(cond, "CAL-")
        F_all = [F0[keep]]
        Y_all = [np.asarray(old["Y"], float)[keep]]
        C_all = list(cond[keep])
        print(f"reused {keep.sum()} mixture px / "
              f"{len(set(C_all))} conditions from {os.path.basename(NPZ)}")
    else:
        F_all, Y_all, C_all = run_jobs(mixture_jobs(), t0)
    with tempfile.TemporaryDirectory() as tmp:
        Fc, Yc, Cc = run_jobs(cal_jobs(tmp), t0)
    F_all += Fc
    Y_all += Yc
    C_all += Cc

    F = np.vstack(F_all)
    Y = np.vstack(Y_all)
    cond = np.asarray(C_all)
    mu = F.mean(0)
    sd = F.std(0) + 1e-9
    Fz = (F - mu) / sd
    if os.path.exists(NPZ):
        bak = NPZ + ".bak-24px"
        if not os.path.exists(bak):
            shutil.copy2(NPZ, bak)
            print("backup ->", bak)
    np.savez(NPZ, Fz=Fz.astype(np.float32), Y=Y.astype(np.float32),
             cond=cond, mu=mu, sd=sd, subs=np.asarray(SUBS),
             F=F.astype(np.float32))
    n_cal = sum(c.startswith("CAL-") for c in C_all)
    print(f"saved {NPZ}: {len(F)} px ({n_cal} calibration), "
          f"{len(set(C_all))} conditions ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
