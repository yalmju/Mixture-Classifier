# -*- coding: utf-8 -*-
"""Fig 4 원천 데이터 — 논문용 PLS 모델(paper/pls-composition) 기준으로 다시 뽑는다.

  1) heldout_maps.csv — 측정 혼합물 맵 전부(학습 82 + 시험 20)의 **held-out** 조성 판독.
     PLS·MLP 는 모델에 저장된 기록(조건 held-out `loo_eval` + 시험 `test_eval`)이고 다시 적용하지 않는다
     (적용하면 자기 답을 외운다). NNLS 는 같은 맵을 세 가지로 푼 것(본문 기준선은 잉크 blank BLK 하나만 넣은 NNLSb) — 파이프라인 NNLS(`unmix_map`, 게이트 0.15)와
     모델 내장 템플릿 NNLS(`surface_detection`, 열 이름 NNLSm).
     `{DQ,TBZ,THI}_band` = 게이트 통과 픽셀 평균 스펙트럼의 특성 밴드 높이(ALS 기준선 제거 뒤 ±10 cm⁻¹ 최댓값, counts),
     `band_calibration.csv` = 단일성분 희석 계열의 같은 밴드 높이 — "신호 세기 + 단일성분 검량선" 판독의 재료.
     옛 Fig 4b 원천(22_parity_uM_knowntotal_FINAL.csv)은 생성 스크립트가 없어 재현되지 않아 쓰지 않는다.
     `--letter-only` 를 주면 1) 을 건너뛴다(102맵 NNLS 가 몇 분 걸린다).
  2) lettering_pixels.npz — 글씨맵(260812, 다른 배치, 12 µM each 1:1:1)의 픽셀 단위 결과:
     NNLS 게이트·NNLS 표면 비율·밴드 세기·PLS/MLP 픽셀 조성·PLS 픽셀 µM.

실행:  python -u documentation/scripts/fig4_paper_data.py
"""
from __future__ import annotations

import csv
import os
import re
import sys

import numpy as np
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

sys.path.insert(0, paths.REPO)
import dl_model  # noqa: E402
from dataset import load_preprocess  # noqa: E402
from io_utils import load_calibration_csv  # noqa: E402
from sers_mixture import desaturate  # noqa: E402
from unmix import _baseline_removed, unmix_map  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "fig4_paper_pls_2026-11-10")
os.makedirs(OUT, exist_ok=True)
PLS = f"{paths.RESULTS}/261110_Model_PLS_paper/pls_composition_261110.dlm"
MLP = f"{paths.RESULTS}/260831_Model_FINAL/mlp_composition_260831_final.dlm"
MIXDIR = f"{paths.DATA}/Ratio/260814_mixture_final"
LETTER = f"{paths.DATA}/Pest/260812_12 trio_THI_TBZ_DQ.csv"
COMP = ("DQ", "TBZ", "THI")
GRID = {3.0, 6.0, 12.0, 24.0}
GATE = 0.15

base = lambda p: os.path.basename(str(p).replace(chr(92), "/"))  # noqa: E731


def true_uM(name):
    m = re.match(r"DQ(\d+)-TB(\d+)-THI?(\d+)", name)
    return np.array([float(m.group(i)) for i in (1, 2, 3)])


def heldout(model):
    """파일명 → (held-out 조성 3성분 정규화, 'loo'|'test')."""
    cols = [list(model["subs"]).index(s) for s in COMP]
    out = {}
    for key, tag in (("loo_eval", "loo"), ("test_eval", "test")):
        ev = model[key]
        for p, pr in zip(ev["paths"], ev["pred"]):
            n = base(p)
            if not re.match(r"DQ\d+-TB\d+-THI?\d+", n):
                continue                                   # 순수 꼭짓점·blank 는 혼합물이 아니다
            v = np.clip(np.asarray(pr, float)[cols], 0, None)
            out[n] = (v / v.sum(), tag)
    return out


def band_heights(wn, spectra, bands, hw):
    """스펙트럼(행) → ALS 기준선 제거 → 특성 밴드 ±hw 안 최댓값 (n, 3; DQ·TBZ·THI). 통상적인 '피크 높이'.
    검량 계열과 혼합물 맵에 똑같이 적용한다."""
    wn = np.asarray(wn, float)
    Y = _baseline_removed(np.atleast_2d(np.asarray(spectra, float)), True)
    return np.stack([Y[:, (wn >= bands[s] - hw) & (wn <= bands[s] + hw)].max(1) for s in COMP], 1)


def band_calibration(model, cfg):
    """단일성분 희석 계열(0.1–1000 µM)의 특성 밴드 높이 → band_calibration.csv. 맵과 같은 구간·같은 기준선 제거."""
    bands = dict(zip(model["uM"]["subs"], np.asarray(model["uM"]["bands_cm"], float)))
    hw = float(model["uM"]["band_half_width_cm"])
    ax, names, dils = load_calibration_csv(paths.calibration())
    ax = np.asarray(ax, float)
    m = (ax >= cfg["trim"][0]) & (ax <= cfg["trim"][1])
    rows = []
    for s in COMP:
        C, Y = dils[list(names).index(s)]
        Y = desaturate(np.atleast_2d(np.asarray(Y, float))[:, m])[0]
        h = band_heights(ax[m], Y, bands, hw)[:, COMP.index(s)]
        rows += [[s, bands[s], round(float(c) * 1e6, 4), round(float(v), 1)] for c, v in zip(np.asarray(C, float), h)]
    with open(os.path.join(OUT, "band_calibration.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["compound", "band_cm", "C_uM", "band_height"])
        w.writerows(sorted(rows, key=lambda r: (COMP.index(r[0]), r[2])))
    print("band_calibration.csv:", len(rows), "점", flush=True)
    return bands, hw


def nnls_bg(r, bg=("BLK",)):
    """파이프라인과 같은 스펙트럼·템플릿으로, 배경 열만 `bg` 로 제한해 다시 푼 픽셀별 3성분 비율 (n_px, 3)."""
    names = list(r.comps)
    T = np.asarray(r.templates, float)[[names.index(s) for s in COMP + tuple(bg)]].T
    A = np.array([nnls(T, s)[0] for s in np.asarray(r.spectra, float)])[:, :3]
    return A / np.clip(A.sum(1, keepdims=True), 1e-12, None)


def nnls_map(path, cfg, baseline, model, bands, hw):
    """같은 픽셀에 대한 NNLS 세 가지 + 같은 픽셀 평균 스펙트럼의 특성 밴드 높이(counts).
    NNLS  = 파이프라인 NNLS(`unmix_map`: Pure 템플릿 + 배경 BLK·INK·LEAF, 게이트 통과 픽셀 평균) — 앱이 보여 주는 값.
    NNLSb = 같은 템플릿에서 배경을 잉크 blank(BLK) 하나로 제한, 같은 게이트 통과 픽셀 평균 — 본문 기준선.
    NNLSm = 모델 내장 템플릿 P(3성분, blank 없음)로 푼 표면 비율(`surface_detection`) — 농도 단계가 쓰는 값."""
    r = unmix_map(data_dir=paths.PURE, test_path=path, method="nnls", baseline=baseline,
                  trim=cfg["trim"], min_frac=GATE, hit_mode="threshold")
    nb = [r.comps[i] for i in r.nonbg]
    v = np.array([float(r.mean_ratio[nb.index(s)]) for s in COMP])
    frac, _ = dl_model.surface_detection(model, r.wn, r.spectra, max_px=10 ** 9)
    u = np.array([frac[s] for s in COMP])
    hit = np.asarray(r.hit, bool)
    sel = hit if hit.any() else slice(None)
    b = nnls_bg(r)[sel].mean(0)
    bh = band_heights(r.wn, np.asarray(r.spectra, float)[sel].mean(0), bands, hw)[0]
    return (v / max(v.sum(), 1e-12), b / max(b.sum(), 1e-12), u / max(u.sum(), 1e-12), bh,
            int(np.sum(r.hit)), int(r.n_pixels))


def main():
    pls, mlp = dl_model.load_model(PLS), dl_model.load_model(MLP)
    cfg = load_preprocess(paths.PURE)
    baseline = bool(pls.get("baseline", cfg["baseline"]))
    hp, hm = heldout(pls), heldout(mlp)
    assert set(hp) == set(hm), "두 모델의 held-out 맵 목록이 다르다"

    if "--letter-only" not in sys.argv:
        heldout_table(pls, cfg, baseline, hp, hm)
    lettering(pls, mlp, cfg, baseline)


def heldout_table(pls, cfg, baseline, hp, hm):
    bands, hw = band_calibration(pls, cfg)
    rows = []
    for i, n in enumerate(sorted(hp, key=lambda s: (tuple(true_uM(s)), s))):
        t = true_uM(n)
        nn, nb, nm, bh, nhit, npx = nnls_map(os.path.join(MIXDIR, n), cfg, baseline, pls, bands, hw)
        rows.append([n[:-4], hp[n][1], int(set(t) <= GRID), *t, *np.round(nn, 4), *np.round(nb, 4), *np.round(nm, 4),
                     *np.round(hp[n][0], 4), *np.round(hm[n][0], 4), *np.round(bh, 1), nhit, npx])
        if i % 10 == 0:
            print(f"[{i + 1}/{len(hp)}] {n}  NNLS {np.round(nn * 100).astype(int)}  NNLSb {np.round(nb * 100).astype(int)}  NNLSm {np.round(nm * 100).astype(int)}"
                  f"  hit {nhit}/{npx}", flush=True)
    with open(os.path.join(OUT, "heldout_maps.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "split", "grid64"] + [f"{k}_true_uM" for k in COMP]
                   + [f"{k}_{m}" for m in ("NNLS", "NNLSb", "NNLSm", "PLS", "MLP") for k in COMP]
                   + [f"{k}_band" for k in COMP] + ["nnls_hit_px", "n_px"])
        w.writerows(rows)
    print("grid64:", sum(r[2] for r in rows), "· 전체", len(rows), flush=True)


def lettering(pls, mlp, cfg, baseline):
    from dl_quantify import surface_composition
    r = unmix_map(data_dir=paths.PURE, test_path=LETTER, method="dlpx", baseline=baseline,
                  trim=cfg["trim"], min_frac=GATE, hit_mode="threshold", dl_model=pls)
    wn, S = np.asarray(r.wn, float), np.asarray(r.spectra, float)
    hit = np.asarray(r.hit, bool)
    names = list(r.comps)
    A_ls = np.asarray(r.A_evidence, float)                  # NNLS 계수 (모델이 덮어쓰기 전)
    nn = A_ls[:, [names.index(s) for s in COMP]]
    nn = nn / np.clip(nn.sum(1, keepdims=True), 1e-12, None)
    nb = nnls_bg(r)                                          # 잉크 blank(BLK) 하나만 넣은 NNLS (본문 기준선)
    mk = (wn >= pls["lo"]) & (wn <= pls["hi"])               # 모델 내장 템플릿 NNLS (heldout 표의 NNLSm 과 같은 정의)
    nm = np.asarray(surface_composition(dl_model._composition_features(np.clip(S[:, mk], 0, None), "legacy_l2"),
                                        pls["P"]), float)
    nm = nm[:, [list(pls["subs"]).index(s) for s in COMP]]
    nm = nm / np.clip(nm.sum(1, keepdims=True), 1e-12, None)
    px = {}
    for tag, m in (("PLS", pls), ("MLP", mlp)):
        pk = np.clip(np.asarray(dl_model.apply_model_pixels(m, wn, S), float), 0, None)
        c = pk[:, [list(m["subs"]).index(s) for s in COMP]]
        px[tag] = c / np.clip(c.sum(1, keepdims=True), 1e-12, None)
        px[tag + "_blank"] = pk[:, list(m["subs"]).index(m["blank"])] / np.clip(pk.sum(1), 1e-12, None)
    bands = dict(zip(pls["uM"]["subs"], np.asarray(pls["uM"]["bands_cm"], float)))
    hw = float(pls["uM"]["band_half_width_cm"])
    band = np.stack([S[:, (wn >= bands[s] - hw) & (wn <= bands[s] + hw)].max(1) for s in COMP], 1)
    um, usubs = dl_model.apply_uM_pixels(pls, wn, S, hit=hit)
    um = np.asarray(um, float)[:, [list(usubs).index(s) for s in COMP]]
    np.savez(os.path.join(OUT, "lettering_pixels.npz"), coords=np.asarray(r.coords, float), hit=hit,
             nnls=nn, nnls_b=nb, nnls_m=nm, band=band, pls=px["PLS"], mlp=px["MLP"], pls_blank=px["PLS_blank"],
             mlp_blank=px["MLP_blank"], uM_pls=um, comp=np.array(COMP), bands_cm=np.array([bands[s] for s in COMP]))
    print(f"글씨맵: hit {hit.sum()}/{len(hit)} px", flush=True)
    for tag, v in (("NNLS", nn), ("NNLSb", nb), ("NNLSm", nm), ("PLS", px["PLS"]), ("MLP", px["MLP"])):
        print(f"  {tag} 픽셀 평균 DQ/TBZ/THI", np.round(v[hit].mean(0) * 100, 1), flush=True)
    print("  NNLS 로 THI 가 최대인 픽셀", int((nn[hit].argmax(1) == 2).sum()), "/", int(hit.sum()))
    print("  PLS 픽셀 µM 중앙값", np.round(np.median(um[hit], 0), 1), "(참 12/12/12)")


if __name__ == "__main__":
    main()
