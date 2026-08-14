"""Hybrid(MLP+Physics) 농도 — held-out µM 판독. **보고값은 이쪽이다.**

  ## conc38 과의 관계

  `conc38.py` 는 `s = g · r · C` 로 신호가 농도에 **비례**한다고 가정한다. DQ 는
  신호–농도 지수가 0.52 라 그 가정이 깨져 있고, 그래서 conc38 의 DQ 는 2배 이내 65% 에
  멈춘다. 여기의 hybrid 는 채택 구성 MLP 의 **농도 헤드**를 그대로 쓴다 — 검량에서 적합한
  K·gA·Sips m 으로 경쟁 등온선 혼합물을 시뮬레이션해 사전학습하고 실측 맵으로 미세조정한
  것이라, 등온선의 휘어짐(포화·경쟁)을 알고 역산한다. 논문 그림(패널 e: 2배 이내
  THI 85 · TBZ 95 · DQ 77%)이 쓰는 경로가 이것이다.

  ## 왜 이 스크립트가 새로 생겼나

  2026-08-13 의 `conc_*_hybrid_origin.csv` 를 만든 스크립트가 저장소에 없었다
  (`tert260805_nnls.npz` · `lod_loq_inkbasis.csv` 에 이어 세 번째). 여기서 다시 만들 수
  있게 한다. 검증은 `retrain_all.py` 와 같은 4-fold **조건단위** 분할(rng seed 0)이라
  held-out 이다 — 맵의 µM 예측은 그 맵이 빠진 fold 의 모델에서 나온다.

  나오는 것 (`{INTERP}/panels/`)
    conc_hybrid_parity.csv    맵 × 성분 — 참/추정/배수오차 (conc38 값도 나란히)
    conc_hybrid_summary.csv   성분 × {all, ge6uM} — 중앙 배수오차 · 2배/3배 이내

  실행:  python3 -u conc_hybrid.py [epochs=120] [folds=4]
"""
import os as _os, sys as _sys                 # paths 부트스트랩 — 기계마다 마운트가 다르다
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import os, sys, csv, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, paths.REPO)

import mixtures as MX
import dl_model
from real_data import load_map

SUB = MX.SUB
OUT = f"{paths.INTERP}/panels"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
FOLDS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
os.makedirs(OUT, exist_ok=True)

CALIB = paths.calibration()
paths.check_pretrain(CALIB, True)

# retrain_all/piemap 과 같은 수집·분할 — 고농도는 항상 학습, 저농도 조건을 fold 로 나눈다.
HI = MX.items(("high",))
LO = MX.groups(("grid",), replicates=True)
keys = sorted(LO)
rng = np.random.default_rng(0)
order = rng.permutation(len(keys))
folds = [[keys[i] for i in order[f::FOLDS]] for f in range(FOLDS)]
print(f"고농도 {len(HI)}맵 · 저농도 {len(keys)}조건 · {FOLDS}-fold · epochs={EPOCHS}", flush=True)

# conc38 추정을 나란히 놓는다 (있으면). 같은 맵·같은 held-out 규약이다.
_c38 = {}
_p38 = f"{HERE}/conc38_result_cliponly.npz"
if os.path.exists(_p38):
    z = np.load(_p38, allow_pickle=True)
    for p, e in zip(z["paths"], np.asarray(z["est"], float)):
        _c38[os.path.basename(str(p))] = e

rows = []
t0 = time.time()
for fi, te in enumerate(folds):
    tr = [k for k in keys if k not in te]
    print(f"\n=== fold {fi + 1}/{FOLDS} — 학습 {len(tr)}조건 / 검증 {len(te)}조건 "
          f"({time.time() - t0:.0f}s)", flush=True)
    m = dl_model.train_model(paths.PURE, HI + MX.items_for(LO, tr), calib_path=CALIB,
                             baseline=True, trim=None, method="mlp", epochs=EPOCHS,
                             seed=0, use_pretrain=True, nnls_screen=True, px_per_map=20,
                             include_blank=True, sim_nuisance=False, sim_iso="sips_dq",
                             progress=lambda s: print("   ", s, flush=True)
                             if "isotherm" in s else None)
    for k in te:
        true = dict(zip(SUB, map(float, k)))
        for p in LO[k]:
            wn, cube, _mm, _co = load_map(p)
            r = dl_model.apply_model(m, np.asarray(wn, float), np.asarray(cube, float))
            uM = r.get("uM") or {}
            c38 = _c38.get(os.path.basename(p))
            for j, s in enumerate(SUB):
                t = true[s]
                est = float(uM.get(s, float("nan"))) if uM else float("nan")
                fold_h = (max(est / t, t / est)
                          if t > 0 and np.isfinite(est) and est > 0 else float("nan"))
                e38 = float(c38[j]) if c38 is not None else float("nan")
                fold_c = (max(e38 / t, t / e38)
                          if t > 0 and np.isfinite(e38) and e38 > 0 else float("nan"))
                rows.append([os.path.basename(p).replace("_corrected.csv", "").replace(".csv", ""),
                             s, t, est, fold_h, e38, fold_c])
            print(f"   {os.path.basename(p):34s} " +
                  " ".join(f"{s} {true[s]:.0f}->{float(uM.get(s, float('nan'))):5.1f}"
                           for s in SUB), flush=True)

with open(f"{OUT}/conc_hybrid_parity.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["map", "compound", "true_uM", "est_hybrid_uM", "fold_hybrid",
                "est_conc38_uM", "fold_conc38"])
    w.writerows([[r[0], r[1], f"{r[2]:g}"] + [f"{v:.4f}" if np.isfinite(v) else ""
                                              for v in r[3:]] for r in rows])
print(f"\n  ✓ conc_hybrid_parity.csv   {len(rows)}행")

summ = []
print(f"\n{'':>5} {'구간':>7} {'n':>4}   {'hybrid 중앙':>10} {'2배':>5} {'3배':>5}   "
      f"{'conc38 중앙':>10} {'2배':>5}")
for s in SUB:
    for lab, lo_ in (("all", 0.0), ("ge6uM", 6.0)):
        v = [(r[4], r[6]) for r in rows if r[1] == s and r[2] >= lo_ and np.isfinite(r[4])]
        if not v:
            continue
        h = np.array([x[0] for x in v])
        c = np.array([x[1] for x in v if np.isfinite(x[1])])
        summ.append([s, lab, len(h), f"{np.median(h):.4f}",
                     f"{100 * (h < 2).mean():.2f}", f"{100 * (h < 3).mean():.2f}",
                     f"{np.median(c):.4f}" if len(c) else "",
                     f"{100 * (c < 2).mean():.2f}" if len(c) else ""])
        print(f"{s:>5} {lab:>7} {len(h):>4}   {np.median(h):10.2f} "
              f"{100 * (h < 2).mean():4.0f}% {100 * (h < 3).mean():4.0f}%   "
              + (f"{np.median(c):10.2f} {100 * (c < 2).mean():4.0f}%" if len(c) else ""))

with open(f"{OUT}/conc_hybrid_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["compound", "group", "n", "median_fold_hybrid", "within_2x_pct",
                "within_3x_pct", "median_fold_conc38", "conc38_within_2x_pct"])
    w.writerows(summ)
print(f"  ✓ conc_hybrid_summary.csv")
print(f"\n{time.time() - t0:.0f}s · → {OUT}")
print("보고값은 hybrid 열이다. conc38 열은 '물리 없이 비례 가정만으로 얼마나 되나' 의 대조군 —")
print("특히 DQ 에서 갈린다 (지수 0.52 라 비례 가정이 깨져 있다).")
