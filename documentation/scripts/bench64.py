"""방법 비교(패널 d)와 검출 ROC(패널 e) — 64조건 격자로 다시.

  디스크에 남아 있던 유일한 벤치마크 export 는 `code results/20260731-03/model bench/`
  로 2026-07-31 자다. 64조건 격자 이전이고 숫자도 논문 그림과 다르다. 자료가 없으니
  다시 돌린다.

  데이터 수집은 `mixtures.py` 한 곳이다 — 라벨은 `260814_mixture_final` 파일명이 최종 µM
  이고, 여기서 나눗셈을 하지 않는다. (예전에는 고농도 파일명을 그대로 µM 로 읽어 이성분
  2배·삼성분 3배 과대였다. `mixtures.py` 머리말 참조.)
  다만 `benchmark_loo` 는 방법 비교용이라 `include_blank` · `sim_iso` 를 받지 않는다 —
  **채택 구성이 아니라 base 구성에서의 방법 비교**다. 표에 그렇게 적을 것.

  실행:  python3 -u bench64.py [epochs=120] [px_per_map=20] [n_trees=100] [cnn_epochs=40]

  ⚠ CNN 은 길이 2001 입력을 full-batch 로 돌아 fold 당 5분이었다. epoch 을 40 으로
    줄여 감당 가능한 시간에 맞췄다. **CNN 이 덜 학습된 상태로 채점된다** — 'CNN 이
    MLP 보다 나쁘다' 를 근거로 쓸 때 이 사실을 같이 적어야 한다.

  RF 가 병목이다. sklearn 의 `RandomForestRegressor` 는 회귀에서 `max_features=1.0`
  이라 분할마다 특징 2001개를 전부 본다 — 300트리면 fold 하나에 5분, 68 fold 면 6시간.
  트리를 100 으로 줄인다. RF 는 여기서 고전 ML 기준선일 뿐이고 성능은 100↔300 에서
  포화한다 — 채택 방법(MLP)과의 격차 안에 묻힌다. 캡션에 트리 수를 적을 것.
"""
import os as _os, sys as _sys                 # paths 부트스트랩 — 기계마다 마운트가 다르다
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import sys, os, re, glob, csv, json, hashlib, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
import dl_model

import mixtures as MX

DB = paths.DB
PURE = paths.PURE
CALIB = paths.calibration()          # 하드코딩하면 조용히 사전학습이 꺼진다
OUT = f"{paths.INTERP}/panels"
SUB = MX.SUB
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
PXPM = int(sys.argv[2]) if len(sys.argv) > 2 else 20
TREES = int(sys.argv[3]) if len(sys.argv) > 3 else 100
CNN_EP = int(sys.argv[4]) if len(sys.argv) > 4 else 40
os.makedirs(OUT, exist_ok=True)


# 등몰 계열(equimolar)은 뺀다 — 그 8맵은 "조성에 희석 정보가 0" 인 판별 장치라 학습셋에
# 넣지 않기로 한 세트다 (TODO.md §6). 예전 수집과 같은 구성이다.
items = MX.items(("high", "grid"), replicates=True)

METHODS = [("nnls", "NNLS"), ("pls", "PLS"), ("rf", "Random Forest"),
           ("cnn", "1D-CNN"), ("mlp", "MLP")]
THR = 0.05                                   # 존재 판정: 참 분율 5% 초과

print(f"{len(items)}맵 · epochs={EPOCHS} · px/map={PXPM} · RF {TREES} trees, max_features=sqrt · CNN {CNN_EP} epochs", flush=True)
t0 = time.time()
# 방법마다 따로 돌리고 **끝나는 즉시 저장**한다. 예전에는 다섯 방법을 한 번에 돌고
# 마지막에 CSV 를 썼는데, CNN 설정을 바꾸려고 중간에 멈추자 이미 끝난 NNLS·PLS·RF 가
# 통째로 날아갔다. 다시 돌리면 저장된 방법은 건너뛴다.
CKPT = f"{OUT}/.bench64_partial"
os.makedirs(CKPT, exist_ok=True)
res = {}
for _meth in [m for m, _l in METHODS]:
    _f = f"{CKPT}/{_meth}.npz"
    if os.path.exists(_f):
        _z = np.load(_f, allow_pickle=True)
        res[_meth] = {"true": _z["true"], "pred": _z["pred"]}
        res.setdefault("subs", list(_z["subs"]))
        print(f"   [건너뜀] {_meth} — 이미 {_f} 에 있다", flush=True)
        continue
    _r = dl_model.benchmark_loo(PURE, items, calib_path=CALIB, baseline=True, trim=None,
                                methods=(_meth,), epochs=EPOCHS, seed=0, use_pretrain=True,
                                px_per_map=PXPM, n_trees=TREES, rf_max_features="sqrt",
                                cnn_epochs=CNN_EP,
                                progress=lambda s: print("   ", s, flush=True))
    res[_meth] = _r[_meth]
    res.setdefault("subs", _r["subs"])
    np.savez(_f, true=np.asarray(_r[_meth]["true"], float),
             pred=np.asarray(_r[_meth]["pred"], float), subs=np.array(_r["subs"], object))
    print(f"   [저장] {_f}", flush=True)
subs = list(res["subs"])
nb = [s for s in subs if s in SUB]
idx = [subs.index(s) for s in nb]

def roc(T, P):
    """검출 ROC — 성분을 다 모아서(pooled) 존재/부재를 예측 분율로 점수 매긴다."""
    y = np.concatenate([[1 if T[i][j] > THR else 0 for i in range(len(T))] for j in idx])
    s = np.concatenate([[P[i][j] for i in range(len(P))] for j in idx])
    o = np.argsort(-s)
    y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    if not tp[-1] or not fp[-1]:
        return None, None, float("nan")
    tpr = np.concatenate([[0.0], tp / tp[-1]]); fpr = np.concatenate([[0.0], fp / fp[-1]])
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


metrics, rocrows, predrows = [], [], []
for key, label in METHODS:
    if key not in res:
        continue
    T = np.asarray(res[key]["true"], float); P = np.asarray(res[key]["pred"], float)
    err = float(np.mean([0.5 * np.abs(P[i][idx] - T[i][idx]).sum() for i in range(len(T))]))
    f, t, auc = roc(T, P)
    metrics.append([label, key, f"{100 * err:.3f}", f"{auc:.4f}", str(len(T))])
    if f is not None:
        for a, b in zip(f, t):
            rocrows.append([label, f"{a:.5f}", f"{b:.5f}"])
    for i in range(len(T)):
        predrows.append([label, str(i + 1)] + [f"{100 * T[i][j]:.3f}" for j in idx]
                        + [f"{100 * P[i][j]:.3f}" for j in idx])
    print(f"  {label:<14} 조성오차 {100 * err:5.1f}%   AUC {auc:.3f}", flush=True)


def write(name, head, body):
    with open(f"{OUT}/{name}", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(head); w.writerows(body)
    print(f"  ✓ {name}  {len(body)}행", flush=True)


write("panel_d_method_comparison.csv",
      ["method", "key", "composition_error_pct", "detection_auc", "n_conditions"], metrics)
write("panel_e_detection_roc.csv", ["method", "fpr", "tpr"], rocrows)
write("panel_de_predictions.csv",
      ["method", "condition"] + [f"true_{s}_pct" for s in nb] + [f"pred_{s}_pct" for s in nb],
      predrows)
import shutil
# 다 끝났으니 부분 저장은 치운다. `ignore_errors=True` 로 삼키면 안 된다 — 이 저장소는
# Google Drive 위에 있어서 동기화가 파일을 잡고 있으면 rmtree 가 조용히 실패하고,
# 남은 폴더를 "아직 도는 중" 으로 오해하게 된다 (2026-08-14 에 실제로 물렸다).
try:
    shutil.rmtree(CKPT)
except OSError as e:
    print(f"⚠ 부분 저장을 못 지웠다 — 손으로 치울 것: {CKPT}\n  ({e})", flush=True)
print(f"\n{time.time() - t0:.0f}s · → {OUT}")
print(f"표에 적을 것: leave-one-condition-out · base 구성(blank/sips 없음, "
      f"benchmark_loo 가 그 인자를 받지 않는다) · RF {TREES} trees, max_features=sqrt · "
      f"**CNN 은 {CNN_EP} epoch 으로 줄여 돌렸다** (다른 방법은 {EPOCHS}) — "
      f"CNN 이 불리한 쪽으로 치우친 비교다. 캡션에 반드시 적을 것.")
