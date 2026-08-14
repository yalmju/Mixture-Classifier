"""260805 삼원 등몰 농도 시리즈 — 농도가 바뀌어도 1:1:1 비율이 복원되는가.

  데이터  Ratio/260805/tert-1, tert-2
          tert_{003,01,03,1}mM_{1,2}_corrected.csv  = 원액 0.03 / 0.1 / 0.3 / 1 mM
          등몰 삼원이므로 참 조성은 언제나 33.3 / 33.3 / 33.3
          원액 3배 관례(성분당 최종 = 원액/3)를 따르면 성분당 10 / 33 / 100 / 333 µM

  이 시리즈는 **학습 구간(3–24 µM/성분) 바깥**이다 — 외삽 검증을 겸한다.
  문서에 따르면 111 µM/성분 위에서 DQ·TBZ가 표면에서 배제되고 THI만 남는다.
  그 전이가 이 네 점에서 보이는지가 핵심.

  학습셋에 이 8맵은 들어가지 않는다 (Ratio_mix · Baseline260808 · tert-new-baseline만).

  실행:  python3 -u tert260805.py [epochs=120]
"""
import os as _os, sys as _sys                 # paths 부트스트랩 — 기계마다 마운트가 다르다
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import sys, os, re, glob, pickle
import numpy as np

REPO = paths.REPO
DB = paths.DB
sys.path.insert(0, REPO)

import dl_model
from real_data import load_map

import mixtures as MX

PURE = paths.PURE
CALIB = paths.calibration()
SUB = MX.SUB
MODEL = f"{REPO}/documentation/scripts/model_full35.pkl"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 120

# 원액(등량 3성분 혼합 전) → 성분당 최종 µM. 폴더가 `Ratio/260805` 에서
# `Ratio/mis/tert-{1,2}` 로 옮겨졌다 — 예전 SERIES 경로는 0맵을 조용히 돌려줬다.
STOCK_UM = {"003mM": 30.0, "01mM": 100.0, "03mM": 300.0, "1mM": 1000.0}


# ---------------------------------------------------------------- 학습셋
# 라벨은 `260814_mixture_final` 파일명 = 최종 µM (`mixtures.py`). 등몰 계열 8맵은
# 여기 안 들어간다 — 적용 대상이다.
def build_items():
    HI = MX.items(("high",))
    LO = MX.groups(("grid",), replicates=True)
    return HI + MX.items_for(LO, sorted(LO)), len(HI), len(LO)


if os.path.exists(MODEL):
    model = pickle.load(open(MODEL, "rb"))
    print("저장된 모델 사용:", os.path.basename(MODEL))
else:
    items, n_hi, n_lo = build_items()
    print(f"학습: 고농도 {n_hi}맵 + 저농도 {n_lo}조건  총 {len(items)}맵 · epochs={EPOCHS}")
    model = dl_model.train_model(
        PURE, items, calib_path=CALIB, baseline=True, trim=None, method="mlp",
        epochs=EPOCHS, seed=0, use_pretrain=True, nnls_screen=True, px_per_map=20,
        progress=lambda m: print("   ", m, flush=True) if "epoch" not in m else None)
    pickle.dump(model, open(MODEL, "wb"))
    print("모델 저장:", os.path.basename(MODEL))

# ---------------------------------------------------------------- 적용
maps = sorted(glob.glob(f"{DB}/Ratio/mis/tert-*/tert_*_corrected.csv"))
if not maps:
    raise FileNotFoundError(f"등몰 계열을 못 찾았다: {DB}/Ratio/mis/tert-*/")
print(f"\n적용 대상 {len(maps)}맵 — 참 조성은 모두 33.3/33.3/33.3\n")
print(f"{'맵':>22} | {'원액':>7} | {'성분당 µM':>9} | {'예측 DQ/TBZ/THI %':>22} | {'오차':>6} | 우세")
print("-" * 92)
rows = []
for p in maps:
    m = re.search(r"tert_(\w+?)_(\d)_corrected", os.path.basename(p))
    tag, rep = m.group(1), int(m.group(2))
    stock = STOCK_UM[tag]; per = stock / 3.0
    wn, cube, _m, _c = load_map(p)
    r = dl_model.apply_model(model, np.asarray(wn, float), np.asarray(cube, float))
    pv = np.array([r["composition"].get(s, 0.0) for s in SUB]); pv = pv / pv.sum() * 100
    t = np.array([100 / 3] * 3)
    e = np.abs(pv - t).sum() / 2
    rows.append((tag, rep, stock, per, pv, e))
    print(f"{os.path.basename(p)[:22]:>22} | {stock:6.0f}µ | {per:8.1f} | " +
          "/".join(f"{v:6.1f}" for v in pv) + f" | {e:5.1f}% | {SUB[int(np.argmax(pv))]}")

print()
print(f"{'성분당 µM':>10} | {'n':>2} | {'평균 DQ/TBZ/THI %':>22} | {'반복차':>7} | {'오차':>6}")
print("-" * 66)
for tag in ["003mM", "01mM", "03mM", "1mM"]:
    g = [r for r in rows if r[0] == tag]
    if not g:
        continue
    A = np.array([r[4] for r in g])
    mu = A.mean(0); spread = np.abs(A[0] - A[1]).sum() / 2 if len(g) > 1 else float("nan")
    e = np.abs(mu - 100 / 3).sum() / 2
    print(f"{g[0][3]:9.1f} | {len(g):2d} | " + "/".join(f"{v:6.1f}" for v in mu) +
          f" | {spread:6.1f}% | {e:5.1f}%")

np.savez(f"{REPO}/documentation/scripts/tert260805_result.npz",
         tags=np.array([r[0] for r in rows]), reps=np.array([r[1] for r in rows]),
         per_uM=np.array([r[3] for r in rows]), pred=np.array([r[4] for r in rows]),
         err=np.array([r[5] for r in rows]))
print("\n저장: tert260805_result.npz")
print("  학습 구간은 성분당 3–24 µM. 10 µM만 그 안이고 33·100·333 µM은 외삽이다.")
