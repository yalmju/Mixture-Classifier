"""채택 구성 Physics-MLP 를 **한 번 학습해 저장**한다 — 매번 다시 돌리지 않게.

  4-fold 검증(`retrain_all.py` · `piemap_compute.py`)은 fold 마다 새로 학습하므로
  held-out 숫자를 내는 데 쓰고, **여기서 나오는 모델은 전 조건으로 학습한 배포용**이다.
  둘을 헷갈리지 말 것 — 이 모델로 학습셋 안의 맵을 예측하면 held-out 이 아니다.

  채택 구성 (`FINDINGS_2026-08-12.md` §1):
      method="mlp" · use_pretrain=True (물리 사전학습) · include_blank=True
      sim_iso="sips_dq" · sim_nuisance=False · nnls_screen=True · px_per_map=20

  라벨은 `260814_mixture_final` 파일명 = 최종 µM (`mixtures.py`). 학습셋은 고농도 34조건
  + 저농도 65조건(반복 포함 80맵) = 114맵. **등몰 계열 8맵은 뺀다** — 조성에 희석 정보가
  0 인 판별 장치라 적용 대상으로 남겨 둔다 (`TODO.md` §6).

  물리 사전학습이 꺼진 채 도는 것을 막으려고 `paths.check_pretrain` 을 먼저 부른다.
  검량 파일이 없으면 여기서 죽는다 — `train_model` 은 조용히 건너뛰기 때문이다.

  나오는 것:  documentation/scripts/model_physics_mlp.pkl   (+ .json 메타)

  실행:  python3 -u train_physics_mlp.py [epochs=120]
"""
import os as _os, sys as _sys                 # paths 부트스트랩 — 기계마다 마운트가 다르다
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import os, sys, json, pickle, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, paths.REPO)

import mixtures as MX
import dl_model

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
OUT = f"{HERE}/model_physics_mlp.pkl"
META = f"{HERE}/model_physics_mlp.json"

CALIB = paths.calibration()
paths.check_pretrain(CALIB, True)          # 사전학습이 실제로 켜지는지 먼저 확인한다

items = MX.items(("high", "grid"), replicates=True)
print(f"학습셋 {len(items)}맵 · epochs={EPOCHS}", flush=True)

t0 = time.time()
model = dl_model.train_model(
    paths.PURE, items, calib_path=CALIB, baseline=True, trim=None, method="mlp",
    epochs=EPOCHS, seed=0, use_pretrain=True, nnls_screen=True, px_per_map=20,
    include_blank=True, sim_nuisance=False, sim_iso="sips_dq",
    progress=lambda m: print("   ", m, flush=True) if "epoch" not in m else None)
dt = time.time() - t0

with open(OUT, "wb") as f:
    pickle.dump(model, f)

meta = {
    "created": time.strftime("%Y-%m-%d %H:%M"),
    "labels": "Ratio/260814_mixture_final (파일명 = 최종 µM)",
    "sets": ["high", "grid"], "replicates": True,
    "n_maps": len(items),
    "n_conditions": len({tuple(sorted(c.items())) for _p, c, _m in items}),
    "arm": "blank,sips — include_blank=True, sim_iso=sips_dq, sim_nuisance=False",
    "use_pretrain": True,
    "calibration": os.path.relpath(CALIB, paths.DB),
    "epochs": EPOCHS, "seed": 0, "px_per_map": 20, "nnls_screen": True,
    "train_seconds": round(dt),
    "note": "전 조건 학습 — held-out 이 아니다. 검증 숫자는 retrain_all/piemap_compute 를 쓸 것.",
}
with open(META, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)

print(f"\n  ✓ {os.path.basename(OUT)}   {os.path.getsize(OUT) / 1e6:.1f} MB   ({dt:.0f}s)")
print(f"  ✓ {os.path.basename(META)}")
print("\n쓰는 법:")
print("    import pickle; model = pickle.load(open('model_physics_mlp.pkl','rb'))")
print("    r = dl_model.apply_model(model, wn, cube)   # r['composition'], r['uM']")
print("⚠ 전 조건으로 학습했다 — 학습셋 안의 맵에 적용한 숫자를 성능으로 보고하지 말 것.")
