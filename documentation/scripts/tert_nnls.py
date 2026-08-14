"""등몰 삼원 계열의 NNLS 표면 조성 — `tert260805_nnls.npz` 를 만든다.

  이 npz 는 `export_tert.py`(패널 c) · `export_origin.py` · `tert260805_fig.py` 가
  읽는데, **만드는 스크립트가 저장소에 없었다.** 2026-08-14 재실행에서 `export_tert` 가
  `FileNotFoundError` 로 죽어서 드러났다 — 메인 클론에 파일만 남아 있고 출처가 없었다.
  여기서 다시 만들 수 있게 해 둔다.

  판독 경로는 `export_nnls.py` 와 **같다** (`surface_composition`, 맵당 평균 스펙트럼
  한 줄). 그래야 패널 a·c 의 NNLS 가 같은 것을 뜻한다.

  참 조성은 네 농도 모두 33.3/33.3/33.3 이다. 농도가 오르면 THI 가 표면을 덮는다 —
  그 관측이 작업구간을 3–24 µM 로 한정하는 근거다 (FINDINGS §4).

  나오는 것:  documentation/scripts/tert260805_nnls.npz
                per   (n,)   성분당 최종 µM — 10 · 33 · 100 · 333
                comp  (n,3)  NNLS 표면 조성 %, 반복 평균, DQ/TBZ/THI 순

  실행:  python3 -u tert_nnls.py
"""
import os as _os, sys as _sys                 # paths 부트스트랩 — 기계마다 마운트가 다르다
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import os, sys, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, paths.REPO)

import mixtures as MX
from dl_model import _refs, _map_spectra, _composition_features
from dl_quantify import surface_composition
from real_data import load_map

SUB = MX.SUB
OUT = f"{HERE}/tert260805_nnls.npz"

subs, wn, mask, P, lo, hi = _refs(paths.PURE, True, None)
idx = [subs.index(s) for s in SUB]

# 등몰 계열 8맵 = 4농도 × 2조제. 라벨은 최종 폴더 파일명 (`mixtures.py`).
#
# 다만 눈금은 **원액÷3 의 정확한 값**으로 적는다. 파일명은 반올림돼 있어서
# (`DQ33-TB33-TH33` 의 실제값은 33.33) 그대로 쓰면 `tert260805.py` 가 내는 MLP 쪽
# 눈금(33.33·333.3)과 `export_tert.py` 에서 안 묶이고 패널 c 가 줄이 어긋난다.
EXACT = [30 / 3, 100 / 3, 300 / 3, 1000 / 3]      # 원액 0.03·0.1·0.3·1 mM 을 1:1:1 혼합


def _snap(v):
    return min(EXACT, key=lambda e: abs(e - v))


maps = MX.catalogue(("equimolar",), replicates=True)
by_level = collections.OrderedDict()
for m in sorted(maps, key=lambda m: m.conc[0]):
    by_level.setdefault(_snap(float(m.conc[0])), []).append(m.path)

print(f"등몰 계열 {len(maps)}맵 · {len(by_level)}농도 · 템플릿 {subs}")

per, comp = [], []
for lv, ps in by_level.items():
    v = []
    for p in ps:
        _w, cube, _m, _c = load_map(p)
        # px_per_map=0 → 맵당 평균 스펙트럼 한 줄. export_nnls.py 와 같은 단위다.
        X = np.array([_composition_features(ya) for ya in _map_spectra(cube, mask, 0)])
        raw = np.expm1(np.clip(X, 0, None))
        v.append(surface_composition(_composition_features(raw, "legacy_l2"), P).mean(0))
    a = np.mean(v, axis=0)[idx]
    a = a / (a.sum() or 1.0) * 100
    per.append(lv); comp.append(a)
    print(f"  {lv:6.4g} µM/성분  ({len(ps)}맵)  " + "/".join(f"{x:5.1f}" for x in a))

np.savez(OUT, per=np.array(per, float), comp=np.array(comp, float))
print(f"\n저장: {os.path.basename(OUT)}")
print("참 조성은 모든 농도에서 33.3/33.3/33.3 — 고농도에서 THI 로 쏠리는 것이 관측 사실이다.")
