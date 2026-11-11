"""농도 개선 탐색용 캐시 (2026-11-11). 라벨 있는 혼합물 맵 전부(격자 64 + 격자 밖)에 대해
모델과 같은 전처리(dl_model._map_spectra: 포화 보정 → 대표 100 픽셀 → ALS)로 픽셀 스펙트럼을 꺼내 둔다.
ALS 를 뺀 같은 픽셀도 같이 저장한다.

  실행:  .venv/bin/python3 documentation/results/conc_improve_2026-11-11/build_cache.py
"""
import os, sys, warnings, numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(f"{HERE}/../../..")
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/documentation/scripts")
sys.argv = sys.argv[:1]
import dl_model, train_paper_pls as tp
from dataset import load_mixture_list
from real_data import load_map
from sers_mixture import desaturate

m = dl_model.load_model(tp.OUT); u = m["uM"]; S3 = list(u["subs"]); subs_all = list(m["subs"])
print("subs", S3, "ink", u["ink_ref_cm"], "kind", u.get("kind"))
PA, PR, TT, RR, MM, paths, mt, npx, nsat, AX = [], [], [], [], [], [], [], [], [], None
for it in load_mixture_list(tp.PURE, filename_truth=True):
    p = tp._here(it)[0]; c = tp.conc_of(p)
    if c is None:
        continue
    wn, cube, _a, _b = load_map(p); wn = np.asarray(wn, float)
    mask = (wn >= m["lo"]) & (wn <= m["hi"]); ax = wn[mask]
    c2 = np.asarray(cube, float).reshape(-1, len(wn))
    kw = dict(sampling=m["pixel_sampling"], sampling_seed=int(m["sampling_seed"]), map_id=p)
    Xa = np.array(dl_model._map_spectra(c2, mask, int(m["px_per_map"]), baseline_correct=True, **kw))
    Xr = np.array(dl_model._map_spectra(c2, mask, int(m["px_per_map"]), baseline_correct=False, **kw))
    if AX is None:
        AX = ax
    if len(ax) != len(AX) or np.abs(ax - AX).max() > 1e-6:
        print("skip (axis)", os.path.basename(p), len(ax)); continue
    pk = np.clip(np.asarray(dl_model.apply_model_pixels(m, ax, Xa), float), 0, None)
    cols = [subs_all.index(s) for s in S3]; mass = pk[:, cols].sum(1); R = pk[:, cols] / (mass[:, None] + 1e-12)
    _d, sat = desaturate(c2[:, mask])
    n0 = int(m["px_per_map"]) - len(Xa)          # 잘린 픽셀이 빠져 100 개가 안 되는 맵: 빈 줄로 채우고 질량 −1 로 표시
    if n0 > 0:
        Xa = np.vstack([Xa, np.zeros((n0, Xa.shape[1]))]); Xr = np.vstack([Xr, np.zeros((n0, Xr.shape[1]))])
        R = np.vstack([R, np.zeros((n0, R.shape[1]))]); mass = np.r_[mass, -np.ones(n0)]
    PA.append(Xa.astype(np.float32)); PR.append(Xr.astype(np.float32)); TT.append(np.asarray(c, float))
    RR.append(R); MM.append(mass); paths.append(p); mt.append(os.path.getmtime(p)); npx.append(len(c2)); nsat.append(int((sat > 0.01).sum()))
    print(len(TT), os.path.basename(p), c, len(c2), nsat[-1], flush=True)
np.savez_compressed(f"{HERE}/cache.npz", PA=np.array(PA), PR=np.array(PR), T=np.array(TT), R=np.array(RR), M=np.array(MM),
                    paths=np.array(paths), mtime=np.array(mt), npx=np.array(npx), nsat=np.array(nsat), ax=AX,
                    bands=np.asarray(u["bands_cm"], float), ink=float(u["ink_ref_cm"]))
print("saved", len(TT))
