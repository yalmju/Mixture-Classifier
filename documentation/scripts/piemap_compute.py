"""파이 + 픽셀맵 재생성 — 1단계: 계산 (오래 걸림, 결과를 npz 로 캐시).

  데이터   Ratio/Baseline260808  DQ 9/18/36/72 × TB × TH  (파일명은 원액, 최종은 ÷3)
           + tert-new-baseline 6조건.  DQ9-sus 제외, md5 중복 제외.  → 65조건 80맵
  구성     채택 구성 그대로 — include_blank=True, sim_iso="sips_dq", 잉크 주입 없음
  검증     retrain64.py 와 **같은 seed·같은 fold** 의 조건단위 4-fold.
           held-out 조건에만 예측을 쓴다 (학습에서 본 조건의 파이를 그리면 그림이 거짓말한다)

  나오는 것 (piemap_data.npz)
    조건별  map-level 조성  = dl_model.apply_model  (retrain64 의 파이와 같은 경로)
    맵별    per-pixel 조성  = unmix_map(method="dlpx")  의 ratio_nb + hit + coords
                             (앱 Real data 탭의 픽셀 파이맵과 같은 경로)

  실행:  python3 -u piemap_compute.py [epochs=120] [folds=4]
"""
import sys, os, re, glob, hashlib, pickle, time
import numpy as np

REPO = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/github/Mixture Classifier"
sys.path.insert(0, REPO)
import dl_model
from unmix import unmix_map
from real_data import load_map

DB = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB"
PURE, CALIB = f"{DB}/Pure", f"{DB}/Ratio/results/calibration_spectra.csv"
OUT = f"{DB}/260808_data interpret/piemap_260812"
BAD = {"DQ500TBZ100", "DQ1000TBZ100"}
SUB = ["DQ", "TBZ", "THI"]

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
FOLDS = int(sys.argv[2]) if len(sys.argv) > 2 else 4


# ------------------------------------------------------------------ 데이터 수집
def parse_hi(nm):
    c = {"DQ": 0.0, "TBZ": 0.0, "THI": 0.0}
    for k, v in re.findall(r"(DQ|TBZ|TH[I1])(\d+)", nm):
        c["THI" if k.startswith("TH") else k] += float(v)
    return c


HI = []
for p in sorted(glob.glob(f"{DB}/Ratio/Ratio_mix/*orrected.csv")):
    nm = os.path.basename(p).split("_corrected")[0].split("-corrected")[0]
    if nm in BAD:
        continue
    c = parse_hi(nm)
    if sum(c.values()):
        HI.append((p, c, {k: v * 1e-6 for k, v in c.items()}))

LO, _seen = {}, set()


def _add(key, p):
    h = hashlib.md5(open(p, "rb").read()).hexdigest()
    if h in _seen:
        return
    _seen.add(h)
    LO.setdefault(key, []).append(p)


for p in sorted(glob.glob(f"{DB}/Ratio/Baseline260808/DQ*/DQ*_corrected.csv")):
    if "DQ9-sus" in p:
        continue
    m = re.match(r"DQ(\d+)-TB(\d+)-TH(\d+)", os.path.basename(p))
    _add(tuple(int(m.group(i)) // 3 for i in (1, 2, 3)), p)
OLD = {(6, 6, 24): 1, (12, 12, 12): 2, (24, 6, 6): 3, (6, 6, 6): 4, (24, 24, 24): 5, (12, 12, 48): 6}
for key, n in OLD.items():
    for r in (1, 2, 3):
        p = f"{DB}/tert-new-baseline/{n}-{r}_corrected.csv"
        if os.path.exists(p):
            _add(key, p)

keys = sorted(LO)
print(f"고농도 {len(HI)}맵 · 저농도 {len(keys)}조건 {sum(len(v) for v in LO.values())}맵", flush=True)


def items_for(ks):
    out = []
    for k in ks:
        d = dict(zip(SUB, map(float, k)))
        for p in LO[k]:
            out.append((p, d, {s: v * 1e-6 for s, v in d.items()}))
    return out


# retrain64.py 와 동일한 fold (seed 0)
rng = np.random.default_rng(0)
order = rng.permutation(len(keys))
folds = [[keys[i] for i in order[f::FOLDS]] for f in range(FOLDS)]

cond = {}      # key -> {"true","pred","paths"}
maps = {}      # path -> {"coords","ratio","hit","reliab","comps","mean_ratio","key"}

t0 = time.time()
for fi, te in enumerate(folds):
    tr = [k for k in keys if k not in te]
    print(f"\n=== fold {fi+1}/{FOLDS} — 학습 {len(tr)}조건 / 검증 {len(te)}조건 "
          f"({time.time()-t0:.0f}s)", flush=True)
    m = dl_model.train_model(PURE, HI + items_for(tr), calib_path=CALIB, baseline=True,
                             trim=None, method="mlp", epochs=EPOCHS, seed=0,
                             use_pretrain=True, nnls_screen=True, px_per_map=20,
                             include_blank=True, sim_nuisance=False, sim_iso="sips_dq",
                             progress=None)
    for k in te:
        C = np.array(k, float)
        t = C / C.sum() * 100
        ps = []
        for p in LO[k]:
            wn, cube, _mm, _co = load_map(p)
            r = dl_model.apply_model(m, np.asarray(wn, float), np.asarray(cube, float))
            ps.append([r["composition"].get(s, 0.0) for s in SUB])

            # 픽셀 단위 — 앱 Real data 탭과 같은 경로 (NNLS 가 hit 을 정하고 모델이 조성을 낸다)
            u = unmix_map(PURE, p, method="dlpx", baseline=True, dl_model=m, progress=None)
            nb = list(u.nonbg)
            maps[p] = {
                "key": k,
                "coords": np.asarray(u.coords, float),
                "ratio": np.asarray(u.ratio_nb, float),        # (n_px, n_nonbg)
                "hit": np.asarray(u.hit, bool),
                "reliab": np.asarray(u.reliab, float),
                "comps": [u.comps[i] for i in nb],
                "mean_ratio": np.asarray(u.mean_ratio, float),
                "hit_frac": float(u.hit_frac),
            }
        pm = np.mean(ps, axis=0)
        pm = pm / pm.sum() * 100
        cond[k] = {"true": t, "pred": pm, "paths": list(LO[k]),
                   "err": float(np.abs(pm - t).sum() / 2)}
        print(f"  {'/'.join(f'{v:.0f}' for v in C):>14} | "
              + "/".join(f"{v:5.1f}" for v in t) + " | "
              + "/".join(f"{v:5.1f}" for v in pm) + f" | {cond[k]['err']:5.1f}%", flush=True)

errs = [cond[k]["err"] for k in keys]
print(f"\n조성 오차 평균 {np.mean(errs):.1f}%  중앙 {np.median(errs):.1f}%  "
      f"({len(errs)}조건, {time.time()-t0:.0f}s)", flush=True)

with open(f"{OUT}/piemap_data.pkl", "wb") as f:
    pickle.dump({"cond": cond, "maps": maps, "sub": SUB, "keys": keys,
                 "epochs": EPOCHS, "folds": FOLDS,
                 "arm": "blank,sips (include_blank=True, sim_iso=sips_dq)"}, f)
print(f"saved {OUT}/piemap_data.pkl")
