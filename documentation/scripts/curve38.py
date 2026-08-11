"""학습곡선 — 저농도 조건 수를 늘리면 조성 오차가 어디까지 내려가는가.

  질문: 35조건에서 12.9%다. 곡선이 아직 내려가는 중인가, 평평해졌는가?
        내려가는 중이면 남은 32조건을 찍을 값어치가 있다.

  방법: 맵 특징을 **한 번만** 만들어 캐시하고, 학습셋에 넣는 저농도 조건 수 n만 바꿔
        같은 시험 fold로 평가한다. 고농도 32맵은 항상 학습에 포함(retrain38과 동일).
        fold 분할도 retrain38과 같은 난수열을 쓴다 — 숫자가 이어붙는다.

  실행:  python3 -u curve38.py [epochs=120] [seeds=2]
"""
import sys, os, re, glob, time
import numpy as np

REPO = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/github/Mixture Classifier"
DB = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB"
sys.path.insert(0, REPO)

import dl_model
from dl_model import (_refs, _map_spectra, _composition_features, _fit_predict,
                      nnls_hit_spectra)
from dl_quantify import _ratio, simulate_mixtures

PURE = f"{DB}/Pure"
CALIB = f"{DB}/Ratio/results/calibration_spectra.csv"
BAD = {"DQ500TBZ100", "DQ1000TBZ100"}
SUB = ["DQ", "TBZ", "THI"]
CACHE = f"{REPO}/documentation/scripts/curve38_feats.npz"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 2

subs, wn, mask, P, lo, hi = _refs(PURE, baseline=True, trim=None)


def _hi_items():
    out = []
    for p in sorted(glob.glob(f"{DB}/Ratio/Ratio_mix/*orrected.csv")):
        nm = os.path.basename(p).split("_corrected")[0].split("-corrected")[0]
        if nm in BAD:
            continue
        c = {"DQ": 0.0, "TBZ": 0.0, "THI": 0.0}
        for k, v in re.findall(r"(DQ|TBZ|TH[I1])(\d+)", nm):
            c["THI" if k.startswith("TH") else k] += float(v)
        if sum(c.values()):
            out.append((p, c))
    return out


def _lo_groups():
    g = {}
    for p in sorted(glob.glob(f"{DB}/Ratio/Baseline260808/DQ*/DQ*_corrected.csv")):
        m = re.match(r"DQ(\d+)-TB(\d+)-TH(\d+)", os.path.basename(p))
        if m:
            g.setdefault(tuple(int(m.group(i)) // 3 for i in (1, 2, 3)), []).append(p)
    OLD = {(6, 6, 24): 1, (12, 12, 12): 2, (24, 6, 6): 3,
           (6, 6, 6): 4, (24, 24, 24): 5, (12, 12, 48): 6}
    for key, n in OLD.items():
        for r in (1, 2, 3):
            p = f"{DB}/tert-new-baseline/{n}-{r}_corrected.csv"
            if os.path.exists(p):
                g.setdefault(key, []).append(p)
    return g


# ------------------------------------------------------------------ 특징 캐시
if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    X, Y, KEY = z["X"], z["Y"], z["KEY"]
    print(f"캐시 사용: X {X.shape}")
else:
    HI = _hi_items()
    LOG = _lo_groups()
    items = [(p, c, "HI") for p, c in HI]
    for k in sorted(LOG):
        for p in LOG[k]:
            items.append((p, dict(zip(SUB, map(float, k))), str(k)))
    X, Y, KEY = [], [], []
    t0 = time.time()
    for i, (p, ratio, key) in enumerate(items, 1):
        vec = _ratio([float(ratio.get(s, 0.0)) for s in subs])
        if vec.sum() <= 0:
            continue
        _w, cube, _s = nnls_hit_spectra(PURE, p, baseline=True, trim=None,
                                        min_frac=0.15, progress=None)
        lm = np.ones(cube.shape[1], bool)
        for ya in _map_spectra(cube, lm, 20, baseline_correct=False):
            X.append(_composition_features(ya)); Y.append(vec); KEY.append(key)
        if i % 10 == 0 or i == len(items):
            print(f"  특징 {i}/{len(items)}  ({time.time()-t0:.0f}s)", flush=True)
    X = np.array(X, np.float32); Y = np.array(Y, np.float32); KEY = np.array(KEY, object)
    np.savez(CACHE, X=X, Y=Y, KEY=KEY.astype(str))
    print(f"캐시 저장: X {X.shape}  ({time.time()-t0:.0f}s)")

KEY = KEY.astype(str)
IS_HI = KEY == "HI"
lo_keys = [k for k in dict.fromkeys(KEY[~IS_HI].tolist())]
print(f"고농도 행 {IS_HI.sum()} · 저농도 조건 {len(lo_keys)} 행 {(~IS_HI).sum()}")

# ------------------------------------------------------------------ 물리 사전학습
pre = None
try:
    from io_utils import load_calibration_csv
    from calibration import calibrate
    ax_c, nc, dils = load_calibration_csv(CALIB); mc = (ax_c >= lo) & (ax_c <= hi)
    dil = [(dils[nc.index(s)][0], np.asarray(dils[nc.index(s)][1])[:, mc]) for s in subs if s in nc]
    if len(dil) == len(subs):
        cal = calibrate(dil, P, subs); rng = np.random.default_rng(0)
        Xs, Cs = simulate_mixtures(P, cal.K, cal.gA, 5000, rng, noise=0.015,
                                   baseline=0.03, gain_lo=0.8, gain_hi=1.25)
        pre = (_composition_features(Xs).astype(np.float32),
               np.array([_ratio(c) for c in Cs]).astype(np.float32))
        print("물리 사전학습 준비됨 (5000)")
except Exception as e:
    print("사전학습 실패:", e)

# ------------------------------------------------------------------ fold (retrain38과 동일)
FOLDS = 4
rng = np.random.default_rng(0)
order = rng.permutation(len(lo_keys))
folds = [[lo_keys[i] for i in order[f::FOLDS]] for f in range(FOLDS)]


def err_for(tr_keys, te_keys, seed):
    tr = IS_HI | np.isin(KEY, list(tr_keys))
    te = np.isin(KEY, list(te_keys))
    pred = _fit_predict("mlp", X[tr], Y[tr], X[te], pre=pre, epochs=EPOCHS,
                        seed=seed, P_ref=P)
    pred = np.asarray(pred, float)
    kte = KEY[te]
    es = []
    for k in te_keys:
        s = kte == k
        if not s.any():
            continue
        pm = pred[s].mean(0); pm = pm / pm.sum()
        tv = Y[te][s][0]
        es.append(float(np.abs(pm - tv).sum() / 2 * 100))
    return es


SIZES = [0, 4, 8, 13, 18, 26]
print(f"\n학습곡선 · epochs={EPOCHS} · fold {FOLDS} · seeds {SEEDS}")
print(f"{'저농도 조건 수':>12} | {'평균 오차':>9} | {'표준편차':>8} | {'n 평가':>6} | {'소요':>6}")
print("-" * 58)
res = {}
for n in SIZES:
    t0 = time.time(); vals = []
    for fi, te_keys in enumerate(folds):
        pool = [k for k in lo_keys if k not in te_keys]
        n_use = min(n, len(pool))
        reps = 1 if n_use in (0, len(pool)) else SEEDS
        for s in range(reps):
            sel = (list(np.random.default_rng(1000 * s + fi).permutation(len(pool))[:n_use]))
            vals += err_for([pool[i] for i in sel], te_keys, seed=fi)
    res[n] = vals
    print(f"{n:>12} | {np.mean(vals):8.1f}% | {np.std(vals):7.1f}% | {len(vals):6d} | {time.time()-t0:5.0f}s",
          flush=True)

np.savez(f"{REPO}/documentation/scripts/curve38_result.npz",
         sizes=np.array(SIZES),
         means=np.array([np.mean(res[n]) for n in SIZES]),
         sds=np.array([np.std(res[n]) for n in SIZES]),
         **{f"v{n}": np.array(res[n]) for n in SIZES})
print("\n저장: curve38_result.npz")
print("  [비교] NNLS 27.6% · 6조건 18.8% · 35조건 12.9% · 반복산포 하한 9.6%")
