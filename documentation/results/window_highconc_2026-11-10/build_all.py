"""260814_mixture_final 의 102맵 전부를 한 가지 방식으로 뽑는다 (NNLS hit → 대표 24픽셀 → ALS)."""
import sys, os, re, time, glob, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
import dl_model
DB = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB/0. data"
PURE = f"{DB}/Pure"; RAT = f"{DB}/Ratio/260814_mixture_final"
files = sorted(os.path.basename(p) for p in glob.glob(f"{RAT}/*.csv"))
subs, wn0, mask0, P, lo, hi = dl_model._refs(PURE, True, None)
print(subs, P.shape, len(files), flush=True)
X, G, C = [], [], []
for fn in files:
    m = re.match(r"DQ(\d+)-TB(\d+)-THI?(\d+)", fn)
    c = [float(m.group(i)) for i in (1, 2, 3)]
    t = time.time()
    try:
        wn, sp, meta = dl_model.nnls_hit_spectra(PURE, f"{RAT}/{fn}", baseline=True)
    except Exception as e:
        print("FAIL", fn, e, flush=True); continue
    out = dl_model._map_spectra(sp, np.ones(sp.shape[1], bool), n_px=24, sampling="representative",
                                sampling_seed=0, map_id=fn.lower())
    out = np.asarray(out, np.float32)
    X.append(out); G += [fn] * len(out); C += [c] * len(out)
    print(fn, sp.shape, len(out), f"{time.time()-t:.1f}s", flush=True)
np.savez("all102.npz", X=np.concatenate(X), g=np.array(G, object), C=np.array(C, float), wn=wn, P=P, subs=np.array(subs, object))
print("saved", sum(map(len, X)))
