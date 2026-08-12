"""저농도 38조건(Baseline260808 32 + tert-new 6)으로 재학습. 4-fold 조건단위 검증."""
import sys, os, re, glob
import numpy as np

REPO = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/github/Mixture Classifier"
sys.path.insert(0, REPO)
import dl_model
from real_data import load_map

DB = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB"
PURE = f"{DB}/Pure"
import paths
CALIB = paths.calibration()          # 하드코딩하면 조용히 사전학습이 꺼진다
BAD = {"DQ500TBZ100", "DQ1000TBZ100"}
SUB = ["DQ", "TBZ", "THI"]


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

# ---- 저농도: 조성(최종 µM)을 키로 묶는다. 같은 조성의 다른 파일은 반복으로 취급 ----
LO = {}
for p in sorted(glob.glob(f"{DB}/Ratio/Baseline260808/DQ*/DQ*_corrected.csv")):
    m = re.match(r"DQ(\d+)-TB(\d+)-TH(\d+)", os.path.basename(p))
    key = tuple(int(m.group(i)) // 3 for i in (1, 2, 3))
    LO.setdefault(key, []).append(p)
OLD = {(6, 6, 24): 1, (12, 12, 12): 2, (24, 6, 6): 3, (6, 6, 6): 4, (24, 24, 24): 5, (12, 12, 48): 6}
for key, n in OLD.items():
    for r in (1, 2, 3):
        p = f"{DB}/tert-new-baseline/{n}-{r}_corrected.csv"
        if os.path.exists(p):
            LO.setdefault(key, []).append(p)

keys = sorted(LO)
print(f"고농도 {len(HI)}맵 · 저농도 {len(keys)}조건 {sum(len(v) for v in LO.values())}맵")


def items_for(ks):
    out = []
    for k in ks:
        d = dict(zip(SUB, map(float, k)))
        for p in LO[k]:
            out.append((p, d, {s: v * 1e-6 for s, v in d.items()}))
    return out


EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
FOLDS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
rng = np.random.default_rng(0)
order = rng.permutation(len(keys))
folds = [[keys[i] for i in order[f::FOLDS]] for f in range(FOLDS)]

print(f"\n{FOLDS}-fold 조건단위 검증 · epochs={EPOCHS}")
print(f"{'조성 (최종 µM)':>16} | {'참 %':>18} | {'예측 %':>18} | {'오차':>6} | 우세")
allerr, nnls_err = [], []
for fi, te in enumerate(folds):
    tr = [k for k in keys if k not in te]
    m = dl_model.train_model(PURE, HI + items_for(tr), calib_path=CALIB, baseline=True,
                             trim=None, method="mlp", epochs=EPOCHS, seed=0,
                             use_pretrain=True, nnls_screen=True, px_per_map=20)
    for k in te:
        C = np.array(k, float); t = C / C.sum() * 100
        ps = []
        for p in LO[k]:
            wn, cube, _mm, _co = load_map(p)
            r = dl_model.apply_model(m, np.asarray(wn, float), np.asarray(cube, float))
            ps.append([r["composition"].get(s, 0.0) for s in SUB])
        pm = np.mean(ps, axis=0); pm = pm / pm.sum() * 100
        e = np.abs(pm - t).sum() / 2
        allerr.append(e)
        tie = t.max() - t.min() < 1
        ok = "—" if tie else ("✅" if int(np.argmax(pm)) == int(np.argmax(t)) else "❌")
        print(f"  {'/'.join(f'{v:.0f}' for v in C):>16} | " + "/".join(f"{v:5.1f}" for v in t) +
              " | " + "/".join(f"{v:5.1f}" for v in pm) + f" | {e:5.1f}% | {ok}")
print(f"\n  저농도 {len(allerr)}조건 조성 오차  평균 {np.mean(allerr):5.1f}%  중앙 {np.median(allerr):5.1f}%")
print(f"  [비교] NNLS 27.6% · 6조건 학습시 18.8% · 반복산포 하한 9.6%")
