"""프로젝트 자체 train_model로 저농도를 포함해 재학습. TBZ 오탐이 사라지나."""
import sys, os, re, glob
import numpy as np

REPO = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/github/Mixture Classifier"
sys.path.insert(0, REPO)
import dl_model
from real_data import load_map

DB = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB"
PURE = f"{DB}/Pure"
BAD = {"DQ500TBZ100", "DQ1000TBZ100"}
CALIB = f"{DB}/Ratio/results/calibration_spectra.csv"
SUB = ["DQ", "TBZ", "THI"]
COMP = {1: (6, 6, 24), 2: (12, 12, 12), 3: (24, 6, 6),
        4: (6, 6, 6), 5: (24, 24, 24), 6: (12, 12, 48)}


def parse(nm):
    c = {"DQ": 0.0, "TBZ": 0.0, "THI": 0.0}
    for k, v in re.findall(r"(DQ|TBZ|TH[I1])(\d+)", nm):
        c["THI" if k.startswith("TH") else k] += float(v)
    return c


HI = []
for p in sorted(glob.glob(f"{DB}/Ratio/Ratio_mix/*orrected.csv")):
    nm = os.path.basename(p).split("_corrected")[0].split("-corrected")[0]
    if nm in BAD:
        continue
    c = parse(nm)
    if sum(c.values()) == 0:
        continue
    HI.append((p, c, {k: v * 1e-6 for k, v in c.items()}))

LO = {}
for n, C in COMP.items():
    d = dict(zip(SUB, map(float, C)))
    LO[n] = [(f"{DB}/tert-new-baseline/{n}-{r}_corrected.csv", d,
              {k: v * 1e-6 for k, v in d.items()}) for r in (1, 2, 3)
             if os.path.exists(f"{DB}/tert-new-baseline/{n}-{r}_corrected.csv")]

print(f"고농도 {len(HI)}맵 · 저농도 {sum(len(v) for v in LO.values())}맵 ({len(LO)}조건)")

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
MODE = sys.argv[2] if len(sys.argv) > 2 else "both"     # both | lo


def evaluate(train_items, test_cond):
    m = dl_model.train_model(PURE, train_items, calib_path=CALIB, baseline=True, trim=None,
                             method="mlp", epochs=EPOCHS, seed=0,
                             use_pretrain=True, nnls_screen=True, px_per_map=20)
    out, uMs = [], []
    for p, ratio, _c in LO[test_cond]:
        wn, cube, _mm, _co = load_map(p)
        r = dl_model.apply_model(m, np.asarray(wn, float), np.asarray(cube, float))
        out.append(np.array([r["composition"].get(s, 0.0) for s in SUB]))
        if r.get("uM"):
            uMs.append(np.array([r["uM"].get(s, np.nan) for s in SUB]) * 1e6)
    return np.array(out), (np.array(uMs) if uMs else None)


print(f"\nLOO (저농도 조건 하나씩 빼고 학습) · epochs={EPOCHS} · mode={MODE}")
print(f"{'조건':>14} | {'참 %':>18} | {'예측 % (3반복 평균)':>22} | {'오차':>6} | 우세")
errs = []
RECS = []
for n in sorted(COMP):
    tr = ([] if MODE == "lo" else list(HI)) + \
         [it for k, v in LO.items() if k != n for it in v]
    try:
        Pd, Ud = evaluate(tr, n)
    except Exception as e:
        print(f"  #{n}: 실패 {type(e).__name__}: {e}")
        continue
    C = np.array(COMP[n], float); t = C / C.sum() * 100
    p = Pd.mean(axis=0) * 100
    e = np.abs(p - t).sum() / 2
    errs.append(e)
    line = (f"  #{n} {'/'.join(str(int(v)) for v in C):>9} | " + "/".join(f"{v:5.1f}" for v in t) +
            " | " + "/".join(f"{v:6.1f}" for v in p) + f" | {e:5.1f}%")
    if Ud is not None:
        u = Ud.mean(axis=0); rec = u / C * 100
        RECS.append(rec)
        line += " | uM " + "/".join(f"{v:6.1f}" for v in u) + " | 회수 " + "/".join(f"{v:4.0f}" for v in rec) + "%"
    print(line)
if errs:
    print(f"\n  저농도 조성 오차 평균 {np.mean(errs):5.1f}%")
    print(f"  [비교] 기존 고농도 모델 38–49% · NNLS 22.6% · 반복산포 하한 9.6%")
if RECS:
    R = np.array(RECS)
    print()
    for j, s_ in enumerate(SUB):
        v = R[:, j]
        print(f"  {s_:>4} 회수율 중앙 {np.median(v):5.0f}%   2배 이내 {100*np.mean((v>=50)&(v<=200)):3.0f}%   3배 이내 {100*np.mean((v>=33)&(v<=300)):3.0f}%")
    v = R.ravel()
    print(f"  전체       2배 이내 {100*np.mean((v>=50)&(v<=200)):3.0f}%   3배 이내 {100*np.mean((v>=33)&(v<=300)):3.0f}%")
