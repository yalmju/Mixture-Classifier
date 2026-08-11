"""DQ 바닥값의 정체 — NNLS 기저에 BLK·INK 를 넣으면 사라지는가.

  현상   DQ 계수를 세 성분 농도로 회귀하면 절편이 중앙값의 104% 다.
         농도 8배(3→24 µM)에 신호는 1.7배. DQ 는 자기 농도를 거의 안 따라간다.

  가설   `dataset.py:29  BLANK_ALIASES = {..., "ink"}` 때문에 **INK 와 BLK 가
         템플릿에서 빠진다.** 잉크·기판 신호는 실제로 스펙트럼에 있는데 기저에
         없으니, NNLS 가 그 몫을 가장 닮은 분석물 템플릿(=DQ)에 얹는다.

  검증   기저를 3성분 → 5성분(DQ/TBZ/THI/BLK/INK)으로 바꿔 같은 회귀를 다시 한다.
         가설이 맞으면 DQ 절편이 내려가고 R² 가 올라간다.

  실행:  python3 -u dqfloor.py
"""
import sys, os, re, glob, json, hashlib
import numpy as np
from scipy.optimize import nnls

REPO = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/github/Mixture Classifier"
DB = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB"
sys.path.insert(0, REPO)

from unmix import _templates, _baseline_removed, _l2
from real_data import load_map

PURE = f"{DB}/Pure"
SUB = ["DQ", "TBZ", "THI"]
CACHE = f"{REPO}/documentation/scripts/dqfloor_signals.json"

names, wn, means = _templates(PURE, True, None)
lo, hi = float(np.min(wn)), float(np.max(wn))
mask = (wn >= lo) & (wn <= hi)
idx = {n: i for i, n in enumerate(names)}
FULL = [n for n in ["DQ", "TBZ", "THI", "BLK", "INK"] if n in idx]
P5 = _l2(_baseline_removed(means[[idx[n] for n in FULL]][:, mask], True))
P3 = P5[:3]
print(f"기저:  3성분 {SUB}   ·   5성분 {FULL}")

# 템플릿끼리 얼마나 닮았나 — 누화가 어디로 갈지 미리 본다
print("\n템플릿 코사인 (DQ 와)")
for k in range(1, len(FULL)):
    print(f"  DQ · {FULL[k]:<4} {float(P5[0] @ P5[k]):.3f}")


def sigs(path):
    wn_, cube, _m, _c = load_map(path)
    wn_ = np.asarray(wn_, float); cube = np.asarray(cube, float)
    m = (wn_ >= lo) & (wn_ <= hi)
    Y = np.clip(cube[:, m], 0.0, None)
    a3 = np.median(np.array([nnls(P3.T, y)[0] for y in Y]), axis=0)
    a5 = np.median(np.array([nnls(P5.T, y)[0] for y in Y]), axis=0)
    return list(a3) + list(a5)


# ---- 저농도 맵 (retrain64 와 같은 규칙) ----
LO, seen = {}, set()


def add(key, p):
    h = hashlib.md5(open(p, "rb").read()).hexdigest()
    if h not in seen:
        seen.add(h); LO.setdefault(key, []).append(p)


for p in sorted(glob.glob(f"{DB}/Ratio/Baseline260808/DQ*/DQ*_corrected.csv")):
    if "DQ9-sus" in p:
        continue
    m = re.match(r"DQ(\d+)-TB(\d+)-TH(\d+)", os.path.basename(p))
    if m:
        add(tuple(int(m.group(i)) // 3 for i in (1, 2, 3)), p)

cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
todo = [(k, p) for k in sorted(LO) for p in LO[k] if p not in cache]
for i, (k, p) in enumerate(todo, 1):
    print(f"  신호 {i}/{len(todo)}  {os.path.basename(p)}", flush=True)
    cache[p] = [float(x) for x in sigs(p)]
if todo:
    json.dump(cache, open(CACHE, "w"))

rows = [(k, p, np.array(cache[p])) for k in sorted(LO) for p in LO[k]]
C = np.array([r[0] for r in rows], float)
A = np.array([r[2] for r in rows])
S3, S5 = A[:, :3], A[:, 3:]
print(f"\n맵 {len(rows)} · 조건 {len(LO)}")


def report(S, tag, ncomp):
    print(f"\n=== {tag} ===")
    X = np.c_[C, np.ones(len(C))]
    print(f"{'':>5} | {'자기 계수':>9} | {'절편':>9} | {'절편/중앙값':>10} | {'R²':>5}")
    for j, s in enumerate(SUB):
        coef, *_ = np.linalg.lstsq(X, S[:, j], rcond=None)
        pred = X @ coef
        r2 = 1 - ((S[:, j] - pred) ** 2).sum() / ((S[:, j] - S[:, j].mean()) ** 2).sum()
        med = np.median(S[:, j])
        print(f"{s:>5} | {coef[j]:9.0f} | {coef[3]:9.0f} | {100*coef[3]/med:9.0f}% | {r2:5.2f}")
    lo3, hi3 = np.median(S[C[:, 0] == 3, 0]), np.median(S[C[:, 0] == 24, 0])
    print(f"  DQ 3→24 µM 신호비 {hi3/lo3:.1f}배 (선형이면 8배)")
    # 조성(자기 기저 안에서 정규화)의 DQ 편향
    comp = S[:, :3] / S[:, :3].sum(1, keepdims=True) * 100
    true = C / C.sum(1, keepdims=True) * 100
    print(f"  NNLS 조성 편향  " + "  ".join(
        f"{SUB[j]} {np.mean(comp[:, j] - true[:, j]):+5.1f}%p" for j in range(3)))
    print(f"  NNLS 조성 오차  평균 {np.mean(np.abs(comp-true).sum(1)/2):.1f}%")


report(S3, "기저 3성분 (현재 앱)", 3)
report(S5, "기저 5성분 (BLK·INK 추가)", 5)
print("\nBLK·INK 계수가 실제로 얼마나 큰가 (5성분 기저)")
for k in range(3, S5.shape[1]):
    print(f"  {FULL[k]:<4} 중앙 {np.median(S5[:, k]):9.0f}  "
          f"(DQ 중앙 {np.median(S5[:, 0]):.0f} 대비 {np.median(S5[:, k])/max(np.median(S5[:,0]),1e-9):.2f}배)")
