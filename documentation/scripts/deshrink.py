"""눌림 보정 — 예측이 p ≈ a·t + b 로 눌려 있는데, 되돌리면 나아지는가.

  성분별 기울기가 0.56–0.62 다. 참 조성이 넓게 퍼져 있는데 예측은 중앙으로 모인다.
  `(p − b)/a` 로 펴면 기울기는 1 이 되지만 **분산이 1/a² 배로 커진다**. 조성 오차가
  줄지 늘지는 부호가 정해져 있지 않아 재봐야 안다.

  누수를 막는 게 핵심이다. a·b 를 **held-out 을 포함해 적합하면 당연히 좋아진다** —
  자기 답을 보고 보정하는 것이다. 여기서는 `piemap_data.pkl` 의 fold 구조를 그대로 써서
  **학습 fold 에서만 a·b 를 적합하고 held-out 조건에 적용**한다.

  세 가지를 비교한다.
    none        보정 없음 (지금)
    per-comp    성분마다 a·b
    global      세 성분 공통 a·b
  펴고 나면 음수·합≠100 이 생기므로 0 에서 자르고 100 으로 다시 정규화한다.

  실행:  python3 -u deshrink.py
"""
import os, sys, csv, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

DB = paths.DB
CACHE = f"{DB}/260808_data interpret/piemap_260812/piemap_data.pkl"
OUT = f"{DB}/260808_data interpret/panels"
SUB = ["DQ", "TBZ", "THI"]
FOLDS = 4
os.makedirs(OUT, exist_ok=True)

d = pickle.load(open(CACHE, "rb"))
cond = d["cond"]
keys = sorted(cond)
T = np.array([cond[k]["true"] for k in keys])
P = np.array([cond[k]["pred"] for k in keys])

# piemap_compute / retrain64 와 같은 fold (seed 0)
rng = np.random.default_rng(0)
order = rng.permutation(len(keys))
folds = [set(keys[i] for i in order[f::FOLDS]) for f in range(FOLDS)]
kidx = {k: i for i, k in enumerate(keys)}


def err(t, p):
    return np.abs(p - t).sum(1) / 2


def renorm(p):
    p = np.clip(p, 0, None)
    s = p.sum(1, keepdims=True)
    return np.divide(p, s, out=np.full_like(p, 100.0 / p.shape[1]), where=s > 0) * 100


def apply_fit(mode):
    """fold 마다 학습 조건에서 a·b 를 적합하고 held-out 조건에만 적용."""
    out = P.copy()
    for te in folds:
        te_i = np.array([kidx[k] for k in te])
        tr_i = np.array([kidx[k] for k in keys if k not in te])
        q = P[te_i].astype(float).copy()
        if mode == "per-comp":
            for j in range(3):
                a, b = np.polyfit(T[tr_i, j], P[tr_i, j], 1)
                q[:, j] = (q[:, j] - b) / a if abs(a) > 1e-9 else q[:, j]
        else:                       # global
            a, b = np.polyfit(T[tr_i].ravel(), P[tr_i].ravel(), 1)
            q = (q - b) / a if abs(a) > 1e-9 else q
        out[te_i] = q
    return renorm(out)


res = {"none": P}
for m in ("per-comp", "global"):
    res[m] = apply_fit(m)

print(f"{len(keys)}조건 · fold {FOLDS} (학습 fold 에서만 a·b 적합)\n")
print(f"{'보정':>10} | {'평균 오차':>9} | {'중앙':>7} | {'기울기 DQ/TBZ/THI':>20} | {'편향(%p)':>18}")
print("-" * 82)
rows = []
for m, Q in res.items():
    e = err(T, Q)
    sl = [np.polyfit(T[:, j], Q[:, j], 1)[0] for j in range(3)]
    bi = [Q[:, j].mean() - T[:, j].mean() for j in range(3)]
    print(f"{m:>10} | {e.mean():8.1f}% | {np.median(e):6.1f}% | "
          + "/".join(f"{v:5.2f}" for v in sl) + " | "
          + "/".join(f"{v:+5.1f}" for v in bi))
    rows.append([m, f"{e.mean():.3f}", f"{np.median(e):.3f}"]
                + [f"{v:.4f}" for v in sl] + [f"{v:.3f}" for v in bi])

base = err(T, res["none"])
print()
from scipy.stats import wilcoxon
for m in ("per-comp", "global"):
    e = err(T, res[m])
    d_ = base - e
    p = wilcoxon(base, e).pvalue
    print(f"  {m:>9}  평균 {d_.mean():+.2f}%p · 개선 {int((d_>0).sum())}/{len(d_)} · "
          f"Wilcoxon p={p:.4f}  → " + ("좋아짐" if d_.mean() > 0 and p < 0.05 else
                                       "나빠짐" if d_.mean() < 0 and p < 0.05 else "차이 없음"))

with open(f"{OUT}/deshrink.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["correction", "mean_error_pct", "median_error_pct"]
               + [f"slope_{s}" for s in SUB] + [f"bias_{s}_pp" for s in SUB])
    w.writerows(rows)
print(f"\n  ✓ deshrink.csv → {OUT}")
print("기울기가 1 로 가는 것과 오차가 주는 것은 별개다 — 펴면 분산이 1/a² 로 커진다.")
