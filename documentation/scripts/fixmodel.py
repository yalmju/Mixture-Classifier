"""모델 개선 시험: 스케일 피처(2137 밴드비)를 넣으면 구간을 넘어 일반화되나.

학습셋 = 고농도 Ratio_mix 34맵 + 저농도 tert-new 18맵(6조건).
조건 단위 leave-one-out. (a) 스펙트럼만  (b) 스펙트럼 + 2137밴드비
"""
import sys, os, re, glob, math
import numpy as np
from scipy.optimize import nnls
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

REPO = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/github/Mixture Classifier"
sys.path.insert(0, REPO)
sys.path.insert(0, "/private/tmp/claude-501/-Users-seungki2-Documents-GitHub-Mixture-Classifier--claude-worktrees-ink-check-6maps-run-ad8cec/61a536fa-e12d-4dee-8d68-95377dbe80ce/scratchpad")
import ink6, unmix
from real_data import load_map
from sers_mixture import als_baseline

names, WN, means = unmix._templates(
    "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB/Pure", True, None)
WN = np.asarray(WN, float)
FP = (WN >= 500) & (WN <= 1800)
P = unmix._l2(unmix._baseline_removed(means[:, FP], True))
SUB = ["DQ", "TBZ", "THI"]
SI = [names.index(s) for s in SUB]
BAND = 2136.7
DB = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB"
BAD = {"DQ500TBZ100", "DQ1000TBZ100"}          # 라벨 오류로 확인된 것


def feats(path, already_bl):
    wn, cube, _m, _c = load_map(path)
    wn = np.asarray(wn, float)
    X = np.asarray(cube, float)
    Y = X if already_bl else np.stack([v - als_baseline(v) for v in X])
    y = np.clip(Y, 0, None).mean(axis=0)
    spec = np.interp(WN[FP], wn, y)
    spec = spec / (np.linalg.norm(spec) + 1e-12)          # 형상 (게인 무관)
    # 2137 밴드비
    m = np.abs(wn - BAND) <= 12
    d = np.abs(wn - BAND)
    side = (d > 20) & (d <= 60)
    band = max(float(y[m].max() - np.median(y[side])), 0.0)
    a, _ = nnls(P.T, np.interp(WN[FP], wn, y))
    ana = a[SI].sum()
    return spec, math.log10(max(band / max(ana, 1e-9), 1e-6))


def parse(nm):
    toks = re.findall(r"(DQ|TBZ|TH[I1])(\d+)", nm)
    c = {"DQ": 0.0, "TBZ": 0.0, "THI": 0.0}
    for k, v in toks:
        c["THI" if k.startswith("TH") else k] += float(v)
    return np.array([c["DQ"], c["TBZ"], c["THI"]])


rows = []                                                  # (cond_id, spec, scale, y)
for p in sorted(glob.glob(f"{DB}/Ratio/Ratio_mix/*_corrected.csv")):
    nm = os.path.basename(p).replace("_corrected.csv", "").replace("-corrected.csv", "")
    if nm in BAD:
        continue
    C = parse(nm)
    if C.sum() == 0:
        continue
    s, sc = feats(p, already_bl=True)
    rows.append((f"hi:{nm}", s, sc, C / C.sum()))
COMP = {1: (6, 6, 24), 2: (12, 12, 12), 3: (24, 6, 6),
        4: (6, 6, 6), 5: (24, 24, 24), 6: (12, 12, 48)}
for n, C in COMP.items():
    for r in (1, 2, 3):
        p = f"{DB}/tert-new-baseline/{n}-{r}_corrected.csv"
        if os.path.exists(p):
            s, sc = feats(p, already_bl=True)
            rows.append((f"lo:{n}", s, sc, np.array(C, float) / sum(C)))

conds = sorted(set(r[0] for r in rows))
print(f"조건 {len(conds)}개 (고농도 {sum(1 for c in conds if c.startswith('hi'))} · "
      f"저농도 {sum(1 for c in conds if c.startswith('lo'))}), 맵 {len(rows)}개")

S = np.array([r[1] for r in rows])
SC = np.array([[r[2]] for r in rows])
Y = np.array([r[3] for r in rows])
G = np.array([r[0] for r in rows])


def run(use_scale, seed=0):
    err = {}
    X = np.hstack([S, SC]) if use_scale else S
    for c in conds:
        te = G == c
        tr = ~te
        sx = StandardScaler().fit(X[tr])
        m = MLPRegressor(hidden_layer_sizes=(128, 32), max_iter=3000,
                         random_state=seed, early_stopping=False, alpha=1e-3)
        m.fit(sx.transform(X[tr]), Y[tr])
        p = np.clip(m.predict(sx.transform(X[te])), 0, None)
        p = p / (p.sum(1, keepdims=True) + 1e-12)
        err[c] = np.mean(np.abs(p - Y[te]).sum(1) / 2 * 100)
    return err


for lab, us in (("스펙트럼만", False), ("스펙트럼 + 2137 밴드비", True)):
    e = run(us)
    hi = [v for k, v in e.items() if k.startswith("hi")]
    lo = [v for k, v in e.items() if k.startswith("lo")]
    print(f"  {lab:24}  전체 {np.mean(list(e.values())):5.1f}%   "
          f"고농도 {np.mean(hi):5.1f}%   저농도 {np.mean(lo):5.1f}%")
    if us:
        print("     저농도 조건별: " + "  ".join(f"#{k[3:]} {v:.0f}%" for k, v in sorted(e.items()) if k.startswith("lo")))

# 기준선
nn = []
for i, r in enumerate(rows):
    a, _ = nnls(P.T, S[i])
    f = a[SI] / max(a[SI].sum(), 1e-9)
    nn.append((G[i], np.abs(f - Y[i]).sum() / 2 * 100))
lo_nn = [v for g, v in nn if g.startswith("lo")]
hi_nn = [v for g, v in nn if g.startswith("hi")]
print(f"  {'NNLS (기준선)':24}  전체 {np.mean([v for _, v in nn]):5.1f}%   "
      f"고농도 {np.mean(hi_nn):5.1f}%   저농도 {np.mean(lo_nn):5.1f}%")
