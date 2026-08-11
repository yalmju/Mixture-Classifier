"""sus 12맵이 사실 '전부 같은 혼합물' 인지 검정."""
import pickle, os, numpy as np, itertools
from scipy.optimize import nnls
from scipy import stats

d = os.path.dirname(__file__)
D = pickle.load(open(os.path.join(d, "data.pkl"), "rb"))
U = pickle.load(open(os.path.join(d, "unmix.pkl"), "rb"))
R, data = D["R"], D["data"]
LV = [9, 18, 36, 72]
conds = [f"DQ9-TB{tb}-TH{th}" for tb in LV for th in LV]
shared = [c for c in conds if ("DQ9-sus", c) in data]

# ---- 픽셀별 조성 분율 (good 픽셀만) ----
PF = os.path.join(d, "pixfrac.pkl")
if os.path.exists(PF):
    F = pickle.load(open(PF, "rb"))
else:
    F = {}
    for k in data:
        X = data[k]["X"][U[k]["good"]]
        C = np.array([nnls(R, np.clip(x, 0, None))[0] for x in X])[:, :3]
        s = C.sum(1, keepdims=True); s[s == 0] = 1
        F[k] = C / s
        print("unmixed", k, F[k].shape, flush=True)
    pickle.dump(F, open(PF, "wb"))

NAMES = ["DQ", "TBZ", "THI"]

print("="*92)
print("1. '모든 맵이 동일한 혼합물' 귀무가설 검정 — 맵간 분산 vs 맵내 픽셀 분산 (one-way ANOVA)")
print("   F가 1에 가까우면 맵끼리 구별 안 됨(=사용자 가설), 크면 맵마다 조성이 실제로 다름")
print("="*92)
for grp in ["DQ9", "DQ9-sus"]:
    ks = [c for c in conds if (grp, c) in data]
    print(f"  {grp}  (n={len(ks)} maps)")
    for j, nm in enumerate(NAMES):
        groups = [F[(grp, c)][:, j] for c in ks]
        Fst, p = stats.f_oneway(*groups)
        bet = np.std([g.mean() for g in groups])
        wit = np.mean([g.std() for g in groups])
        print(f"     {nm:4} F={Fst:8.1f}  p={p:.2e}   맵간 SD={bet:.3f}  맵내 SD={wit:.3f}  비율={bet/wit:5.2f}")

print()
print("="*92)
print("2. sus 맵끼리 서로 구별되는가? — 맵 쌍별 조성 차이 (Δ분율 L2) 분포")
print("="*92)
for grp in ["DQ9", "DQ9-sus"]:
    ks = [c for c in conds if (grp, c) in data]
    ds = [np.linalg.norm(U[(grp,a)]["frac"] - U[(grp,b)]["frac"]) for a, b in itertools.combinations(ks, 2)]
    print(f"  {grp:9} 맵쌍 거리: median {np.median(ds):.3f}  IQR [{np.percentile(ds,25):.3f}, {np.percentile(ds,75):.3f}]  max {np.max(ds):.3f}")

print()
print("="*92)
print("3. sus 조성의 '설정값 의존성' 분해 — 2-way ANOVA 대용: TB수준/TH수준별 평균")
print("="*92)
for grp in ["DQ9", "DQ9-sus"]:
    ks = [c for c in conds if (grp, c) in data]
    tb = np.array([int(c.split("-TB")[1].split("-")[0]) for c in ks])
    th = np.array([int(c.split("-TH")[1]) for c in ks])
    for j, nm in enumerate(NAMES):
        f = np.array([U[(grp,c)]["frac"][j] for c in ks])
        rTB = f.max() - f.min()
        eTB = np.ptp([f[tb==v].mean() for v in sorted(set(tb))])
        eTH = np.ptp([f[th==v].mean() for v in sorted(set(th))])
        print(f"  {grp:9} {nm:4} 전체범위={rTB:.3f}  TB수준효과={eTB:.3f}  TH수준효과={eTH:.3f}")

print()
print("="*92)
print("4. 농도 역추정: DQ9로 응답계수를 학습하고 sus 맵의 '실제' TB/TH 농도를 추정")
print("   log(f_X/f_DQ) = a + b*log(C_X/9) 을 DQ9 16맵으로 적합 → sus에 역적용")
print("="*92)
ks9 = [c for c in conds if ("DQ9", c) in data]
tb9 = np.array([int(c.split("-TB")[1].split("-")[0]) for c in ks9], float)
th9 = np.array([int(c.split("-TH")[1]) for c in ks9], float)
fit = {}
for j, nm, cc in [(1, "TBZ", tb9), (2, "THI", th9)]:
    y = np.log(np.array([max(U[("DQ9",c)]["frac"][j], 1e-3) / max(U[("DQ9",c)]["frac"][0], 1e-3) for c in ks9]))
    x = np.log(cc / 9.0)
    b, a = np.polyfit(x, y, 1)
    pred = a + b*x
    r2 = 1 - np.var(y - pred)/np.var(y)
    fit[nm] = (a, b)
    print(f"  DQ9 적합 {nm}: 기울기 b={b:.3f}  절편 a={a:.3f}  R²={r2:.3f}")
print()
print(f"  {'sus map':16} {'추정 TB (µM)':>13} {'추정 TH (µM)':>13}   {'설정 TB':>7} {'설정 TH':>7}")
est = []
for c in shared:
    f = U[("DQ9-sus", c)]["frac"]
    out = {}
    for j, nm in [(1, "TBZ"), (2, "THI")]:
        a, b = fit[nm]
        y = np.log(max(f[j],1e-3)/max(f[0],1e-3))
        out[nm] = 9.0*np.exp((y - a)/b)
    est.append((c, out["TBZ"], out["THI"]))
    print(f"  {c:16} {out['TBZ']:13.1f} {out['THI']:13.1f}   "
          f"{c.split('-TB')[1].split('-')[0]:>7} {c.split('-TH')[1]:>7}")
E = np.array([[e[1], e[2]] for e in est])
print(f"\n  추정 TB: 중앙값 {np.median(E[:,0]):.1f} µM, 범위 {E[:,0].min():.1f}–{E[:,0].max():.1f}, "
      f"기하 CV={np.std(np.log(E[:,0])):.2f}")
print(f"  추정 TH: 중앙값 {np.median(E[:,1]):.1f} µM, 범위 {E[:,1].min():.1f}–{E[:,1].max():.1f}, "
      f"기하 CV={np.std(np.log(E[:,1])):.2f}")
# DQ9 자신에 대해 같은 역추정 (기준선: 이 방법의 고유 오차)
print()
print("  [기준선] 같은 방법을 DQ9 자신에 적용했을 때 회수율:")
for c in ks9:
    f = U[("DQ9", c)]["frac"]
    o = {}
    for j, nm in [(1,"TBZ"),(2,"THI")]:
        a, b = fit[nm]
        o[nm] = 9.0*np.exp((np.log(max(f[j],1e-3)/max(f[0],1e-3)) - a)/b)
    print(f"    {c:16} 추정 TB={o['TBZ']:7.1f} (설정 {c.split('-TB')[1].split('-')[0]:>2})   "
          f"추정 TH={o['THI']:7.1f} (설정 {c.split('-TH')[1]:>2})")

print()
print("="*92)
print("5. 가설 3종 직접 비교: sus 12맵을 DQ9의 어떤 라벨로 설명해야 오차가 최소인가")
print("="*92)
def d2(susc, dq9c):
    return float(np.linalg.norm(U[("DQ9-sus",susc)]["frac"] - U[("DQ9",dq9c)]["frac"]))
cands = {
    "identity (라벨 그대로)":      [(c, c) for c in shared],
    "TBZ↔THI swap":               [(c, f"DQ9-TB{c.split('-TH')[1]}-TH{c.split('-TB')[1].split('-')[0]}") for c in shared],
}
for fixed in ks9:                       # '전부 동일' 가설: 모든 sus 맵 = 고정 조건 하나
    cands[f"전부 동일 = {fixed}"] = [(c, fixed) for c in shared]
res = sorted(((np.mean([d2(a,b) for a,b in p]), n) for n, p in cands.items()))
for v, n in res:
    print(f"  {n:34} 평균거리={v:.3f}")
