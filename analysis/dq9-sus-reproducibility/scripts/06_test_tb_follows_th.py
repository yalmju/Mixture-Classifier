"""사용자 가설: sus에서 TH는 정확, TB는 라벨이 아니라 TH를 따라갔다 (TB_actual = TH_label)."""
import pickle, os, numpy as np
from scipy import stats

d = os.path.dirname(__file__)
D = pickle.load(open(os.path.join(d, "data.pkl"), "rb"))
U = pickle.load(open(os.path.join(d, "unmix.pkl"), "rb"))
LV = [9, 18, 36, 72]
conds = [f"DQ9-TB{tb}-TH{th}" for tb in LV for th in LV]
ks9 = [c for c in conds if ("DQ9", c) in D["data"]]
shared = [c for c in conds if ("DQ9-sus", c) in D["data"]]
lab = lambda c: (int(c.split("-TB")[1].split("-")[0]), int(c.split("-TH")[1]))

# --- DQ9로 응답식 적합 → 농도 역추정 ---
tb9 = np.array([lab(c)[0] for c in ks9], float); th9 = np.array([lab(c)[1] for c in ks9], float)
fit = {}
for j, nm, cc in [(1,"TBZ",tb9), (2,"THI",th9)]:
    y = np.log([max(U[("DQ9",c)]["frac"][j],1e-3)/max(U[("DQ9",c)]["frac"][0],1e-3) for c in ks9])
    b, a = np.polyfit(np.log(cc/9.0), y, 1); fit[nm] = (a, b)
def est(grp, c):
    f = U[(grp,c)]["frac"]
    return [9.0*np.exp((np.log(max(f[j],1e-3)/max(f[0],1e-3))-fit[nm][0])/fit[nm][1])
            for j, nm in [(1,"TBZ"),(2,"THI")]]

print("="*94)
print("1. 역추정 농도가 '어느 라벨'을 따라가는가 (Spearman)")
print("   사용자 가설이 맞다면 sus에서: 추정TBZ ~ TH라벨 (강함), 추정TBZ ~ TB라벨 (없음)")
print("="*94)
for grp, ks in [("DQ9", ks9), ("DQ9-sus", shared)]:
    tb = np.array([lab(c)[0] for c in ks], float); th = np.array([lab(c)[1] for c in ks], float)
    E = np.array([est(grp,c) for c in ks])
    print(f"  {grp}")
    for j, nm in [(0,"추정TBZ"), (1,"추정THI")]:
        rb = stats.spearmanr(tb, E[:,j]); rh = stats.spearmanr(th, E[:,j])
        print(f"     {nm} ~ TB라벨  rho={rb.statistic:+.3f} (p={rb.pvalue:.3f})    "
              f"{nm} ~ TH라벨  rho={rh.statistic:+.3f} (p={rh.pvalue:.3f})")

print()
print("="*94)
print("2. sus: 라벨 수준별 추정 농도 (중앙값) — TB라벨로 묶을 때 vs TH라벨로 묶을 때")
print("="*94)
E = {c: est("DQ9-sus", c) for c in shared}
for axis, idx in [("TB라벨", 0), ("TH라벨", 1)]:
    print(f"  {axis}로 묶기:")
    for v in LV:
        g = [E[c] for c in shared if lab(c)[idx] == v]
        if not g: continue
        g = np.array(g)
        print(f"     {axis[:2]}={v:<3} n={len(g)}  추정TBZ 중앙값={np.median(g[:,0]):6.1f}   "
              f"추정THI 중앙값={np.median(g[:,1]):6.1f}")

print()
print("="*94)
print("3. 라벨→실제 매핑 가설 모음 비교 (sus 맵 ↔ 해당 DQ9 맵, 조성분율 L2, 낮을수록 좋음)")
print("   ※ DQ9 자기 자신을 서로 비교한 값이 아니라, sus를 DQ9의 어떤 조건으로 설명하는지")
print("="*94)
def dist(susc, tb, th):
    k = ("DQ9", f"DQ9-TB{tb}-TH{th}")
    return None if k not in U else float(np.linalg.norm(U[("DQ9-sus",susc)]["frac"] - U[k]["frac"]))
HYP = {
    "① 라벨 그대로 (TB,TH)":            lambda i,j: (LV[i], LV[j]),
    "② TBZ↔THI 뒤바뀜 (TH,TB)":         lambda i,j: (LV[j], LV[i]),
    "③ 사용자 가설: TB가 TH를 따라감 (TH,TH)": lambda i,j: (LV[j], LV[j]),
    "④ 반대: TH가 TB를 따라감 (TB,TB)":   lambda i,j: (LV[i], LV[i]),
    "⑤ TB만 맞고 TH가 TB를 따라감 → ④와 동일": lambda i,j: (LV[i], LV[i]),
}
scores = {}
for name, fmap in HYP.items():
    ds = []
    for c in shared:
        i, j = LV.index(lab(c)[0]), LV.index(lab(c)[1])
        tb, th = fmap(i, j)
        v = dist(c, tb, th)
        if v is not None: ds.append(v)
    scores[name] = (np.mean(ds), len(ds))
for n, (v, k) in sorted(scores.items(), key=lambda x: x[1][0]):
    print(f"  {n:42} 평균거리={v:.3f}  (n={k})")

print()
print("="*94)
print("4. 가설 ③ 상세: sus 각 맵 vs DQ9의 대각선 맵 (TB=TH=TH라벨)")
print("="*94)
print(f"  {'sus map':16} {'sus DQ/TBZ/THI':22} {'대응 DQ9 (TB=TH=THlabel)':26} {'DQ9 DQ/TBZ/THI':22} {'거리':>6}")
for c in shared:
    v = lab(c)[1]
    k = ("DQ9", f"DQ9-TB{v}-TH{v}")
    fb, fa = U[("DQ9-sus",c)]["frac"], U[k]["frac"]
    print(f"  {c:16} {'/'.join(f'{x:.2f}' for x in fb):22} {k[1]:26} "
          f"{'/'.join(f'{x:.2f}' for x in fa):22} {np.linalg.norm(fa-fb):6.3f}")

print()
print("="*94)
print("5. 가설 ③의 직접 예측: sus에서 TB라벨은 아무 영향 없어야 한다")
print("   같은 TH라벨을 공유하는 sus 맵끼리의 거리 vs 다른 TH라벨 맵끼리의 거리")
print("="*94)
import itertools
same_th = [np.linalg.norm(U[("DQ9-sus",a)]["frac"]-U[("DQ9-sus",b)]["frac"])
           for a,b in itertools.combinations(shared,2) if lab(a)[1]==lab(b)[1]]
diff_th = [np.linalg.norm(U[("DQ9-sus",a)]["frac"]-U[("DQ9-sus",b)]["frac"])
           for a,b in itertools.combinations(shared,2) if lab(a)[1]!=lab(b)[1]]
same_tb = [np.linalg.norm(U[("DQ9-sus",a)]["frac"]-U[("DQ9-sus",b)]["frac"])
           for a,b in itertools.combinations(shared,2) if lab(a)[0]==lab(b)[0]]
diff_tb = [np.linalg.norm(U[("DQ9-sus",a)]["frac"]-U[("DQ9-sus",b)]["frac"])
           for a,b in itertools.combinations(shared,2) if lab(a)[0]!=lab(b)[0]]
print(f"  같은 TH라벨끼리 거리 중앙값 {np.median(same_th):.3f} (n={len(same_th)})  vs 다른 TH라벨 {np.median(diff_th):.3f} (n={len(diff_th)})")
print(f"  같은 TB라벨끼리 거리 중앙값 {np.median(same_tb):.3f} (n={len(same_tb)})  vs 다른 TB라벨 {np.median(diff_tb):.3f} (n={len(diff_tb)})")
print(f"  Mann-Whitney (같은TH vs 다른TH): p={stats.mannwhitneyu(same_th,diff_th).pvalue:.4f}")
print(f"  Mann-Whitney (같은TB vs 다른TB): p={stats.mannwhitneyu(same_tb,diff_tb).pvalue:.4f}")
print("  → 가설 ③이면 '같은 TH끼리'가 확 작아야 하고 '같은 TB끼리'는 차이가 없어야 함")

print()
print("="*94)
print("6. 같은 검정을 DQ9(정상 대조군)에 적용 — 이 지표가 실제로 작동하는지 확인")
print("="*94)
for nm, key in [("TH", 1), ("TB", 0)]:
    s = [np.linalg.norm(U[("DQ9",a)]["frac"]-U[("DQ9",b)]["frac"])
         for a,b in itertools.combinations(ks9,2) if lab(a)[key]==lab(b)[key]]
    dfr = [np.linalg.norm(U[("DQ9",a)]["frac"]-U[("DQ9",b)]["frac"])
         for a,b in itertools.combinations(ks9,2) if lab(a)[key]!=lab(b)[key]]
    print(f"  DQ9: 같은 {nm}라벨 {np.median(s):.3f} vs 다른 {nm}라벨 {np.median(dfr):.3f}   "
          f"p={stats.mannwhitneyu(s,dfr).pvalue:.4f}")
