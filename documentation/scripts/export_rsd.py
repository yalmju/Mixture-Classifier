"""회수율과 RSD — SANTE 대조용.

  SANTE/11312/2021 은 **회수율 70–120% · RSD ≤ 20%** 를 요구한다. 지금까지는 평균
  배수오차만 있어서 그 표에 올릴 수가 없었다.

  RSD 를 어디서 재느냐가 핵심이다.
    반복성(RSDr)  같은 조건을 **다시 조제해** 찍은 맵들 사이의 산포.
                  SANTE 가 말하는 RSD 가 이것이다. 그런데 65조건 중 **반복이 있는
                  조건은 6개뿐**이다 (3장 3조건 + 4장 3조건). n 을 반드시 같이 적을 것.
    조건간 산포    65조건 회수율의 표준편차. 이건 RSD 가 아니다 — 조건이 다르면
                  회수율이 다른 건 당연하므로 SANTE 와 비교하면 안 된다.

  둘을 한 표에 넣되 열을 갈라 놓는다. 합치면 반드시 오독된다.

  실행:  python3 -u export_rsd.py
"""
import os, sys, csv, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

DB = paths.DB
OUT = f"{DB}/260808_data interpret/panels"
SUB = ["DQ", "TBZ", "THI"]
SRC = f"{HERE}/conc38_result_cliponly.npz"
FLOOR_uM = 6.0        # 보고 하한 — 3 µM 은 별도 표기 대상
os.makedirs(OUT, exist_ok=True)

z = np.load(SRC, allow_pickle=True)
C, EST, V = z["C"], z["est"], np.asarray(z["valid"], bool)
paths_ = [os.path.basename(str(p)).replace("_corrected.csv", "") for p in z["paths"]]
design = np.array([48 not in C[i] for i in range(len(C))])      # THI 48 = 설계 밖

groups = collections.defaultdict(list)
for i in range(len(C)):
    groups[tuple(np.round(C[i], 3))].append(i)

# ------------------------------------------------------------------ 조건별 (반복이 있는 것만)
rows, pooled = [], collections.defaultdict(list)
for key in sorted(groups):
    idx = [i for i in groups[key] if design[i]]
    if len(idx) < 2:
        continue
    for j, s in enumerate(SUB):
        ok = [i for i in idx if V[i, j] and C[i, j] >= FLOOR_uM]
        if len(ok) < 2:
            continue
        est = EST[ok, j]
        true = C[ok[0], j]
        rec = est / true * 100
        rsd = est.std(ddof=1) / est.mean() * 100 if est.mean() > 0 else np.nan
        rows.append([f"{key[0]:g}/{key[1]:g}/{key[2]:g}", s, f"{true:g}", str(len(ok)),
                     f"{rec.mean():.1f}", f"{est.mean():.4f}", f"{est.std(ddof=1):.4f}",
                     f"{rsd:.1f}",
                     str(int(70 <= rec.mean() <= 120)), str(int(rsd <= 20))])
        pooled[s].append((rsd, len(ok), rec.mean()))

with open(f"{OUT}/recovery_rsd_by_condition.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["condition_uM", "compound", "true_uM", "n_replicates", "mean_recovery_pct",
                "mean_estimate_uM", "sd_uM", "rsd_pct",
                "sante_recovery_ok_70_120", "sante_rsd_ok_le20"])
    w.writerows(rows)
print(f"  ✓ recovery_rsd_by_condition.csv   {len(rows)}행  "
      f"(반복 ≥2 이고 {FLOOR_uM:g} µM 이상인 조건×성분)")

# ------------------------------------------------------------------ 성분별 요약
print(f"\n반복성 RSDr — 같은 조건을 다시 조제해 찍은 맵 사이 (성분당 {FLOOR_uM:g} µM 이상)")
print(f"{'성분':>5} | {'조건수':>5} | {'RSDr 중앙':>9} | {'RSDr 범위':>13} | "
      f"{'회수율 평균':>10} | SANTE")
summ = []
for s in SUB:
    v = pooled[s]
    if not v:
        continue
    r = np.array([x[0] for x in v]); rec = np.array([x[2] for x in v])
    ok = int(((r <= 20) & (rec >= 70) & (rec <= 120)).sum())
    print(f"{s:>5} | {len(v):>5} | {np.median(r):8.1f}% | "
          f"{r.min():5.1f}–{r.max():5.1f}% | {rec.mean():9.0f}% | {ok}/{len(v)} 통과")
    summ.append([s, str(len(v)), f"{np.median(r):.1f}", f"{r.min():.1f}", f"{r.max():.1f}",
                 f"{rec.mean():.1f}", f"{np.median(rec):.1f}", str(ok)])

# ------------------------------------------------------------------ 조건간 산포 (RSD 아님)
print(f"\n조건간 산포 — 65조건 회수율의 흩어짐. **RSD 가 아니다** (SANTE 대조 금지)")
between = []
for j, s in enumerate(SUB):
    ok = design & V[:, j] & (C[:, j] >= FLOOR_uM)
    rec = EST[ok, j] / C[ok, j] * 100
    print(f"{s:>5} | n={ok.sum():>3} | 평균 {rec.mean():5.0f}% · 중앙 {np.median(rec):5.0f}% "
          f"· SD {rec.std(ddof=1):5.0f}%p · 70–120% 안 {100*((rec>=70)&(rec<=120)).mean():.0f}%")
    between.append([s, str(int(ok.sum())), f"{rec.mean():.1f}", f"{np.median(rec):.1f}",
                    f"{rec.std(ddof=1):.1f}",
                    f"{100 * ((rec >= 70) & (rec <= 120)).mean():.1f}"])

with open(f"{OUT}/recovery_rsd_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["metric", "compound", "n", "a", "b", "c", "d", "e"])
    for r in summ:
        w.writerow(["repeatability_RSDr", r[0], r[1], "median_rsd_pct=" + r[2],
                    "min_rsd=" + r[3], "max_rsd=" + r[4], "mean_recovery_pct=" + r[5],
                    "sante_pass=" + r[7]])
    for r in between:
        w.writerow(["between_condition_spread_NOT_RSD", r[0], r[1],
                    "mean_recovery_pct=" + r[2], "median=" + r[3], "sd_pp=" + r[4],
                    "within_70_120_pct=" + r[5], ""])
print(f"\n  ✓ recovery_rsd_summary.csv")
print(f"\n반복 있는 조건이 {sum(1 for v in groups.values() if len(v) >= 2)}개뿐이다 "
      f"(3장 3조건 + 4장 3조건). SANTE 는 보통 반복 5회를 본다 — n 을 반드시 적을 것.")
