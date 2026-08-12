"""회수율과 RSD — SANTE 대조용.

  SANTE/11312/2021 은 **회수율 70–120% · RSD ≤ 20%** 를 요구한다.

  RSD 를 어디서 재느냐가 핵심이다.
    반복성(RSDr)  같은 조건을 **다시 조제해** 찍은 맵들 사이의 산포. SANTE 가 말하는
                  RSD 가 이것이다.
    조건간 산포    65조건 회수율의 표준편차. 이건 RSD 가 아니다 — 조건이 다르면 회수율이
                  다른 건 당연하므로 SANTE 와 비교하면 안 된다.
  둘을 한 표에 넣되 열을 갈라 놓는다. 합치면 반드시 오독된다.

  **재라벨본을 반복으로 쓴다.** `Ratio/mis/DQ9-sus-relabeled/` 12맵은 파일명 두 축이
  뒤바뀐 배치를 바로잡은 것으로, 12조건 모두 격자에 이미 있는 조건의 **두 번째 독립
  조제**다 (analysis/dq9-sus-reproducibility/README.md). 학습에는 넣지 않고 재현성
  계산에만 쓴다 — 그러면 반복 있는 조건이 6 → 18 로 늘고, 특히 반복이 하나도 없던
  **DQ 3 µM 구간**이 채워진다.

  실행:  python3 -u export_rsd.py
"""
import os, sys, csv, re, glob, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
import paths

DB = paths.DB
OUT = f"{DB}/260808_data interpret/panels"
SUB = ["DQ", "TBZ", "THI"]
SRC = f"{HERE}/conc38_result_cliponly.npz"
RELABEL = f"{DB}/Ratio/mis/DQ9-sus-relabeled"
FLOOR_uM = 6.0        # 보고 하한 — 3 µM 은 별도 표기 대상
os.makedirs(OUT, exist_ok=True)

z = np.load(SRC, allow_pickle=True)
C = np.asarray(z["C"], float)
EST = np.asarray(z["est"], float)
V = np.asarray(z["valid"], bool)
names = [os.path.basename(str(p)).replace("_corrected.csv", "") for p in z["paths"]]
design = np.array([48 not in C[i] for i in range(len(C))])      # THI 48 = 설계 밖
src_tag = ["grid"] * len(C)

# ------------------------------------------------------------------ 재라벨본을 반복으로 추가
# 이 맵들은 conc38 파이프라인을 안 거쳤으므로 같은 방식으로 추정치를 만들어야 공정하다.
# conc38 은 (신호 → 응답계수 → 게인) 경로를 쓰는데, 여기서는 그 결과 npz 만 있고
# 파이프라인 함수는 없다. 그래서 **격자 쪽과 같은 함수로 다시 추정**한다.
rel_files = sorted(glob.glob(f"{RELABEL}/DQ*_corrected.csv"))
if rel_files:
    from unmix import _templates, _baseline_removed, _l2
    from calibration import fit_B
    from dl_model import _refs, _map_spectra
    from real_data import load_map
    subs, wn, mask, _P0, lo, hi = _refs(f"{DB}/Pure", True, None)
    tnames, _wn, means = _templates(f"{DB}/Pure", True, None)
    P = _l2(_baseline_removed(means[:, mask], True))
    idx = [tnames.index(s) for s in SUB]

    # 격자 맵에서 (B → 추정 µM) 변환을 재적합한다. 원점을 지나는 성분별 기울기 하나.
    gridB, gridC = [], []
    seen = set()
    for p in sorted(glob.glob(f"{DB}/Ratio/Baseline260808/DQ*/DQ*_corrected.csv")):
        m = re.match(r"DQ(\d+)-TB(\d+)-TH(\d+)", os.path.basename(p))
        k = tuple(int(m.group(i)) / 3.0 for i in (1, 2, 3))
        if k in seen:
            continue
        seen.add(k)
        _w, cube, _mm, _c = load_map(p)
        gridB.append(fit_B(_map_spectra(cube, mask, 0)[0], P)[0][idx]); gridC.append(k)
    gridB = np.array(gridB); gridC = np.array(gridC)
    slope = np.array([(gridC[:, j] @ gridB[:, j]) / (gridC[:, j] @ gridC[:, j])
                      for j in range(3)])

    add_C, add_E = [], []
    for p in rel_files:
        m = re.match(r"DQ(\d+)-TB(\d+)-TH(\d+)", os.path.basename(p))
        c = np.array([int(m.group(i)) / 3.0 for i in (1, 2, 3)])
        _w, cube, _mm, _cc = load_map(p)
        b = fit_B(_map_spectra(cube, mask, 0)[0], P)[0][idx]
        add_C.append(c); add_E.append(b / slope)
        names.append(os.path.basename(p).replace("_corrected.csv", "") + " (relabeled)")
        src_tag.append("relabeled")
    C = np.vstack([C, add_C]); EST = np.vstack([EST, add_E])
    V = np.vstack([V, np.ones((len(add_C), 3), bool)])
    design = np.concatenate([design, np.ones(len(add_C), bool)])
    print(f"재라벨본 {len(add_C)}맵 추가 — 재현성 계산 전용 (학습에는 안 넣는다)")

    # 격자 쪽 추정도 같은 함수로 다시 만들어야 두 집단이 비교 가능하다. 그렇지 않으면
    # conc38 추정(격자)과 B/기울기 추정(재라벨)이 섞여 산포가 방법 차이로 오염된다.
    for i, nm in enumerate(names):
        if src_tag[i] != "grid":
            continue
        hit = [p for p in glob.glob(f"{DB}/Ratio/Baseline260808/DQ*/{nm}_corrected.csv")]
        if not hit:
            continue
        _w, cube, _mm, _c = load_map(hit[0])
        EST[i] = fit_B(_map_spectra(cube, mask, 0)[0], P)[0][idx] / slope
    print("격자 쪽 추정도 같은 B/기울기 경로로 다시 계산했다 (두 집단 비교 가능하게)")
else:
    print(f"⚠ 재라벨본을 못 찾았다: {RELABEL}")

groups = collections.defaultdict(list)
for i in range(len(C)):
    groups[tuple(np.round(C[i], 3))].append(i)

# ------------------------------------------------------------------ 조건별 (반복이 있는 것만)
rows, pooled = [], collections.defaultdict(list)
for key in sorted(groups):
    idxs = [i for i in groups[key] if design[i]]
    if len(idxs) < 2:
        continue
    for j, s in enumerate(SUB):
        ok = [i for i in idxs if V[i, j] and C[i, j] >= FLOOR_uM]
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
n_rep = sum(1 for v in groups.values() if len(v) >= 2)
print(f"\n  ✓ recovery_rsd_by_condition.csv   {len(rows)}행  "
      f"(반복 ≥2 조건 {n_rep}개 중 {FLOOR_uM:g} µM 이상)")

print(f"\n반복성 RSDr — 같은 조건 재조제 (성분당 {FLOOR_uM:g} µM 이상)")
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

print(f"\n조건간 산포 — **RSD 가 아니다** (SANTE 대조 금지)")
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
print(f"반복 있는 조건 {n_rep}개. SANTE 는 보통 반복 5회 — n 을 반드시 적을 것.")
