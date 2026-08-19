"""패널 c — 등몰 삼원 농도 시리즈 (누적 막대).

  참 조성은 네 농도 모두 33.3/33.3/33.3 이다. 그런데 **표면 조성은 농도가 오르면
  THI 로 쏠린다** — 100·333 µM/성분에서 NNLS 가 0/0/100 을 읽는다(순수 THI 와 코사인
  0.99). 작업구간을 3–24 µM 으로 한정하는 근거 그림이다.

  두 갈래를 같이 낸다.
    NNLS  표면에 실제로 보이는 것            `tert260805_nnls.npz`   농도당 한 줄
    MLP   학습 모델이 답하는 것              `tert260805_result.npz` 농도당 반복 2장
  MLP 는 **학습 구간 밖**이라 표면에 DQ 가 0인데도 45–54% 를 답한다. 그 대비가 이
  패널의 두 번째 논점이므로 둘 다 실어야 한다.

  나오는 것 (`260808_data interpret/panels/`)
    panel_c_tert_series.csv        농도 × 방법 × 성분 — 누적 막대용 (MLP 는 반복 평균)
    panel_c_tert_series_reps.csv   MLP 반복 낱장 — 오차막대가 필요하면 이것으로

  실행:  python3 -u export_tert.py
"""
import os, sys, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

DB = paths.DB
OUT = f"{paths.INTERP}/panels"
SUB = ["DQ", "TBZ", "THI"]
TRUE = 100.0 / 3
os.makedirs(OUT, exist_ok=True)

NN = f"{HERE}/tert260805_nnls.npz"
TR = f"{HERE}/tert260805_result.npz"
for p in (NN, TR):
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} 가 없다 — tert260805.py 를 먼저 돌려야 한다")

n = np.load(NN, allow_pickle=True)
per, comp = np.asarray(n["per"], float), np.asarray(n["comp"], float)
t = np.load(TR, allow_pickle=True)
per_r, pred_r = np.asarray(t["per_uM"], float), np.asarray(t["pred"], float)

rows, reps = [], []
levels = sorted(set(np.round(per, 6)) | set(np.round(per_r, 6)))
for lv in levels:
    # NNLS — 농도당 한 줄
    m = np.isclose(per, lv)
    if m.any():
        v = comp[m][0]
        for i, s in enumerate(SUB):
            rows.append([f"{lv:.4g}", "NNLS", s, f"{v[i]:.3f}", f"{TRUE:.3f}", "1"])
    # MLP — 반복 평균 + 낱장
    mr = np.isclose(per_r, lv)
    if mr.any():
        V = pred_r[mr]
        mu = V.mean(0)
        for i, s in enumerate(SUB):
            rows.append([f"{lv:.4g}", "MLP", s, f"{mu[i]:.3f}", f"{TRUE:.3f}",
                         str(int(mr.sum()))])
        for j, row in enumerate(V):
            for i, s in enumerate(SUB):
                reps.append([f"{lv:.4g}", "MLP", str(int(t["reps"][mr][j])), s,
                             f"{row[i]:.3f}", f"{TRUE:.3f}"])


def write(name, head, body):
    with open(f"{OUT}/{name}", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(head); w.writerows(body)
    print(f"  ✓ {name:<36} {len(body)}행 × {len(head)}열")


write("panel_c_tert_series.csv",
      ["per_compound_uM", "method", "compound", "composition_pct", "true_pct", "n_maps"],
      rows)
write("panel_c_tert_series_reps.csv",
      ["per_compound_uM", "method", "replicate", "compound", "composition_pct", "true_pct"],
      reps)

print(f"\n  {'µM/성분':>8}  {'NNLS DQ/TBZ/THI':>22}   {'MLP (반복평균)':>22}")
for lv in levels:
    m = np.isclose(per, lv); mr = np.isclose(per_r, lv)
    a = "/".join(f"{v:5.1f}" for v in comp[m][0]) if m.any() else " " * 17
    b = "/".join(f"{v:5.1f}" for v in pred_r[mr].mean(0)) if mr.any() else ""
    print(f"  {lv:8.4g}  {a:>22}   {b:>22}")
print(f"\n참 조성은 모든 농도에서 {TRUE:.1f}/{TRUE:.1f}/{TRUE:.1f} — 기준 점선.")
print("캡션에 쓸 것: 고농도에서 THI 가 표면을 덮는다(관측 사실). Langmuir 로 설명하지 말 것 —")
print("K 는 TBZ 가 최대라 정반대를 예측한다. 그리고 MLP 는 학습 구간 밖이라 DQ 를 못 뺀다.")
