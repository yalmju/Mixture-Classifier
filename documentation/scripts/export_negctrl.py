"""음성 대조 — 없는 성분을 있다고 하는가.

  체크리스트 §5 가 요구하던 것. 새로 측정할 필요가 없다는 걸 확인했다: **고농도
  Ratio_mix 의 이원 혼합 24조건은 한 성분이 진짜 0** 이다. 즉 이미 성분별 음성 대조가
  들어 있다 ("thiram 만 있고 TBZ 없는 시료" 가 바로 `THI1000` 계열).

  두 가지를 잰다.
    거짓 양성 분율   참값이 0 인 성분을 몇 % 라고 답하는가. 분포와 백분위.
    문턱별 오탐률    "이 값 이상이면 있다" 로 판정할 때 음성 대조에서 몇 개가 걸리는가.
                     검출 문턱을 정하려면 이 곡선이 있어야 한다.

  자료는 `panel_f_drift.csv` (retrain_all held-out, blank 구성). 조성 헤드는 softmax 라
  절대 0 을 못 뱉는다 — **얼마나 작게 뱉느냐**가 전부다.

  실행:  python3 -u export_negctrl.py
"""
import os, sys, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

DB = paths.DB
OUT = f"{paths.INTERP}/panels"
SUB = ["DQ", "TBZ", "THI"]
SRC = f"{OUT}/panel_f_drift.csv"
os.makedirs(OUT, exist_ok=True)

if not os.path.exists(SRC):
    raise FileNotFoundError(f"{SRC} 가 없다 — export_drift.py 를 먼저 돌려야 한다")

rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
neg = {s: [] for s in SUB}          # 참값이 0 인 경우의 예측 %
pos = {s: [] for s in SUB}          # 참값이 0 이 아닌 경우
detail = []
for r in rows:
    for s in SUB:
        t = float(r[f"true_{s}_pct"]); p = float(r[f"pred_{s}_pct"])
        (neg if t == 0 else pos)[s].append(p)
        if t == 0:
            detail.append([r["mixture"], r["set"], s, f"{p:.3f}"])

print(f"{SRC.split('/')[-1]} — {len(rows)}조건")
print(f"\n{'성분':>5} | {'음성 n':>6} | {'예측 중앙':>9} | {'평균':>7} | {'95백분위':>9} | {'최대':>7}")
print("-" * 60)
summ = []
for s in SUB:
    v = np.array(neg[s])
    if not len(v):
        print(f"{s:>5} |      0 |  (음성 대조 없음)"); continue
    print(f"{s:>5} | {len(v):>6} | {np.median(v):8.2f}% | {v.mean():6.2f}% | "
          f"{np.percentile(v, 95):8.2f}% | {v.max():6.2f}%")
    summ.append([s, str(len(v)), f"{np.median(v):.3f}", f"{v.mean():.3f}",
                 f"{np.percentile(v, 95):.3f}", f"{v.max():.3f}"])

# ------------------------------------------------------------------ 문턱별 오탐 / 검출
print(f"\n문턱별 — 음성 대조 오탐률과, 참으로 존재하는 경우의 검출률")
print(f"{'문턱':>6} | " + " | ".join(f"{s:>16}" for s in SUB))
print(f"{'':>6} | " + " | ".join(f"{'오탐 / 검출':>16}" for _ in SUB))
print("-" * 62)
thr_rows = []
for thr in (1, 2, 5, 10, 15, 20):
    cells = []
    for s in SUB:
        v = np.array(neg[s]); w = np.array(pos[s])
        fp = 100 * (v >= thr).mean() if len(v) else np.nan
        tp = 100 * (w >= thr).mean() if len(w) else np.nan
        cells.append(f"{fp:5.0f}% / {tp:5.0f}%")
        thr_rows.append([str(thr), s, f"{fp:.2f}", f"{tp:.2f}",
                         str(len(v)), str(len(w))])
    print(f"{thr:>5}% | " + " | ".join(f"{c:>16}" for c in cells))


def write(name, head, body):
    with open(f"{OUT}/{name}", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(head); w.writerows(body)
    print(f"  ✓ {name:<34} {len(body)}행")


print()
write("negative_control_summary.csv",
      ["compound", "n_negative", "median_pred_pct", "mean_pred_pct",
       "p95_pred_pct", "max_pred_pct"], summ)
write("negative_control_thresholds.csv",
      ["threshold_pct", "compound", "false_positive_pct", "detection_pct",
       "n_negative", "n_positive"], thr_rows)
write("negative_control_detail.csv",
      ["mixture", "set", "absent_compound", "predicted_pct"], detail)

print("\n조성 헤드는 softmax 라 0 을 못 뱉는다 — 절대 0 이 아니라 **작다** 로 읽을 것.")
print("검출 문턱을 정하려면 negative_control_thresholds.csv 의 오탐/검출 짝을 보고 고른다.")
