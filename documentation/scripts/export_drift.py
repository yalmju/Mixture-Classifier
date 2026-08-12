"""패널 f — 상대 드리프트 막대 (2성분 / 3성분).

  앱 Recovery 탭의 `_plot_reldrift` 와 같은 계산이다:
    드리프트 = 조성거리(참, 예측) = ½·Σ|pred − true|
    막대 길이 = 그 그룹에서 가장 나쁜 것 대비 %  (그룹마다 따로 정규화)
    그룹 = 성분 수. 3성분 경쟁은 2성분과 다른 문제라 한 축에 섞으면 스케일이 망가진다.

  성분 수는 **투입된 성분 수**로 센다. 2% 같은 문턱을 두면 DQ 가 0.5% 인
  `1000/1000/10` 이 조용히 "2성분" 으로 강등돼서, 정작 보고 싶은 묻힌-미량 삼원이
  그림에서 사라진다.

  자료는 `retrain_all.txt` — 고농도 32조건까지 조성 키로 묶어 held-out 으로 채점한
  유일한 실행이다 (4-fold 조건단위, `blank` 구성). **채택 구성(blank+sips)이 아니다** —
  캡션에 그렇게 적을 것.

  실행:  python3 -u export_drift.py [retrain_all.txt]
"""
import os, sys, re, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DB = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB"
SRC = sys.argv[1] if len(sys.argv) > 1 else \
    f"{DB}/260808_data interpret/64conditions/retrain_all.txt"
OUT = f"{DB}/260808_data interpret/panels"
SUB = ["DQ", "TBZ", "THI"]
os.makedirs(OUT, exist_ok=True)

rows = []
for line in open(SRC):
    if not re.search(r"\|\s+[\d.]+% \|", line):
        continue
    hi = line.strip().startswith("HI")
    parts = [p.strip() for p in line.replace("HI", "", 1).split("|")]
    if len(parts) < 4:
        continue
    try:
        conc = [float(v) for v in parts[0].split("/")]
        true = [float(v) for v in parts[1].split("/")]
        pred = [float(v) for v in parts[2].split("/")]
    except ValueError:
        continue
    rows.append({"set": "high" if hi else "low", "conc": conc, "true": true, "pred": pred,
                 "drift": float(np.abs(np.array(pred) - np.array(true)).sum() / 2)})

print(f"{SRC.split('/')[-1]} — {len(rows)}조건 "
      f"(고농도 {sum(1 for r in rows if r['set'] == 'high')} · "
      f"저농도 {sum(1 for r in rows if r['set'] == 'low')})")

for r in rows:
    r["ncomp"] = int(sum(1 for v in r["conc"] if v > 0))       # 투입된 성분 수
    r["dominant"] = SUB[int(np.argmax(r["true"]))]
    r["name"] = "".join(f"{SUB[i]}{r['conc'][i]:g}" for i in range(3) if r["conc"][i] > 0)

# 그룹마다 따로 정규화 — 3성분이 스케일을 잡아먹으면 2성분 막대가 전부 눌린다
body = []
for st in ("high", "low"):
    for nc in sorted({r['ncomp'] for r in rows if r['set'] == st}):
        grp = [r for r in rows if r["set"] == st and r["ncomp"] == nc]
        if not grp:
            continue
        worst = max(r["drift"] for r in grp) or 1.0
        grp.sort(key=lambda r: (-r["drift"]))
        print(f"  {st:<5} {nc}성분 {len(grp):>2}조건 — 최악 {worst:.1f}% · "
              f"평균 {np.mean([r['drift'] for r in grp]):.1f}%")
        for r in grp:
            body.append([r["name"], st, str(nc), r["dominant"]]
                        + [f"{v:g}" for v in r["conc"]]
                        + [f"{v:.3f}" for v in r["true"]] + [f"{v:.3f}" for v in r["pred"]]
                        + [f"{r['drift']:.3f}", f"{100 * r['drift'] / worst:.2f}",
                           f"{worst:.3f}"])

head = (["mixture", "set", "n_components", "dominant_compound"]
        + [f"{s}_uM" for s in SUB] + [f"true_{s}_pct" for s in SUB]
        + [f"pred_{s}_pct" for s in SUB]
        + ["drift_pct", "drift_pct_of_group_worst", "group_worst_pct"])
with open(f"{OUT}/panel_f_drift.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(head); w.writerows(body)
dropped = len(rows) - len(body)
if dropped:
    print(f"  ⚠ {dropped}조건이 표에 안 들어갔다 — 확인할 것")
print(f"\n  ✓ panel_f_drift.csv   {len(body)}행 × {len(head)}열  → {OUT}")
print("그리기: set='high' 만 골라 n_components 로 두 패널, 막대 길이는 "
      "drift_pct_of_group_worst, 색은 dominant_compound.")
print("캡션: 4-fold 조건단위 held-out · blank 구성 (채택 구성 blank+sips 아님).")
