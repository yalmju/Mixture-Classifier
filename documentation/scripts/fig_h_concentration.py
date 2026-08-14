"""패널 h — 성분별 농도 회수 (파리티 3개 + 배수오차 누적분포).

  **CSV 만 읽는다.** `export_origin.py` 가 낸 `concentration_parity.csv` ·
  `concentration_fold_cdf.csv` 를 그대로 쓰므로, 여기서 나오는 그림과 Origin 으로
  다시 그린 그림이 같은 숫자를 본다는 게 보장된다. 붙어 있던 옛 패널이 35조건 시절
  값(2배 이내 83%)이었던 것 같은 사고를 막으려는 것이다.

  거르는 기준은 세 패널이 전부 같다 — `in_design=1`(THI 48 µM 은 설계 밖) 이고
  `valid=1`. 예전에는 CDF 만 valid 를 안 걸러서 곡선과 옆의 "2배 이내 %" 가 서로 다른
  n 에서 나왔다.

  실행:  python3 -u fig_h_concentration.py [origin_data 폴더]
"""
import os as _os, sys as _sys                 # paths 부트스트랩 — 기계마다 마운트가 다르다
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import os, sys, csv, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import labfig
labfig.setup()
import matplotlib.pyplot as plt

DB = paths.DB
SRC = sys.argv[1] if len(sys.argv) > 1 else f"{paths.INTERP}/origin_data"
OUT = os.path.dirname(SRC)
CO, SUB = labfig.CO, labfig.SUB
BAND = "#dfe4ea"
GUIDE = "#9aa3ad"


def rows(name):
    with open(os.path.join(SRC, name), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


par = [r for r in rows("concentration_parity.csv")
       if r["in_design"] == "1" and r["valid"] == "1"]
cdf = rows("concentration_fold_cdf.csv")
print(f"{len(par)}행 (성분별 " + " · ".join(
    f"{s} {sum(1 for r in par if r['compound'] == s)}" for s in SUB) + ")")

fig, ax = plt.subplots(1, 4, figsize=(14.6, 3.9))

# ---------------------------------------------------------------- 파리티 3개 (로그축)
# 로그축으로 간다: 참값이 3–24 µM 로 8배 범위인데 추정은 100 µM 을 넘는 것이 있어,
# 선형축이면 저농도 세 수준이 왼쪽 끝에 뭉쳐 DQ 의 바닥값이 안 보인다.
LO, HI = 1.0, 200.0
for i, s in enumerate(SUB):
    A = ax[i]
    d = [r for r in par if r["compound"] == s]
    t = np.array([float(r["true_uM"]) for r in d])
    e = np.array([float(r["estimated_uM"]) for r in d])
    f = np.array([float(r["fold_error"]) for r in d])
    xx = np.array([LO, HI])
    A.fill_between(xx, xx / 2, xx * 2, color=BAND, alpha=.55, lw=0, zorder=1,
                   label="2× band")
    A.plot(xx, xx, ls=(0, (4, 3)), lw=.8, color=GUIDE, zorder=2, label="1:1")
    # 같은 참농도에 점이 겹치므로 x 를 로그 공간에서 살짝 흩는다 (값은 그대로)
    j = np.exp(np.random.default_rng(i).uniform(-.06, .06, t.size))
    A.scatter(t * j, e, s=30, c=CO[s], alpha=.85, edgecolor="white", lw=.5, zorder=3)
    A.set_xscale("log"); A.set_yscale("log")
    A.set_xlim(2, 34); A.set_ylim(LO, HI)
    A.set_xticks([3, 6, 12, 24]); A.set_xticklabels(["3", "6", "12", "24"])
    A.set_title(s, color=CO[s])
    A.set_xlabel("true (μM)")
    if i == 0:
        A.set_ylabel("estimated (μM)")
    A.text(.05, .95, f"median {np.median(f):.2f}×\nwithin 2× {100 * (f < 2).mean():.0f}%\n"
                     f"n {len(d)}",
           transform=A.transAxes, va="top", ha="left", fontsize=8.5)
    if i == 0:
        A.legend(loc="lower right", fontsize=8)

# ---------------------------------------------------------------- 배수오차 누적분포
A = ax[3]
for s in SUB:
    d = [r for r in cdf if r["compound"] == s]
    x = [float(r["fold_error"]) for r in d]
    y = [float(r["cumulative_pct"]) for r in d]
    A.step(x, y, where="post", lw=1.8, color=CO[s], label=s)
A.axvline(2, ls=(0, (4, 3)), lw=.8, color=GUIDE)
A.text(2.08, 10, "2×", fontsize=8.5, color="#5b6673")
A.set_xlim(1, 6); A.set_ylim(0, 100)
A.set_xlabel("fold error (×)"); A.set_ylabel("cumulative (%)")
A.set_title("How often within a given fold")
A.legend(loc="lower right")

fig.suptitle("Concentration — per-compound recovery  (64-condition grid, "
             "leave-one-condition-out; THI 48 μM out-of-design excluded)",
             fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .93])
fig.savefig(f"{OUT}/figh_concentration64.png", dpi=600, transparent=True,
            bbox_inches="tight", pad_inches=0.01)
print(f"figh_concentration64.png → {OUT}")

for s in SUB:
    f = np.array([float(r["fold_error"]) for r in par if r["compound"] == s])
    print(f"  {s:<4} median {np.median(f):.2f}×  ·  2배 이내 {100 * (f < 2).mean():.0f}%  "
          f"·  3배 이내 {100 * (f < 3).mean():.0f}%  ·  n {len(f)}")
