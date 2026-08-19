"""랩미팅 그림 — 조성·농도 성분별 복원. cnsplots 형식 + Pure/colors.json 색.

  조성  retrain64.txt              MLP, 조건단위 4-fold, 64조건 완전격자
  농도  conc38_result_cliponly.npz  s = g·r·C 두방향 적합, 조건단위 LOO
                                  전처리는 clip-only (`_corrected` 에 ALS 재적용 안 함)

  실행:  python3 -u labfigs_make.py
"""
import re, os, sys
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import labfig
labfig.setup()
import matplotlib.pyplot as plt

CO = labfig.CO
SUB = labfig.SUB
DESK = "/Users/seungki2/Desktop"
SRC = f"{DESK}/retrain64_blanksips.txt"


# ------------------------------------------------------------------ 조성 데이터
def load_comp(grid_only=True):
    L = [l for l in open(SRC, encoding="utf-8") if re.search(r"\|\s+[\d.]+% \|", l)]
    C, T, P = [], [], []
    for l in L:
        p = [x.strip() for x in l.split("|")]
        c = [int(float(v)) for v in p[0].split("/")]
        if grid_only and not all(v in (3, 6, 12, 24) for v in c):
            continue
        C.append(c); T.append([float(v) for v in p[1].split("/")])
        P.append([float(v) for v in p[2].split("/")])
    return np.array(C), np.array(T), np.array(P)


# ------------------------------------------------------------------ 그림 1: 조성
Cc, T, Pr = load_comp()
fig, ax = plt.subplots(2, 3, figsize=(11.4, 7.2))
for i, s in enumerate(SUB):
    t, p = T[:, i], Pr[:, i]
    a, b = np.polyfit(t, p, 1)
    r2 = 1 - ((p - (a * t + b)) ** 2).sum() / ((p - p.mean()) ** 2).sum()
    rho = spearmanr(t, p).correlation
    A = ax[0, i]
    labfig.unity(A, 0, 85)
    xx = np.linspace(0, 85, 10)
    A.plot(xx, a * xx + b, lw=1.4, color=CO[s], zorder=2)
    A.scatter(t, p, s=34, c=CO[s], alpha=.85, edgecolor="white", lw=.5, zorder=3)
    A.set_title(s, color=CO[s])
    A.set_xlabel("true composition (%)")
    if i == 0:
        A.set_ylabel("predicted (%)")
    A.text(.05, .95, f"slope {a:.2f}\n$R^2$ {r2:.2f}\n$\\rho$ {rho:.2f}\nbias {p.mean()-t.mean():+.1f}%p",
           transform=A.transAxes, va="top", ha="left", fontsize=8.5)

    # x는 실제 µM 눈금(선형)이다. 3/6/12/24를 등간격 칸으로 놓으면 로그축처럼 보이고,
    # 농도 간격이 실제로 2배씩 벌어진다는 사실이 그림에서 사라진다.
    B = ax[1, i]
    lvs = sorted(set(Cc[:, i]))
    tm = [T[Cc[:, i] == v, i].mean() for v in lvs]
    pm = [Pr[Cc[:, i] == v, i].mean() for v in lvs]
    sd = [Pr[Cc[:, i] == v, i].std() for v in lvs]
    xs = np.asarray(lvs, float)
    B.plot(xs, tm, ls=(0, (4, 3)), marker="o", ms=5, lw=.9, color="#5b6673", label="true", zorder=2)
    B.errorbar(xs, pm, yerr=sd, fmt="o-", color=CO[s], ms=6, lw=1.6, capsize=3,
               label="predicted", zorder=3)
    B.set_xticks([3, 6, 12, 24]); B.set_xticklabels(["3", "6", "12", "24"])
    B.set_xlim(0, 27)
    B.set_xlabel(f"{s} (μM)"); B.set_ylim(0, 62)
    if i == 0:
        B.set_ylabel("composition (%)")
    B.legend(loc="upper left")
fig.suptitle("Composition — per-compound recovery  (64-condition grid, "
             "leave-one-condition-out)", fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .955])
fig.savefig(f"{DESK}/fig1_composition.png", **labfig.SAVE); plt.close(fig)
print("fig1_composition.png")

# ------------------------------------------------------------------ 그림 2: 농도
d = np.load(f"{HERE}/conc38_result_cliponly.npz", allow_pickle=True)
S, Cm, F = d["S"], d["C"], d["fold"]
paths = d["paths"]
# held-out 추정치를 그대로 쓴다. 전체적합 추정치를 그리면 산점도만 좋아 보이고
# 옆에 적힌 LOO 통계와 어긋난다 — 그림과 숫자는 같은 프로토콜이어야 한다.
EST = d["est"]
design = np.array([48 not in Cm[m] for m in range(len(Cm))])   # THI 48 = 설계 밖

fig, ax = plt.subplots(1, 4, figsize=(14.6, 3.9))
for i, s in enumerate(SUB):
    A = ax[i]
    ok = design & (S[:, i] > 0)
    t, e = Cm[ok, i], EST[ok, i]
    # 선형 축. 로그로 그리면 2배 밴드가 일정 폭 띠로 예뻐 보이지만, 실제 µM 간격과
    # 고농도 쪽 산포 크기가 눌려서 안 보인다.
    xx = np.array([0, 30.0])
    A.fill_between(xx, xx / 2, xx * 2, color="#dfe4ea", alpha=.55, lw=0, zorder=1,
                   label="2× band")
    A.plot(xx, xx, ls=(0, (4, 3)), lw=.8, color="#9aa3ad", zorder=2, label="1:1")
    A.scatter(t + np.random.default_rng(i).uniform(-.6, .6, t.size), e,
              s=34, c=CO[s], alpha=.85, edgecolor="white", lw=.5, zorder=3)
    A.set_xlim(0, 27); A.set_ylim(0, 50)
    A.set_xticks([3, 6, 12, 24]); A.set_xticklabels(["3", "6", "12", "24"])
    A.set_title(s, color=CO[s])
    A.set_xlabel("true (μM)")
    if i == 0:
        A.set_ylabel("estimated (μM)")
    f = F[ok, i]
    A.text(.05, .95, f"median {np.median(f):.2f}×\nwithin 2× {100*(f<2).mean():.0f}%",
           transform=A.transAxes, va="top", ha="left", fontsize=8.5)
    if i == 0:
        A.legend(loc="upper left", bbox_to_anchor=(0.0, 0.80), fontsize=8)

A = ax[3]
for i, s in enumerate(SUB):
    ok = design & np.isfinite(F[:, i])
    f = np.sort(F[ok, i])
    A.step(f, np.arange(1, f.size + 1) / f.size * 100, where="post", lw=1.8, color=CO[s], label=s)
A.axvline(2, ls=(0, (4, 3)), lw=.8, color="#9aa3ad")
A.text(2.08, 12, "2×", fontsize=8.5, color="#5b6673")
A.set_xlim(1, 6); A.set_ylim(0, 100)
A.set_xlabel("fold error (×)"); A.set_ylabel("cumulative (%)")
A.set_title("How often within a given fold")
A.legend(loc="lower right")
fig.suptitle("Concentration — per-compound recovery  (64-condition grid, "
             "leave-one-condition-out; THI 48 μM out-of-design excluded)",
             fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .93])
fig.savefig(f"{DESK}/fig2_concentration.png", **labfig.SAVE); plt.close(fig)
print("fig2_concentration.png")
