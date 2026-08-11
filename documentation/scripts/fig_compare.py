"""세 모델의 조성 예측을 한 장으로: 기존(이중보정) · 기존(보정수정) · 저농도 재학습"""
import sys, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

COMP = {1: (6, 6, 24), 2: (12, 12, 12), 3: (24, 6, 6),
        4: (6, 6, 6), 5: (24, 24, 24), 6: (12, 12, 48)}
SUB = ["DQ", "TBZ", "THI"]
COL = {"DQ": "#2980b9", "TBZ": "#27ae60", "THI": "#c0392b"}

# 앞선 실행에서 얻은 값들 (260806 모델)
DOUBLE = {1: (5.2, 60.9, 33.9), 2: (9.6, 79.3, 11.1), 3: (55.9, 37.3, 6.8),
          4: (22.1, 58.3, 19.5), 5: (1.0, 87.8, 11.2), 6: (1.8, 72.9, 25.2)}
SINGLE = {1: (10.7, 41.2, 48.1), 2: (12.4, 60.2, 27.5), 3: (64.7, 20.7, 14.6),
          4: (25.1, 43.6, 31.3), 5: (2.9, 78.4, 18.7), 6: (4.9, 58.8, 36.4)}

RETRAIN = {}
log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrain_log.txt")
for ln in open(log, encoding="utf-8"):
    m = re.match(r"\s*#(\d)\s+\S+\s*\|.*\|\s*([\d.]+)/\s*([\d.]+)/\s*([\d.]+)\s*\|", ln)
    if m:
        RETRAIN[int(m.group(1))] = tuple(float(m.group(i)) for i in (2, 3, 4))
print("재학습 결과 파싱:", sorted(RETRAIN))

panels = [("기존 모델 — ALS 이중보정\n(지금 앱 상태)", DOUBLE),
          ("기존 모델 — 전처리 수정\n(baseline 끄기)", SINGLE),
          ("저농도 포함 재학습\n(고농도 32 + 저농도 15맵)", RETRAIN)]
panels = [(t, d) for t, d in panels if d]

fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 5.6), sharey=True)
if len(panels) == 1:
    axes = [axes]
x = np.arange(len(COMP))
w = 0.26
for ax, (title, D) in zip(axes, panels):
    errs = []
    for j, s in enumerate(SUB):
        true = [COMP[n][j] / sum(COMP[n]) * 100 for n in sorted(COMP)]
        pred = [D[n][j] if n in D else np.nan for n in sorted(COMP)]
        ax.bar(x + (j - 1) * w, pred, w, color=COL[s], label=s, zorder=3)
        ax.scatter(x + (j - 1) * w, true, marker="_", s=260, color="black",
                   linewidths=2.2, zorder=4, label="참값" if j == 0 else None)
    for n in sorted(COMP):
        if n in D:
            t = np.array([COMP[n][k] / sum(COMP[n]) * 100 for k in range(3)])
            errs.append(np.abs(np.array(D[n]) - t).sum() / 2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{n}\n{'/'.join(str(v) for v in COMP[n])}" for n in sorted(COMP)],
                       fontsize=9)
    ax.set_title(title + f"\n평균 조성오차 {np.mean(errs):.1f}%", fontsize=11)
    ax.grid(axis="y", alpha=.25, zorder=0)
    ax.set_ylim(0, 100)
axes[0].set_ylabel("조성 (%)   가로선 = 참값")
axes[0].legend(fontsize=9, loc="upper left")
fig.suptitle("저농도 6조건 × 3반복 — 조성 예측 비교", fontsize=13)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_compare.png")
fig.savefig(out, dpi=160)
print("saved", out)
