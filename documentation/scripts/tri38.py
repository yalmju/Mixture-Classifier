"""삼각형(ternary) + 조성 파이 — 저농도 35조건. 슬라이드 13·15의 저농도판.

  NNLS와 MLP를 **같은 특징·같은 조건**에서 비교한다 (curve38_feats.npz 재사용).
    NNLS  학습 없음, 앱과 같은 추정기 (_fit_predict("nnls"))
    MLP   retrain38.txt — 조건단위 4-fold held-out

  실행:  python3 -u tri38.py
"""
import os, sys, re, ast
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

import labfig
labfig.setup()
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from dl_model import _refs, _fit_predict
import triangle_figs as TF

DB = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB"
DESK = "/Users/seungki2/Desktop"
SUB = labfig.SUB
COLS = [labfig.CO[s] for s in SUB]

# ------------------------------------------------------------------ MLP (retrain38)
mlp = {}
for l in open(f"{DESK}/retrain64_blanksips.txt"):
    if not re.search(r"\|\s+[\d.]+% \|", l):
        continue
    p = [x.strip() for x in l.split("|")]
    key = tuple(int(float(v)) for v in p[0].split("/"))
    mlp[key] = (np.array([float(v) for v in p[1].split("/")]) / 100,
                np.array([float(v) for v in p[2].split("/")]) / 100)
print(f"MLP 조건 {len(mlp)}")

# ------------------------------------------------------------------ NNLS (같은 특징)
subs, wn, mask, P, lo, hi = _refs(f"{DB}/Pure", True, None)
z = np.load(f"{HERE}/curve38_feats.npz", allow_pickle=True)
X, Y, KEY = z["X"], z["Y"], z["KEY"].astype(str)
m = KEY != "HI"
pred = np.asarray(_fit_predict("nnls", X[m], Y[m], X[m], P_ref=P), float)
K, Yl = KEY[m], Y[m]
nnls = {}
for k in dict.fromkeys(K.tolist()):
    s = K == k
    p = pred[s].mean(0); p = p / p.sum()
    nnls[tuple(ast.literal_eval(k))] = (Yl[s][0].astype(float), p)
print(f"NNLS 조건 {len(nnls)}")

keys = sorted(set(mlp) & set(nnls))
rows_nnls = [(str(k), nnls[k][0], nnls[k][1]) for k in keys]
rows_mlp = [(str(k), mlp[k][0], mlp[k][1]) for k in keys]


def merr(rows):
    return float(np.mean([0.5 * np.abs(np.asarray(r[2]) - np.asarray(r[1])).sum() for r in rows]) * 100)


e_n, e_m = merr(rows_nnls), merr(rows_mlp)
print(f"공통 {len(keys)}조건 — NNLS {e_n:.1f}%  ·  MLP {e_m:.1f}%")

# ------------------------------------------------------------------ 그림 4: 삼각형 3panel
# 앱(page_validate._export_ternaries)과 같은 방식: 함수마다 **자기 크기의 독립 figure**를
# 만들게 두고, 투명 배경 · 600 dpi 로 각각 저장한다. subfigure 에 우겨넣으면 안에서 잡아둔
# 여백·글자 크기가 다 어긋난다.
# 삼각형은 **MLP 65조건 전부**로 그린다. NNLS 캐시(curve38_feats.npz)는 35조건 시절
# 것이라 교집합을 쓰면 격자의 절반만 그려진다.
all_keys = sorted(mlp)
rows_all = [(str(k), mlp[k][0], mlp[k][1]) for k in all_keys]
print(f"삼각형에 쓰는 조건 {len(rows_all)}  ·  평균 오차 {merr(rows_all):.1f}%")
for fn, name in ((TF.rgb_triangle, "fig4a_triangle_rgb64"),
                 (TF.accuracy_triangle, "fig4b_triangle_mlp64")):
    f = fn(rows_all, SUB, COLS)
    f.savefig(f"{DESK}/{name}.png", **labfig.SAVE)
    print(name + ".png")

# ------------------------------------------------------------------ 그림 5: 조성 파이
g32 = [k for k in mlp if k[0] in (12, 24) and k[1] in (3, 6, 12, 24) and k[2] in (3, 6, 12, 24)]
g32.sort(key=lambda k: (k[0], k[1], k[2]))
nr, nc = 4, 8
fig2, axes = plt.subplots(nr, nc, figsize=(14.6, 8.4))
for ax in axes.ravel():
    ax.axis("off")
INK = "#1c2430"
# 앱(page_real._plot_comp)과 같은 파이: 성분명 라벨 + 조각 안 % 숫자, 1% 미만은 뺀다.
for idx, k in enumerate(g32):
    ax = axes[idx // nc, idx % nc]; ax.axis("off")
    t, p = mlp[k]
    keep = [i for i in range(len(SUB)) if p[i] >= 0.01] or [int(np.argmax(p))]
    ax.pie([p[i] for i in keep], labels=[SUB[i] for i in keep],
           colors=[COLS[i] for i in keep], autopct="%1.0f%%",
           textprops={"fontsize": 6.5, "color": INK})
    ax.set_aspect("equal")
    ax.set_title(f"{k[0]}/{k[1]}/{k[2]} μM\ntrue {'/'.join(f'{v*100:.0f}' for v in t)}",
                 fontsize=7.5, pad=2)
fig2.suptitle("Predicted composition (%) per condition — MLP, leave-one-condition-out",
              fontsize=11, fontweight="bold")
fig2.tight_layout(rect=[0, 0.01, 1, 0.945])
fig2.savefig(f"{DESK}/fig5_pies64.png", **labfig.SAVE)
print("fig5_pies64.png")
