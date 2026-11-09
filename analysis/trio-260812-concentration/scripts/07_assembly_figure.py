"""3단 조립 그림 — "픽셀이 분류한다 → 신호가 읽는 건 표면 → 학습이 용액으로 되돌린다".

  글씨("SERS") 맵 하나로 메시지 전체를 보인다:
    A  픽셀 단위 조성(RGB 혼색) — 글씨가 공간적으로 분해된다
    B  맵 수준 NNLS 표면 조성 — THI 쪽으로 무너진 신호 + 왜(교차점 11×/5×)
    C  혼합 학습 MLP 의 용액 조성 vs 분주 참값 33/33/33 + 절대 µM

  A·B 는 여기서 재계산한다 (01 과 같은 설정: nnls · baseline off · trim 500–2500 ·
  min_frac 0.20 · hit threshold). C 의 수치는 확정 문서값이다 — 조성 36.5/30.1/33.5
  (README 표, 260806 혼합 학습 모델), µM 10.1±1.9 / 2.7±0.5 / 14.2±2.8 (05 질량수지).
  재계산이 아니므로 모델·스크립트가 바뀌면 여기 상수도 같이 바꿀 것.

  실행:  ../../../.venv/bin/python3 -u 07_assembly_figure.py
  출력:  ../figures/Fig_assembly.png
"""
import os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))                      # 저장소 루트 (unmix)
sys.path.insert(0, str(HERE.parents[2] / "documentation" / "scripts"))  # paths
import paths
import numpy as np

from unmix import unmix_map

DATA = f"{paths.DB}/Reference"          # Reference 는 재편 후에도 DB 루트에 있다
TEST = f"{paths.DATA}/Pest/260812_12 trio_THI_TBZ_DQ.csv"
CAL = f"{paths.DATA}/Pest/Standard/260729/calibration_spectra.csv"
FIG = HERE.parent / "figures" / "Fig_assembly.png"

SUB = ["DQ", "TBZ", "THI"]
COL = {"DQ": (0.12, 0.47, 0.81), "TBZ": (0.47, 0.76, 0.25), "THI": (0.89, 0.26, 0.38)}

# ── 확정 문서값 (README · FINDINGS §5) ──────────────────────────────────────
MLP_COMP = {"DQ": 36.5, "TBZ": 30.1, "THI": 33.5}       # 260806 혼합 학습 모델
TRUE_COMP = 100 / 3
UM = {"DQ": (10.1, 1.9), "TBZ": (2.7, 0.5), "THI": (14.2, 2.8)}   # 05 질량수지
UM_TRUE = 12.0
X0 = {"THI/DQ": -1.05, "THI/TBZ": -0.71}                 # binary_response.npz

CACHE = HERE.parent / "out" / "assembly_cache.npz"
if CACHE.exists():
    d = np.load(CACHE, allow_pickle=True)
    coords, ratio, hit, mean_ratio, n_px = d["coords"], d["ratio"], d["hit"], d["mean_ratio"], int(d["n_px"])
else:
    print("unmix_map …")
    r = unmix_map(data_dir=DATA, test_path=TEST, method="nnls", baseline=False,
                  trim=(500, 2500), min_frac=0.20, hit_mode="threshold",
                  calib_path=None, progress=lambda m: None)
    nb = [r.comps[i] for i in r.nonbg]
    order = [nb.index(s) for s in SUB]
    coords, ratio, hit = r.coords, r.ratio_nb[:, order], r.hit
    mean_ratio, n_px = r.mean_ratio[order], r.n_pixels
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, coords=coords, ratio=ratio, hit=hit, mean_ratio=mean_ratio, n_px=n_px)
surf = {s: 100 * mean_ratio[i] for i, s in enumerate(SUB)}
print(f"hit {int(hit.sum())}/{n_px} px · NNLS 표면 조성 "
      + " / ".join(f"{s} {surf[s]:.1f}%" for s in SUB))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(13.2, 4.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.28)

# ── A: 픽셀 혼색 지도 (앱과 같은 격자 imshow — 260812_SERS PNG 와 방향 일치) ──
ax = fig.add_subplot(gs[0])
x, y = coords[:, 0], coords[:, 1]
ux, uy = np.unique(x), np.unique(y)
xi = {v: i for i, v in enumerate(ux)}; yi = {v: i for i, v in enumerate(uy)}
rgb = np.ones((len(coords), 3)) * 0.92                   # 배경(비-hit) 연회색
base = np.array([COL[s] for s in SUB])                    # (3, 3)
rgb[hit] = np.clip(ratio[hit] @ base, 0, 1)
img = np.ones((len(uy), len(ux), 3)) * 0.92
img[[yi[v] for v in y], [xi[v] for v in x]] = rgb
if os.environ.get("ASSEMBLY_FLIP", "lower") == "upper":   # 방향은 기준 PNG 에 맞춘다
    img = img[::-1]
ax.imshow(img, origin="lower", aspect="equal", interpolation="nearest")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("A  Per-pixel composition (NNLS gate + read)\n"
             f"hit {int(hit.sum())}/{n_px} px", fontsize=10)
for i, s in enumerate(SUB):
    ax.scatter([], [], c=[COL[s]], s=40, marker="s", label=s)
ax.legend(loc="lower right", frameon=False, fontsize=8, ncol=3)

# ── B: 표면(신호) 조성 + 왜 표면≠용액 ────────────────────────────────────────
ax = fig.add_subplot(gs[1])
left = 0.0
for s in SUB:
    ax.barh(1.0, surf[s], left=left, color=COL[s], height=0.45)
    if surf[s] > 6:
        ax.text(left + surf[s] / 2, 1.0, f"{s}\n{surf[s]:.0f}%",
                ha="center", va="center", fontsize=9, color="w", weight="bold")
    left += surf[s]
ax.set_xlim(0, 100); ax.set_ylim(0.3, 1.7)
ax.set_yticks([]); ax.set_xlabel("NNLS signal (= surface) composition, %")
ax.set_title("B  The signal reads the surface,\nnot the solution", fontsize=10)
ax.text(50, 0.52,
        "competitive adsorption bends the scale:\n"
        f"50:50 signal needs DQ {10**-X0['THI/DQ']:.0f}x or TBZ {10**-X0['THI/TBZ']:.0f}x\n"
        "excess over THI (binary response curves + DFT)",
        ha="center", va="center", fontsize=8.5, color="0.25",
        bbox=dict(boxstyle="round,pad=0.45", fc="0.96", ec="0.8"))

# ── C: 학습 모델의 용액 조성 복원 + 절대 µM ──────────────────────────────────
ax = fig.add_subplot(gs[2])
xs = np.arange(3)
ax.bar(xs - 0.19, [TRUE_COMP] * 3, width=0.36, color="0.82", label="dispensed 33.3%")
ax.bar(xs + 0.19, [MLP_COMP[s] for s in SUB], width=0.36,
       color=[COL[s] for s in SUB], label="mixture-trained MLP")
for i, s in enumerate(SUB):
    um, se = UM[s]
    note = f"{um:.1f}±{se:.1f} µM" + ("*" if s == "TBZ" else "")
    ax.text(i + 0.19, MLP_COMP[s] + 1.2, note, ha="center", fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels(SUB)
ax.set_ylim(0, 46); ax.set_ylabel("solution composition, %")
ax.set_title("C  Learning inverts the surface back\nto the dispensed solution", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.text(1.0, 2.5, "*SERS-effective surface loading (12 µM dispensed)",
        ha="center", fontsize=7.2, color="0.35")

fig.suptitle('One "SERS" lettering map (12 µM each, 1:1:1 ink): '
             "pixels classify → signal reads the surface → learning recovers the solution",
             fontsize=11, y=1.02)
fig.savefig(FIG, dpi=180, bbox_inches="tight")
print(f"저장: {FIG}")
