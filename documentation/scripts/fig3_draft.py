"""Figure 3 한 장 초안 — "경쟁 흡착이 신호를 결정한다 → 그래서 학습 역변환".

  (a) DFT 에너지 다이어그램 (DFT_results_summary.md Fig 3b 값)      [계산]
  (b) 이원 응답곡선 — 비율 축 (binary_response.npz, 반복 포함)       [실측]
  (c) 등몰 희석 시리즈 — 총농도 축 (FINDINGS §4, 2조제 평균)         [실측]
  (d) 귀결 문장 상자 — 신호≠용액 → duet net, 포화는 가드 → Fig 4

  초안이다 — 최종 외형은 Origin(FIGURE_REDO 워크플로). 주장 수위는
  DFT_BINARY_RESPONSE 2026-11-09 반복 조제 절을 따른다(부호·자릿수만).

  실행:  ../../.venv/bin/python3 -u fig3_draft.py  →  fig3_draft.png
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
COL = {"DQ": "#1f78c8", "TBZ": "#78c240", "THI": "#e3425f"}

# (a) CNF→금 이동 에너지 (eV): CNF 덮임(0) → 표면 여유 → 포화
STATES = ["CNF-covered", "< 50% coverage", "saturated"]
DFT = {"THI": [0, -2.40, -2.29], "DQ": [0, -1.82, -1.12], "TBZ": [0, +1.21, +0.09]}

# (c) 등몰 33/33/33 의 NNLS 신호 조성 (FINDINGS §4 · tert_nnls 2조제 평균, DQ/TBZ/THI)
EQ_UM = [10, 33.3, 100, 333.3]
EQ = np.array([[30.0, 21.0, 49.0], [18.1, 32.4, 49.4], [0, 0, 100], [0, 0, 100]])

fig = plt.figure(figsize=(12.6, 7.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.08], hspace=0.42, wspace=0.26)

# ── (a) DFT 다이어그램 ──────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
xpos = [0, 1, 2]
for s, v in DFT.items():
    ax.plot(xpos, v, "o-", color=COL[s], lw=2, ms=6, label=s)
    for x_, y_ in zip(xpos[1:], v[1:]):
        ax.annotate(f"{y_:+.2f}", (x_, y_), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color=COL[s])
ax.axhline(0, color="0.85", lw=0.8)
ax.set_xticks(xpos); ax.set_xticklabels(STATES, fontsize=9)
ax.set_ylabel("CNF→Au transfer energy (eV, aq)")
ax.set_title("a  DFT: site competition on the crowded surface", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=9)
ax.text(1.5, 0.8, "TBZ cannot displace CNF;\nDQ loses its lying site when crowded",
        fontsize=8, color="0.3", ha="center")

# ── (b) 이원 응답곡선 ────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
d = np.load(f"{HERE}/binary_response.npz", allow_pickle=True)
style = {"THI/DQ": (COL["THI"], "o"), "THI/TBZ": ("#a0522d", "s"), "DQ/TBZ": ("0.55", "^")}
gx = np.linspace(-2.3, 2.3, 300)
for name, x, f, x0, w in zip(d["pair"], d["x"], d["f"], d["x0"], d["w"]):
    c, m = style[str(name)]
    if str(name) != "DQ/TBZ":                       # 상쇄 쌍은 점만 (부호도 불안정)
        ax.plot(gx, 1 / (1 + np.exp(-(gx - x0) / w)), "-", color=c, lw=1.2, alpha=0.6)
    ax.plot(np.asarray(x, float), np.asarray(f, float), m, color=c, ms=5,
            mfc="none", mew=1.4, label=str(name))
ax.axvline(0, color="0.85", lw=0.8); ax.axhline(0.5, color="0.85", lw=0.8)
ax.set_xlabel("log10( [winner] / [other] )  in solution")
ax.set_ylabel("NNLS signal fraction")
ax.set_title("b  Binary series: the signal scale is bent ~1 decade", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.text(-2.2, 0.93, "at 1:1 the signal already reads 96–98% THI;\nDQ–TBZ: opposing factors cancel (no stable winner)",
        fontsize=8, color="0.3", va="top")

# ── (c) 등몰 희석 시리즈 ─────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
xb = np.arange(len(EQ_UM))
bot = np.zeros(len(EQ_UM))
for i, s in enumerate(["DQ", "TBZ", "THI"]):
    ax.bar(xb, EQ[:, i], 0.62, bottom=bot, color=COL[s], label=s)
    bot += EQ[:, i]
ax.set_xticks(xb); ax.set_xticklabels([f"{u:g}" for u in EQ_UM])
ax.set_xlabel("per-component concentration (µM), true composition 33:33:33 everywhere")
ax.set_ylabel("NNLS signal composition (%)")
ax.set_title("c  Equimolar 1:1:1: saturation hands the signal to THI", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
ax.axvspan(1.5, 3.5, color="0.92", zorder=0)
ax.text(2.5, 104, "saturated: composition unreadable → dilute", ha="center", fontsize=8, color="0.3")

# ── (d) 귀결 ────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1]); ax.set_axis_off()
ax.text(0.02, 0.95, "d  Consequence", fontsize=10, weight="bold", va="top")
ax.text(0.02, 0.80,
        "SERS reads the surface, not the solution:\n"
        "site competition (a) bends the signal scale by ~a decade (b)\n"
        "and hands saturated surfaces to THI entirely (c).\n\n"
        "→ linear unmixing (NNLS) is kept as the signal reader,\n"
        "   but solution composition needs the learned inversion\n"
        "   (physics-pretrained duet net) — demonstrated in Fig 4\n"
        "   on the “SERS” lettering map (87% THI signal → 1:1:1).\n\n"
        "→ where the signal is saturated (c, ≥100 µM equimolar)\n"
        "   no method can invert: report “saturated — dilute &\n"
        "   re-measure” instead of a composition.",
        fontsize=9.5, va="top", linespacing=1.5)

fig.suptitle("Figure 3 (draft) — Competitive adsorption decides the signal; "
             "recovery therefore needs a learned inversion", fontsize=12, y=0.99)
out = f"{HERE}/fig3_draft.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print("저장:", out)
