"""단일성분 등온선 — 성분마다 **따로** 그림 파일 하나. 논문 figure 용.

  검량 희석계열(0.1–1000 µM, 9점)에서 NNLS 계수 B 를 뽑아 세 모델을 적합한다.

    Langmuir     B = gA·KC/(1+KC)                    자리 하나, 상호작용 없음
    Sips (L-F)   B = gA·(KC)^m/(1+(KC)^m)            K 는 Langmuir 값으로 고정, m 만 자유
    Freundlich   B = a·C^(1/n)                       포화 없음

  Sips 의 K 를 고정하는 이유: 셋 다 자유로 두면 DQ 가 K=2e-8, gA=7e8 로 퇴화한다.
  자기 곡선은 재현하지만 다른 성분과 축이 안 맞아 경쟁 분모에서 무의미해진다.

  위    전체 희석계열 (x 로그 — 4자릿수라 선형으로는 저농도가 안 보인다)
  아래  혼합물 작업구간 0–30 µM (x 선형)

  **그래프 안에는 아무 글자도 넣지 않는다.** 범례는 그림 밖 아래로 빼고, K·m·R² 는
  콘솔 표와 캡션으로 간다. 개별 파일(iso_DQ/TBZ/THI.png)과 합본(iso_all.png) 둘 다 낸다.

  실행:  python3 -u isotherm_figs.py
"""
import os, sys
import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/github/Mixture Classifier"
DB = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB"
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

import labfig
labfig.setup()
import matplotlib.pyplot as plt

from io_utils import load_calibration_csv
from calibration import fit_B, fit_isotherm, fit_sips
from dl_model import _refs

DESK = "/Users/seungki2/Desktop"
CO, SUB = labfig.CO, labfig.SUB
WORK = (3.0, 24.0)                      # 혼합물 작업 구간 (µM/성분)

subs, wn, mask, P, lo, hi = _refs(f"{DB}/Pure", True, None)
ax_c, names, dils = load_calibration_csv(f"{DB}/Ratio/results/calibration_spectra.csv")
mc = (ax_c >= lo) & (ax_c <= hi)


def freundlich(C, a, inv_n):
    return a * C ** inv_n


def r2_of(B, pred):
    ss = ((B - B.mean()) ** 2).sum()
    return 1 - ((B - pred) ** 2).sum() / ss if ss > 0 else np.nan


print(f"{'성분':>5} | {'K (1/M)':>10} | {'1/K (µM)':>9} | {'gA':>10} | {'m (Sips)':>8} | "
      f"{'1/n (Fr)':>8} | {'R² L / S / F':>20}")
print("-" * 92)
rows = {}
for i, s in enumerate(SUB):
    j = names.index(s)
    C = np.asarray(dils[j][0], float)
    Y = np.asarray(dils[j][1], float)[:, mc]
    B = np.array([fit_B(y, P)[0][i] for y in Y])
    o = np.argsort(C); C, B = C[o], B[o]

    gA, K = fit_isotherm(C, B)
    gA_s, K_s, m = fit_sips(C, B)                     # K 고정
    try:
        pf, _ = curve_fit(freundlich, C, B, p0=[B.max(), 0.5],
                          bounds=([0, 0.05], [np.inf, 3.0]), maxfev=40000)
    except Exception:
        pf = [np.nan, np.nan]

    uL = K * C
    r2L = r2_of(B, gA * uL / (1 + uL))
    uS = np.power(K_s * C, m)
    r2S = r2_of(B, gA_s * uS / (1 + uS))
    r2F = r2_of(B, freundlich(C, *pf))
    rows[s] = dict(C=C, B=B, gA=gA, K=K, gA_s=gA_s, m=m, pf=pf,
                   r2=(r2L, r2S, r2F))
    print(f"{s:>5} | {K:10.3g} | {1e6/K:9.1f} | {gA:10.3g} | {m:8.3f} | {pf[1]:8.3f} | "
          f"{r2L:6.3f} / {r2S:5.3f} / {r2F:5.3f}")

# ---------------------------------------------------------------- 그리기
def draw(A, s, kind):
    """패널 하나. 내부에는 선·점·음영만 — 글자·범례·상자는 넣지 않는다."""
    d = rows[s]; C, B = d["C"], d["B"]
    if kind == "full":
        xs = np.logspace(np.log10(C.min() * 0.7), np.log10(C.max() * 1.4), 400)
    else:
        xs = np.linspace(0, 30e-6, 400)
    uL = d["K"] * xs
    uS = np.power(d["K"] * xs, d["m"])
    A.axvspan(WORK[0], WORK[1], color="#dfe4ea", alpha=.5, lw=0, zorder=0)
    h1, = A.plot(xs * 1e6, d["gA"] * uL / (1 + uL), lw=1.6, color=CO[s])
    h2, = A.plot(xs * 1e6, d["gA_s"] * uS / (1 + uS), lw=1.4, ls=(0, (5, 2)), color="#5b6673")
    h3, = A.plot(xs * 1e6, freundlich(xs, *d["pf"]), lw=1.2, ls=(0, (1.5, 1.5)), color="#9aa3ad")
    h4 = A.scatter(C * 1e6, B, s=38, c=CO[s], edgecolor="white", lw=.6, zorder=5)
    if kind == "full":
        A.set_xscale("log")
    else:
        A.set_xlim(0, 30); A.set_ylim(0, None)
    A.set_xlabel("concentration (μM)")
    return [h4, h1, h2, h3]


LBL = ["measured", "Langmuir", "Sips", "Freundlich"]

# 개별 파일 — 성분마다 위/아래 두 패널
for s in SUB:
    fig, ax = plt.subplots(2, 1, figsize=(4.2, 6.2))
    hs = draw(ax[0], s, "full")
    draw(ax[1], s, "work")
    ax[0].set_ylabel("NNLS coefficient  B")
    ax[1].set_ylabel("NNLS coefficient  B")
    ax[0].set_title(s, color=CO[s])
    fig.legend(hs, LBL, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(f"{DESK}/iso_{s}.png", **labfig.SAVE); plt.close(fig)
    print("saved", f"iso_{s}.png")

# 합본 — 열 = 성분, 위 전체계열 / 아래 작업구간
fig, ax = plt.subplots(2, 3, figsize=(11.4, 6.4))
for k, s in enumerate(SUB):
    hs = draw(ax[0, k], s, "full")
    draw(ax[1, k], s, "work")
    ax[0, k].set_title(s, color=CO[s])
    if k == 0:
        ax[0, k].set_ylabel("NNLS coefficient  B")
        ax[1, k].set_ylabel("NNLS coefficient  B")
fig.legend(hs, LBL, loc="lower center", ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(f"{DESK}/iso_all.png", **labfig.SAVE); plt.close(fig)
print("saved iso_all.png")

# 겹쳐 그린 판 — 성분 비교용. 모델은 Langmuir 하나만 (셋×셋이면 9줄이라 못 읽는다).
fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.8))
hs, lbl = [], []
for s in SUB:
    d = rows[s]
    for pi, A in enumerate(ax):
        xs = (np.logspace(np.log10(d["C"].min() * .7), np.log10(d["C"].max() * 1.4), 400)
              if pi == 0 else np.linspace(0, 30e-6, 400))
        u = d["K"] * xs
        A.plot(xs * 1e6, d["gA"] * u / (1 + u), lw=1.6, color=CO[s])
        h = A.scatter(d["C"] * 1e6, d["B"], s=38, c=CO[s], edgecolor="white", lw=.6, zorder=5)
        if pi == 0:
            hs.append(h); lbl.append(s)
for pi, A in enumerate(ax):
    A.axvspan(WORK[0], WORK[1], color="#dfe4ea", alpha=.5, lw=0, zorder=0)
    A.set_xlabel("concentration (μM)")
    if pi == 0:
        A.set_xscale("log"); A.set_ylabel("NNLS coefficient  B")
    else:
        A.set_xlim(0, 30); A.set_ylim(0, None)
fig.legend(hs, lbl, loc="lower center", ncol=3, frameon=False, fontsize=9)
fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig(f"{DESK}/iso_overlay.png", **labfig.SAVE); plt.close(fig)
print("saved iso_overlay.png")
