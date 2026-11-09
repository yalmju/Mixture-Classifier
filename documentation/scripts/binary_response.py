"""이원 비율 계열의 NNLS 신호 응답곡선 — DFT 경쟁 흡착의 실험 짝.

  고농도 `high` 세트에서 한 성분이 0인 맵(이원)을 쌍별로 모아,
  **용액 비율 log10(c_A/c_B) vs NNLS 신호 분율 f_A** 곡선을 그린다.

  왜 NNLS 인가 — duet net 이 아니라:
    · DFT 가 예측하는 것은 표면(신호) 조성이다. 학습 모델은 용액 조성을
      되돌리도록 훈련돼 있어, 잘 작동할수록 경쟁 흡착의 흔적을 지운다.
    · duet net 은 Sips 등온선으로 사전학습을 받는다. 그걸로 흡착 물리를
      검증하면 순환이다. NNLS 는 학습·사전확률이 없는 선형 투영이라
      곡선의 휘어짐이 전부 물리에서 왔다고 말할 수 있다.
  판독 경로는 `tert_nnls.py`·`export_nnls.py` 와 같다 (`surface_composition`,
  맵당 평균 스펙트럼 한 줄) — 패널 a·c 의 NNLS 와 같은 것을 뜻해야 한다.

  곡선마다 로지스틱 f_A = 1/(1+exp(−(x−x0)/w)) 을 맞춰 **교차점 x0**
  (신호가 50:50 이 되는 용액 비율)를 뽑는다. 눈금이 안 휘었으면 x0=0 이다.
  DFT 서열(흡착 THI > DQ > TBZ · 분자당 신호 TBZ 최대)과 부호·순서를 대조한다.

  나오는 것:  documentation/scripts/binary_response.npz
                pair      (k,)   "THI/DQ" 꼴 — 분자 쪽이 f 의 기준 성분 A
                x         object 쌍별 log10(c_A/c_B)
                f         object 쌍별 NNLS 신호 분율 f_A (쌍 2성분으로 재정규화)
                total     object 쌍별 총농도 µM (비율 축과 얽혀 있음 — 캡션에 명기)
                x0, w     (k,)   로지스틱 교차점·폭 (log10 단위)
              documentation/scripts/binary_response.png  (3패널 요약)

  실행:  python3 -u binary_response.py            # NNLS 계산 + npz + png
         python3 -u binary_response.py --replot   # npz 에서 그림만 다시
         python3 -u binary_response.py --export   # Origin 용 csv → INTERP/panels/
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import os, sys, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, paths.REPO)

import mixtures as MX
from dl_model import _refs, _map_spectra, _composition_features
from dl_quantify import surface_composition
from real_data import load_map

SUB = MX.SUB                                   # ["DQ", "TBZ", "THI"]
OUT = f"{HERE}/binary_response"

# 쌍의 방향: A 가 DFT 흡착 서열에서 이기는 쪽 (THI > DQ > TBZ).
# 눈금이 흡착 경쟁으로 휜다면 x0 < 0 (A 가 용액에서 적어도 신호 50% — A 쪽으로 유리).
PAIRS = [("THI", "DQ"), ("THI", "TBZ"), ("DQ", "TBZ")]

subs, wn, mask, P, lo, hi = _refs(paths.PURE, True, None)
idx = {s: subs.index(s) for s in SUB}


def nnls_fraction(path):
    """맵 하나 → 세 성분 NNLS 신호 조성 (tert_nnls.py 와 같은 판독)."""
    _w, cube, _m, _c = load_map(path)
    X = np.array([_composition_features(ya) for ya in _map_spectra(cube, mask, 0)])
    raw = np.expm1(np.clip(X, 0, None))
    v = surface_composition(_composition_features(raw, "legacy_l2"), P).mean(0)
    a = np.array([v[idx[s]] for s in SUB], float)
    return a / (a.sum() or 1.0)


def logistic_fit(x, f):
    """f = 1/(1+exp(−(x−x0)/w)) 최소제곱. 점이 적으니 격자+정밀화로 충분하다."""
    from scipy.optimize import least_squares
    def r(p):
        x0, lw = p
        return 1.0 / (1.0 + np.exp(-(x - x0) / np.exp(lw))) - f
    best = None
    for x0 in np.linspace(-2, 2, 9):
        for lw in (np.log(0.3), np.log(0.7), np.log(1.5)):
            s = least_squares(r, [x0, lw])
            if best is None or s.cost < best.cost:
                best = s
    return best.x[0], float(np.exp(best.x[1]))


def replot():
    d = np.load(f"{OUT}.npz", allow_pickle=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(d["pair"])
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 3.8), sharey=True)
    gx = np.linspace(-2.3, 2.3, 300)
    sc = None
    for ax, name, x, f, t, x0, w in zip(np.atleast_1d(axes), names, d["x"], d["f"],
                                        d["total"], d["x0"], d["w"]):
        A, B = name.split("/")
        ax.axhline(0.5, color="0.85", lw=0.8)
        ax.axvline(0, color="0.85", lw=0.8)
        ax.plot(gx, 1 / (1 + np.exp(-(gx - x0) / w)), "-", color="0.4", lw=1.3)
        t = np.asarray(t, float); x = np.asarray(x, float); f = np.asarray(f, float)
        kn = np.isfinite(t)
        sc = ax.scatter(x[kn], f[kn], c=t[kn], cmap="viridis", vmin=0, vmax=1000,
                        s=42, edgecolor="k", lw=0.5, zorder=3)
        ax.scatter(x[~kn], f[~kn], marker="o", s=42, facecolor="none",
                   edgecolor="k", lw=1.1, zorder=3)   # 별도 조제 (총농도 불명)
        ax.axvline(x0, color="tab:red", ls=":", lw=1)
        ax.text(x0, 0.03, f" x0 = {x0:+.2f}", color="tab:red", fontsize=9)
        ax.set_xlabel(f"log10( [{A}] / [{B}] )  in solution")
        ax.set_title(f"{A} vs {B}   (w = {w:.2f})")
    np.atleast_1d(axes)[0].set_ylabel("NNLS signal fraction of winner")
    cb = fig.colorbar(sc, ax=np.atleast_1d(axes)[-1], pad=0.02)
    cb.set_label("total concentration (uM)")
    fig.suptitle("Binary ratio series: solution ratio vs NNLS signal composition "
                 "(unbent scale would give x0 = 0)", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}.png", dpi=180, bbox_inches="tight")
    print(f"저장: {os.path.basename(OUT)}.png")


def export():
    """Origin 재-import 용 csv (FIGURE_REDO 워크플로, `INTERP/panels/`)."""
    d = np.load(f"{OUT}.npz", allow_pickle=True)
    pd = os.path.join(paths.INTERP, "panels")
    os.makedirs(pd, exist_ok=True)
    gx = np.linspace(-2.3, 2.3, 231)
    with open(f"{pd}/panel_binary_response_points.csv", "w") as fp:
        fp.write("pair,log10_ratio,f_winner,total_uM,prep\n")
        for name, x, f, t in zip(d["pair"], d["x"], d["f"], d["total"]):
            for xi, fi, ti in zip(x, f, t):
                # 총농도 NaN = 최종 카탈로그 밖 별도 조제 (비율만 확실)
                prep = "replicate" if np.isnan(ti) else "final"
                tv = "" if np.isnan(ti) else f"{ti:.0f}"
                fp.write(f"{name},{xi:.4f},{fi:.4f},{tv},{prep}\n")
    with open(f"{pd}/panel_binary_response_fit.csv", "w") as fp:
        fp.write("pair,log10_ratio,f_fit\n")
        for name, x0, w in zip(d["pair"], d["x0"], d["w"]):
            for xi in gx:
                fp.write(f"{name},{xi:.3f},{1/(1+np.exp(-(xi-x0)/w)):.5f}\n")
    with open(f"{pd}/panel_binary_response_x0.csv", "w") as fp:
        fp.write("pair,x0_log10,w_log10,loser_excess_for_5050\n")
        for name, x0, w in zip(d["pair"], d["x0"], d["w"]):
            fp.write(f"{name},{x0:.3f},{w:.3f},{10**abs(x0):.2f}\n")
    print(f"저장: {os.path.relpath(pd, paths.DB)}/panel_binary_response_{{points,fit,x0}}.csv")


def merge():
    """합동 척도: 27 이원 점 전부를 성분별 세기 a_i 와 공유 폭 w 하나로 적합하고
    (f_A = σ((x − (a_B − a_A))/w)), 같은 모델 f_i ∝ (c_i·10^{a_i})^{1/(w·ln10)} 로
    삼원(1:1:x 포함)·등몰 맵의 NNLS 점유율을 예측해 실측과 대조한다.
    삼각 닫힘 0.04 dex 가 전제 — 닫히니까 한 척도로 합쳐진다."""
    from scipy.optimize import least_squares
    d = np.load(f"{OUT}.npz", allow_pickle=True)
    pts = []                                           # (iA, iB, x, f)
    for name, x, f in zip(d["pair"], d["x"], d["f"]):
        A, B = name.split("/")
        for xi, fi in zip(x, f):
            pts.append((SUB.index(A), SUB.index(B), float(xi), float(fi)))

    def model(p):
        a = np.array([p[0], 0.0, p[1]])                # a_DQ, a_TBZ≡0, a_THI
        w = np.exp(p[2])
        return a, w

    def resid(p):
        a, w = model(p)
        return [fi - 1 / (1 + np.exp(-((xi - (a[iB] - a[iA])) / w)))
                for iA, iB, xi, fi in pts]

    s = least_squares(resid, [-0.3, 0.7, np.log(0.4)])
    a, w = model(s.x)
    gamma = 1 / (w * np.log(10))
    rms = float(np.sqrt(np.mean(np.square(resid(s.x)))))
    print(f"합동 적합: a_DQ {a[0]:+.2f} · a_TBZ 0 · a_THI {a[2]:+.2f} (dex) · "
          f"w {w:.2f} · γ {gamma:.2f} · 잔차 RMS {rms:.3f} (27점)")
    print("  쌍별 x0 재구성: " + " · ".join(
        f"{A}/{B} {a[SUB.index(B)] - a[SUB.index(A)]:+.2f}"
        for A, B in [("THI", "DQ"), ("THI", "TBZ"), ("DQ", "TBZ")]))

    def predict(conc):
        z = np.array([(c * 10 ** ai) ** gamma if c > 0 else 0.0
                      for c, ai in zip(conc, a)])
        return z / z.sum()

    rows = []                                          # (set, label, true3, meas3, pred3)
    for sets, tag in ((("high",), "high"), (("equimolar",), "equimolar")):
        for m in MX.catalogue(sets, replicates=True, verbose=False):
            c = np.array(list(map(float, m.conc)))
            if tag == "high" and (c > 0).sum() != 3:
                continue
            meas = nnls_fraction(m.path)
            rows.append((tag, os.path.basename(m.path), c, meas, predict(c)))
            print(f"  [{tag}] {os.path.basename(m.path):28s} 실측 "
                  + "/".join(f"{v:4.2f}" for v in meas) + "  예측 "
                  + "/".join(f"{v:4.2f}" for v in predict(c)))
    for tag in ("high", "equimolar"):
        e = np.concatenate([r[3] - r[4] for r in rows if r[0] == tag])
        print(f"  삼원 예측 RMSE [{tag}]: {np.sqrt((e**2).mean()):.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    gx = np.linspace(-2.6, 2.6, 300)
    axL.plot(gx, 1 / (1 + np.exp(-gx / w)), "-", color="0.4", lw=1.4)
    axL.axhline(0.5, color="0.85", lw=0.8); axL.axvline(0, color="0.85", lw=0.8)
    mk = {"THI/DQ": "o", "THI/TBZ": "s", "DQ/TBZ": "^"}
    for name, x, f in zip(d["pair"], d["x"], d["f"]):
        A, B = name.split("/")
        x0 = a[SUB.index(B)] - a[SUB.index(A)]
        axL.plot(np.asarray(x, float) - x0, f, mk[name], ms=5.5,
                 mfc="none", mew=1.3, label=name)
    axL.set_xlabel("log10(ratio) − x0   (shifted by the pair's crossover)")
    axL.set_ylabel("NNLS signal fraction of winner")
    axL.set_title(f"All 3 pairs collapse on one curve (w = {w:.2f})", fontsize=10)
    axL.legend(frameon=False, fontsize=8)
    col = {"DQ": "tab:blue", "TBZ": "tab:green", "THI": "tab:red"}
    for tag, m2, fill in (("high", "o", True), ("equimolar", "D", False)):
        for _, _, c, meas, pred in [r for r in rows if r[0] == tag]:
            for i, sname in enumerate(SUB):
                axR.plot(pred[i], meas[i], m2, ms=6, mew=1.2, color=col[sname],
                         mfc=col[sname] if fill else "none")
    axR.plot([0, 1], [0, 1], "-", color="0.8", lw=1)
    axR.set_xlabel("predicted share  (binary-fitted scale)")
    axR.set_ylabel("measured NNLS share")
    axR.set_title("Ternary maps vs the same scale\n(filled high · open equimolar)",
                  fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}_merged.png", dpi=180, bbox_inches="tight")
    print(f"저장: {os.path.basename(OUT)}_merged.png")


def master():
    """마스터 곡선 한 장만 (npz 만 사용, 맵 재계산 없음) — 붙여넣기용."""
    from scipy.optimize import least_squares
    d = np.load(f"{OUT}.npz", allow_pickle=True)
    final_only = os.environ.get("MASTER_FINAL_ONLY") == "1"   # 별도 조제(total NaN) 제외
    pts = []
    for name, x, f, t in zip(d["pair"], d["x"], d["f"], d["total"]):
        A, B = name.split("/")
        for xi, fi, ti in zip(x, f, t):
            if final_only and not np.isfinite(float(ti)):
                continue
            pts.append((SUB.index(A), SUB.index(B), float(xi), float(fi)))

    def resid(p):
        a = np.array([p[0], 0.0, p[1]]); w = np.exp(p[2])
        return [fi - 1 / (1 + np.exp(-((xi - (a[iB] - a[iA])) / w)))
                for iA, iB, xi, fi in pts]

    s = least_squares(resid, [-0.3, 0.7, np.log(0.4)])
    a = np.array([s.x[0], 0.0, s.x[1]]); w = float(np.exp(s.x[2]))
    print(f"a_DQ {a[0]:+.2f} · a_TBZ 0 · a_THI {a[2]:+.2f} · w {w:.2f} · n {len(pts)}")

    gamma = 1 / (w * np.log(10))

    # 삼원 1:1:x 시리즈 + 등몰 — NNLS 실측 점유율 (--merge 2026-11-09 출력, 같은 판독
    # 경로). 각 성분을 "그 성분 vs 나머지 합"의 유효 초과량으로 같은 축에 올린다.
    TERN = [((3.3, 333.3, 333.3), (0.00, 0.00, 1.00)),     # 0.01:1:1 (DQ 희석)
            ((33.3, 333.3, 333.3), (0.00, 0.00, 1.00)),    # 0.1:1:1
            ((333.3, 3.3, 333.3), (0.00, 0.00, 1.00)),     # TBZ 희석
            ((333.3, 33.3, 333.3), (0.00, 0.00, 1.00)),
            ((333.3, 333.3, 3.3), (0.65, 0.29, 0.05)),     # THI 희석
            ((333.3, 333.3, 33.3), (0.51, 0.15, 0.34)),
            ((333.3, 333.3, 333.3), (0.00, 0.01, 0.99))]   # 1:1:1 (333 µM)
    EQM = [(10.0, (0.33, 0.21, 0.46)), (33.3, (0.21, 0.39, 0.41)),
           (100.0, (0.00, 0.00, 1.00)), (333.3, (0.00, 0.00, 1.00))]

    def x_eff(conc, i):
        z = np.array([(c * 10 ** ai) ** gamma for c, ai in zip(conc, a)])
        return w * np.log(z[i] / (z.sum() - z[i]))

    # ── Origin 용 마스터 csv: 모든 점을 같은 축(x = 50:50 기준 log10 초과량)으로 ──
    pd_ = os.path.join(paths.INTERP, "panels")
    os.makedirs(pd_, exist_ok=True)
    with open(f"{pd_}/panel_master_points.csv", "w") as fp:
        fp.write("kind,series,x_log10_excess,f_signal_share,f_curve_at_x,prep_or_totaluM\n")

        def crv(xi):
            return 1 / (1 + np.exp(-xi / w))

        for name, x, f, t in zip(d["pair"], d["x"], d["f"], d["total"]):
            A, B = name.split("/")
            x0 = a[SUB.index(B)] - a[SUB.index(A)]
            for xi, fi, ti in zip(x, f, t):
                if final_only and not np.isfinite(float(ti)):
                    continue                     # 최종판 csv 에는 별도 조제를 넣지 않는다
                prep = "replicate" if not np.isfinite(float(ti)) else "final"
                xs = float(xi) - x0
                fp.write(f"binary,{name},{xs:.4f},{float(fi):.4f},{crv(xs):.4f},{prep}\n")
        for conc, meas in TERN:
            for i, s in enumerate(SUB):
                xe = x_eff(conc, i)
                fp.write(f"ternary_1:1:x,{s},{xe:.4f},{meas[i]:.4f},{crv(xe):.4f},"
                         f"{sum(conc):.0f}\n")
        for tot, meas in EQM:
            for i, s in enumerate(SUB):
                xe = x_eff((tot,) * 3, i)
                fp.write(f"equimolar_1:1:1,{s},{xe:.4f},{meas[i]:.4f},{crv(xe):.4f},"
                         f"{3 * tot:.0f}\n")
    with open(f"{pd_}/panel_master_fit.csv", "w") as fp:
        fp.write(f"# a_DQ={a[0]:.3f}, a_TBZ=0, a_THI={a[2]:.3f} (dex), w={w:.3f}\n")
        fp.write("x_log10_excess,f_fit\n")
        for xi in np.linspace(-3.2, 3.6, 341):
            fp.write(f"{xi:.3f},{1 / (1 + np.exp(-xi / w)):.5f}\n")
    print(f"저장: panels/panel_master_{{points,fit}}.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Nature 규격: 전체 산세리프(Helvetica/Arial) 통일 · 그림 안 제목 없음(캡션으로)
    # · 축 = "양 (단위)" · 로그축 실값 눈금 · 열린 축(위·오른쪽 스파인 제거)
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.linewidth": 0.6,
                         "xtick.direction": "out", "ytick.direction": "out"})
    COLS = {"DQ": "#1f78c8", "TBZ": "#78c240", "THI": "#e3425f"}
    fig, ax = plt.subplots(figsize=(3.5, 2.9))          # 1단 폭 89 mm
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    gx = np.linspace(-3.2, 3.6, 300)
    ax.plot(gx, 100 / (1 + np.exp(-gx / w)), "-", color="0.2", lw=1.0, zorder=2,
            label="Logistic fit (binary)")

    # 최종 통일 규칙: 점 하나 = 어떤 혼합이든 "한 성분의 신호 분율".
    # 색 = 읽는 성분 · 모양 = 혼합 종류(빈 ○ 이원, 채운 ◆ 삼원) — 범례는 성분 3색만.
    for name, x, f, t in zip(d["pair"], d["x"], d["f"], d["total"]):
        A, B = name.split("/")
        x0 = a[SUB.index(B)] - a[SUB.index(A)]
        keep = (np.isfinite(np.asarray(t, float)) if final_only
                else np.ones(len(x), bool))
        xs = np.asarray(x, float)[keep] - x0
        fs = 100 * np.asarray(f, float)[keep]
        ax.plot(xs, fs, "o", color=COLS[A], ms=3.6, mew=0.8, mfc="none", lw=0)
    def dev_point(conc, meas, i, s, open_marker):
        """삼원·등몰 점 + 곡선 예측과의 이탈 선분 (위 = 승자 독식, 아래 = 패자 소멸)."""
        xe = x_eff(conc, i)
        pred = 100 / (1 + np.exp(-xe / w))
        ax.plot([xe, xe], [pred, 100 * meas], "-", color=COLS[s], lw=0.7, alpha=0.45,
                zorder=1)
        ax.plot(xe, 100 * meas, "D", color=COLS[s], ms=4.0, lw=0, zorder=3,
                mew=0.9 if open_marker else 0, mfc="none" if open_marker else COLS[s])

    binary_only = os.environ.get("MASTER_BINARY_ONLY") == "1"   # 본문용(삼원은 SI 판)
    if not binary_only:
        for conc, meas in TERN:
            for i, s in enumerate(SUB):
                dev_point(conc, meas[i], i, s, open_marker=False)
        # 등몰 1:1:1 은 비율이 고정이라 이 축의 변수가 아니다 — 자기(농도) 패널로 분리
        # (equimolar_panel / panel_equimolar.csv)
    for s in SUB:                                           # 범례: 성분 3색
        ax.plot([], [], "s", color=COLS[s], ms=4.5, mew=0, label=s)
    ax.plot([], [], "o", color="0.4", ms=3.6, mew=0.8, mfc="none", label="Binary mixture")
    ax.plot([], [], "D", color="0.4", ms=4, mew=0, label="Ternary mixture")

    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xticklabels([r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", "1",
                        "10", r"$10^{2}$", r"$10^{3}$"])
    ax.set_xlabel(r"Normalized molar ratio, $r/r_{50}$")
    ax.set_ylabel("Signal fraction (%)")
    ax.set_xlim(-3.4, 3.7); ax.set_ylim(-4, 104)
    ax.legend(frameon=False, fontsize=6, loc="upper left", handlelength=1.2,
              borderaxespad=0.2, labelspacing=0.35)
    fig.tight_layout()
    suffix = "_master_binary" if binary_only else "_master"
    if final_only:
        suffix += "_final"
    fig.savefig(f"{OUT}{suffix}.png", dpi=200, bbox_inches="tight")
    print(f"저장: {os.path.basename(OUT)}{suffix}.png")


def parity():
    """가장 직관적인 판: x = 용액에서 넣은 비율(%), y = 신호에서 보이는 비율(%).
    대각선 = 넣은 만큼 보임. 축 변환·이동이 없어 설명이 필요 없다.
    이원 점은 쌍 안에서의 분율, 삼원·등몰 점은 세 성분 전체에서의 분율."""
    d = np.load(f"{OUT}.npz", allow_pickle=True)
    TERN = [((3.3, 333.3, 333.3), (0.00, 0.00, 1.00)),
            ((33.3, 333.3, 333.3), (0.00, 0.00, 1.00)),
            ((333.3, 3.3, 333.3), (0.00, 0.00, 1.00)),
            ((333.3, 33.3, 333.3), (0.00, 0.00, 1.00)),
            ((333.3, 333.3, 3.3), (0.65, 0.29, 0.05)),
            ((333.3, 333.3, 33.3), (0.51, 0.15, 0.34)),
            ((333.3, 333.3, 333.3), (0.00, 0.01, 0.99))]
    EQM = [(10.0, (0.33, 0.21, 0.46)), (33.3, (0.21, 0.39, 0.41)),
           (100.0, (0.00, 0.00, 1.00)), (333.3, (0.00, 0.00, 1.00))]

    pd_ = os.path.join(paths.INTERP, "panels")
    os.makedirs(pd_, exist_ok=True)
    rows = []                                     # (kind, substance, in%, out%, note)
    for name, x, f, t in zip(d["pair"], d["x"], d["f"], d["total"]):
        A, B = name.split("/")
        for xi, fi, ti in zip(x, f, t):
            sol = 100 / (1 + 10 ** -float(xi))    # c_A/(c_A+c_B)
            prep = "replicate" if not np.isfinite(float(ti)) else "final"
            rows.append((f"binary({name})", A, sol, 100 * float(fi), prep))
            rows.append((f"binary({name})", B, 100 - sol, 100 * (1 - float(fi)), prep))
    for conc, meas in TERN:
        for i, s in enumerate(SUB):
            rows.append(("ternary", s, 100 * conc[i] / sum(conc), 100 * meas[i],
                         f"{sum(conc):.0f}uM"))
    for tot, meas in EQM:
        for i, s in enumerate(SUB):
            rows.append(("equimolar", s, 100 / 3, 100 * meas[i], f"{3 * tot:.0f}uM"))
    with open(f"{pd_}/panel_parity_points.csv", "w") as fp:
        fp.write("kind,substance,solution_pct,signal_pct,prep_or_totaluM\n")
        for r in rows:
            fp.write(f"{r[0]},{r[1]},{r[2]:.2f},{r[3]:.2f},{r[4]}\n")
    print("저장: panels/panel_parity_points.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COLS = {"DQ": "#1f78c8", "TBZ": "#78c240", "THI": "#e3425f"}
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 100], [0, 100], "-", color="0.75", lw=1.2, zorder=1)
    ax.text(72, 66, "what you put in\n= what you see", fontsize=8, color="0.5",
            rotation=45, ha="center", va="top")
    for kind, s, xi, yi, note in rows:
        open_m = (note == "replicate") or kind == "equimolar"
        m = "o" if kind.startswith("binary") else "D"
        ax.plot(xi, yi, m, color=COLS[s], ms=5.5 if m == "o" else 6.5, lw=0,
                mew=1.2 if open_m else 0, mfc="none" if open_m else COLS[s],
                alpha=0.85)
    for s in SUB:
        ax.plot([], [], "s", color=COLS[s], ms=7, lw=0, label=s)
    ax.plot([], [], "o", color="0.4", ms=5.5, lw=0, mew=0, label="two-substance map")
    ax.plot([], [], "D", color="0.4", ms=6.5, lw=0, mew=0, label="three-substance map")
    ax.plot([], [], "o", color="0.4", ms=5.5, lw=0, mew=1.2, mfc="none",
            label="replicate / equimolar")
    ax.set_xlabel("fraction in the SOLUTION (%)")
    ax.set_ylabel("fraction in the SERS SIGNAL (%)")
    ax.set_title("Put in vs. seen — adsorption competition bends the readout\n"
                 "(THI above the line, TBZ/DQ below; saturation pins points to 0 or 100)",
                 fontsize=9.5)
    ax.set_xlim(-3, 103); ax.set_ylim(-3, 103)
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}_parity.png", dpi=200, bbox_inches="tight")
    print(f"저장: {os.path.basename(OUT)}_parity.png")


if "--parity" in sys.argv:
    parity()
    sys.exit(0)

if "--master" in sys.argv:
    master()
    sys.exit(0)

if "--merge" in sys.argv:
    merge()
    sys.exit(0)

if "--export" in sys.argv:
    export()
    sys.exit(0)

if "--replot" in sys.argv:
    replot()
    sys.exit(0)

# 최종 카탈로그에 없는 별도 조제 7장 (md5 로 확인, 2026-11-09). 파일명 비율이
# 라벨이고 절대 µM 은 불명 — x 축에는 쓰고 총농도는 NaN. `_old` 도 반복으로 센다.
EXTRA = [  # (Ratio/ 상대경로, A, B, log10(c_A/c_B))
    ("Binary/DQ3TH1_corrected.csv",              "THI", "DQ",  -np.log10(3)),
    ("Binary/THI100DQ1_corrected.csv",           "THI", "DQ",  2.0),
    ("Binary/TB3TH1_corrected.csv",              "THI", "TBZ", -np.log10(3)),
    ("Binary - 복사본 2/TBZ3THI1_2_old_corrected.csv", "THI", "TBZ", -np.log10(3)),
    ("Binary/THI1TBZ10_corrected.csv",           "THI", "TBZ", -1.0),
    ("Binary/TBZ10DQ1-2_corrected.csv",          "DQ", "TBZ", -1.0),
    ("Binary/TBZ1DQ100_corrected.csv",           "DQ", "TBZ",  2.0),
]

maps = MX.catalogue(("high",), replicates=True)
rows = collections.defaultdict(list)           # pair → [(x, f_A, total, 파일)]
for m in maps:
    c = dict(zip(SUB, map(float, m.conc)))
    on = [s for s in SUB if c[s] > 0]
    if len(on) != 2:
        continue
    for A, B in PAIRS:
        if set(on) == {A, B}:
            comp = nnls_fraction(m.path)
            fA = comp[SUB.index(A)] / (comp[SUB.index(A)] + comp[SUB.index(B)] or 1.0)
            x = np.log10(c[A] / c[B])
            rows[(A, B)].append((x, fA, c[A] + c[B], os.path.basename(m.path)))
            print(f"  {A}:{B}  log10비율 {x:+5.2f}  총 {c[A]+c[B]:5.0f} µM  f_{A} {fA:5.3f}  {os.path.basename(m.path)}")

for rel, A, B, x in EXTRA:
    p = f"{paths.DATA}/Ratio/{rel}"
    comp = nnls_fraction(p)
    fA = comp[SUB.index(A)] / (comp[SUB.index(A)] + comp[SUB.index(B)] or 1.0)
    rows[(A, B)].append((x, fA, float("nan"), os.path.basename(p)))
    print(f"  {A}:{B}  log10비율 {x:+5.2f}  총  별도조제  f_{A} {fA:5.3f}  {os.path.basename(p)} [extra]")

pair_names, xs, fs, totals, x0s, ws = [], [], [], [], [], []
print("\n== 로지스틱 교차점 (x0 < 0 이면 A 쪽으로 눈금이 휨) ==")
for (A, B) in PAIRS:
    v = sorted(rows[(A, B)])
    if len(v) < 4:
        print(f"  {A}/{B}: 점 {len(v)}개 — 건너뜀"); continue
    x = np.array([t[0] for t in v]); f = np.array([t[1] for t in v])
    x0, w = logistic_fit(x, f)
    pair_names.append(f"{A}/{B}"); xs.append(x); fs.append(f)
    totals.append(np.array([t[2] for t in v])); x0s.append(x0); ws.append(w)
    print(f"  {A}/{B}:  n={len(v)}  x0={x0:+.2f} (용액 {B}:{A} = {10**(-x0):.1f}:1 에서 신호 50:50)  w={w:.2f}")

np.savez(f"{OUT}.npz", pair=np.array(pair_names), x0=np.array(x0s), w=np.array(ws),
         x=np.array(xs, object), f=np.array(fs, object), total=np.array(totals, object),
         allow_pickle=True)

replot()
print(f"저장: {os.path.basename(OUT)}.npz")
print("총농도가 비율 축과 얽혀 있다 — 곡선은 '비율+총량'의 합성 응답이고, 캡션에 명기할 것.")
