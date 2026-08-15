"""잎 시료 맵 시리즈 — 전처리 + 픽셀별 판독값 추출 (그림 없음, 숫자만).

  입력: `*_corrected.csv` 맵 (앱이 쓰는 형식 — 1행 `X num`, 2행 `Y num`,
        3행 헤더 `X,Y,<파장수...>`, 이후 한 줄 = 한 픽셀).
        격자 크기는 맵마다 달라도 된다 (10x10/50 µm 와 20x20/100 µm 가 섞여 있다).
  전처리: ALS 베이스라인 제거 (lam=1e5, p=0.01, 10회) — `analysis/dq9-sus-
        reproducibility/scripts/01_load_preprocess.py` 와 같은 설정.
  판독값:
        ink2137   2100-2180 cm-1 적분 — 잉크 리포터 (silent region)
        <성분>_<밴드>  마커밴드 +-10 cm-1 적분
        <성분>    그 성분 마커밴드 3개의 평균
        <성분>_over_ink   마커/잉크 — 기판 gain 이 상쇄되는 판독값
  peel: 필름이 **잉크째** 떼어낸다 (사용자 확인 2026-08-15, 같은 잎 같은 자리).
        그래서 잉크는 두 맵 사이에서 고정 기준물질이 아니다 — 픽셀 안 gain 상쇄에는
        그대로 쓰되, 맵끼리는 잉크 자신의 제거율과 나란히 봐야 한다.
  대조군: `cov3` (잎에 잉크만) 를 blank 로 삼아 검출 대비를 낸다.
        contrast = 중앙값 - blank 중앙값,  SNR = contrast / blank SD,
        검출픽셀 = blank 평균 + 3 SD 를 넘는 픽셀 비율.

  출력: `data/pixels.npz` · `representative.npz` (다음 스크립트용) + `data/*.csv` (Origin 용).

  실행:  python3 -u 01_extract.py [데이터폴더]
"""
import os, sys, glob, csv
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "data")
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), "raw")

# 시료 순서 = 그림의 열 순서.  (이름, 그룹) — 그룹은 그림 상단 묶음 라벨에 쓴다.
SAMPLES = ["leaf_dq", "leaf_tb1", "leaf_thi2", "cov3",
           "leavesmix", "leavesmix2peel", "leavesmix2peel2"]
GROUPS = [("single analyte", ["leaf_dq", "leaf_tb1", "leaf_thi2"]),
          ("control", ["cov3"]),
          ("mixture", ["leavesmix", "leavesmix2peel", "leavesmix2peel2"])]
CONTROL = "cov3"                     # 잎에 잉크만 올린 대조군 (사용자 확인 2026-08-15)

INK_BAND = (2100.0, 2180.0)          # 잉크 2137 cm-1 — 분석물 밴드가 없는 침묵 구간
HALF = 10.0                          # 마커밴드 적분 반폭 (cm-1)
MARKERS = {                          # dq9-sus README 4절의 선택적 마커밴드
    "DQ":  [1179.0, 1521.0, 1576.0],
    "TBZ": [779.0, 1011.0, 1549.0],
    "THI": [556.0, 1138.0, 1365.0],
}
SUB = list(MARKERS)


def load(path):
    """맵 CSV -> (wn, XY, A).  A 는 (픽셀, 파장수) 원본 세기."""
    lines = open(path).read().strip().split("\n")
    hdr = [v for v in lines[2].split(",") if v.strip()]
    wn = np.array([float(v) for v in hdr[2:]])
    n = len(wn)
    rows = [l.split(",") for l in lines[3:] if l.strip()]
    XY = np.array([[float(r[0]), float(r[1])] for r in rows], float)
    A = np.array([[float(v) for v in r[2:2 + n]] for r in rows], float)
    return wn, XY, A


def als(y, lam=1e5, p=0.01, n=10):
    """asymmetric least squares baseline"""
    L = len(y)
    D = sparse.diags([1., -2., 1.], [0, -1, -2], shape=(L, L - 2))
    DD = lam * D.dot(D.T)
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    z = y
    for _ in range(n):
        W.setdiag(w)
        z = spsolve((W + DD).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def area(B, wn, lo, hi):
    """(픽셀,) 구간 적분.  B 는 베이스라인 제거 후."""
    m = (wn >= lo) & (wn <= hi)
    return np.trapezoid(B[:, m], wn[m], axis=1)


def grid_index(XY):
    """픽셀 -> (행, 열).  행은 Y 오름차순, 열은 X 오름차순 (page_real._grid_rc 와 동일)."""
    ux, uy = np.unique(XY[:, 0]), np.unique(XY[:, 1])
    xi = {v: i for i, v in enumerate(ux)}
    yi = {v: i for i, v in enumerate(uy)}
    rows = np.array([yi[v] for v in XY[:, 1]])
    cols = np.array([xi[v] for v in XY[:, 0]])
    return rows, cols, ux, uy


def find(name):
    hit = [p for p in glob.glob(os.path.join(DATA, "*.csv"))
           if os.path.basename(p).endswith(name + "_corrected.csv")]
    if not hit:
        raise SystemExit(f"없음: {name}_corrected.csv  (찾은 폴더: {DATA})")
    return sorted(hit)[0]


def main():
    os.makedirs(OUT, exist_ok=True)
    store = {}
    wn0 = None
    for s in SAMPLES:
        p = find(s)
        wn, XY, A = load(p)
        if wn0 is None:
            wn0 = wn
        assert np.allclose(wn, wn0), f"{s}: 파장축이 다르다"
        B = np.array([y - als(y) for y in A])            # 베이스라인 제거
        rows, cols, ux, uy = grid_index(XY)
        step = float(np.median(np.diff(ux))) if len(ux) > 1 else 0.0
        store[s] = dict(B=B, XY=XY, rows=rows, cols=cols, ux=ux, uy=uy, step=step,
                        span=float(ux.max() - ux.min()), src=os.path.basename(p))
        print(f"loaded {s:>16}  {A.shape}  {len(ux)}x{len(uy)} grid  step {step:.0f} um  "
              f"span {store[s]['span']:.0f} um", flush=True)

    # ---- 픽셀별 판독값 ----
    feat = {}
    for s in SAMPLES:
        B = store[s]["B"]
        d = {"ink2137": area(B, wn0, *INK_BAND)}
        for nm, bands in MARKERS.items():
            per = []
            for b in bands:
                v = area(B, wn0, b - HALF, b + HALF)
                d[f"{nm}_{int(b)}"] = v
                per.append(v)
            d[nm] = np.mean(per, axis=0)
            d[f"{nm}_over_ink"] = d[nm] / d["ink2137"]
        feat[s] = d

    np.savez_compressed(
        os.path.join(OUT, "pixels.npz"), wn=wn0, samples=np.array(SAMPLES),
        **{f"{s}__B": store[s]["B"] for s in SAMPLES},
        **{f"{s}__XY": store[s]["XY"] for s in SAMPLES},
        **{f"{s}__rows": store[s]["rows"] for s in SAMPLES},
        **{f"{s}__cols": store[s]["cols"] for s in SAMPLES},
        **{f"{s}__step": np.array(store[s]["step"]) for s in SAMPLES},
        **{f"{s}__span": np.array(store[s]["span"]) for s in SAMPLES},
        **{f"{s}__{k}": v for s in SAMPLES for k, v in feat[s].items()})

    # ---- Origin: 픽셀 long form ----
    keys = ["ink2137"] + [k for nm in SUB for k in
                          ([f"{nm}_{int(b)}" for b in MARKERS[nm]] + [nm, f"{nm}_over_ink"])]
    with open(os.path.join(OUT, "map_pixels_long.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["sample", "row", "col", "x_um", "y_um"] + keys)
        for s in SAMPLES:
            st, ft = store[s], feat[s]
            x0, y0 = st["ux"].min(), st["uy"].min()
            for i in range(len(st["XY"])):
                w.writerow([s, st["rows"][i], st["cols"][i],
                            f"{st['XY'][i,0]-x0:.0f}", f"{st['XY'][i,1]-y0:.0f}"]
                           + [f"{ft[k][i]:.6g}" for k in keys])

    # ---- Origin: 히트맵은 행렬로 (Origin 이 행렬이라야 heat map 을 만든다) ----
    for s in SAMPLES:
        st, ft = store[s], feat[s]
        ny, nx = len(st["uy"]), len(st["ux"])
        for k in ["ink2137"] + [f"{nm}_over_ink" for nm in SUB]:
            M = np.full((ny, nx), np.nan)
            M[st["rows"], st["cols"]] = ft[k]
            with open(os.path.join(OUT, f"matrix_{s}_{k}.csv"), "w", newline="",
                      encoding="utf-8-sig") as f:
                w = csv.writer(f)
                for r in range(ny - 1, -1, -1):          # 위쪽 행 = 큰 Y
                    w.writerow([f"{v:.6g}" for v in M[r]])

    # ---- 대표 스펙트럼 (맵 픽셀의 중앙값 + 사분위) ----
    rep = {}
    for s in SAMPLES:
        B = store[s]["B"]
        rep[s] = dict(med=np.median(B, axis=0),
                      q25=np.percentile(B, 25, axis=0),
                      q75=np.percentile(B, 75, axis=0),
                      ink=np.median(feat[s]["ink2137"]))
    blank_spec = rep[CONTROL]["med"] / rep[CONTROL]["ink"]
    for s in SAMPLES:                                    # 대조군을 뺀 차이 스펙트럼
        rep[s]["diff"] = rep[s]["med"] / rep[s]["ink"] - blank_spec
    np.savez_compressed(os.path.join(OUT, "representative.npz"), wn=wn0,
                        **{f"{s}__{k}": v for s in SAMPLES for k, v in rep[s].items()})

    with open(os.path.join(OUT, "spectra_representative.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        head = ["wavenumber_cm-1"]
        for s in SAMPLES:
            head += [f"{s}_median", f"{s}_q25", f"{s}_q75", f"{s}_median_over_ink",
                     f"{s}_minus_control"]
        w.writerow(head)
        for i, v in enumerate(wn0):
            row = [f"{v:.2f}"]
            for s in SAMPLES:
                r = rep[s]
                row += [f"{r['med'][i]:.6g}", f"{r['q25'][i]:.6g}", f"{r['q75'][i]:.6g}",
                        f"{r['med'][i]/r['ink']:.6g}", f"{r['diff'][i]:.6g}"]
            w.writerow(row)

    # ---- 요약: 맵 대표값과 조제 내 산포 ----
    with open(os.path.join(OUT, "summary_stats.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["sample", "readout", "median", "within_prep_SD", "within_prep_CV_pct",
                    "p10", "p90", "n_pixels"])
        for s in SAMPLES:
            for k in ["ink2137"] + SUB + [f"{nm}_over_ink" for nm in SUB]:
                v = feat[s][k]
                w.writerow([s, k, f"{np.median(v):.6g}", f"{v.std(ddof=1):.6g}",
                            f"{v.std(ddof=1)/v.mean()*100:.1f}", f"{np.percentile(v,10):.6g}",
                            f"{np.percentile(v,90):.6g}", len(v)])

    # ---- 대조군 대비 검출 ----
    det = {}
    with open(os.path.join(OUT, "detection_vs_control.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["sample", "substance", "median_over_ink", "control_median",
                    "contrast", "fold_over_control", "control_SD", "SNR_contrast_over_SD",
                    "threshold_3SD", "pct_pixels_above_3SD", "n_pixels"])
        for nm in SUB:
            b = feat[CONTROL][f"{nm}_over_ink"]
            bmed, bmean, bsd = np.median(b), b.mean(), b.std(ddof=1)
            thr = bmean + 3 * bsd
            for s in SAMPLES:
                v = feat[s][f"{nm}_over_ink"]
                pct = float((v > thr).mean() * 100)
                det[(s, nm)] = dict(med=float(np.median(v)), contrast=float(np.median(v) - bmed),
                                    snr=float((np.median(v) - bmed) / bsd), pct=pct,
                                    fold=float(np.median(v) / bmed), thr=float(thr))
                w.writerow([s, nm, f"{np.median(v):.4f}", f"{bmed:.4f}",
                            f"{np.median(v)-bmed:.4f}", f"{np.median(v)/bmed:.2f}",
                            f"{bsd:.4f}", f"{(np.median(v)-bmed)/bsd:.2f}",
                            f"{thr:.4f}", f"{pct:.0f}", len(v)])

    # ---- 밴드별 선택성 ----
    with open(os.path.join(OUT, "band_selectivity.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["substance", "band_cm-1"] + [f"{s}_over_ink" for s in SAMPLES]
                   + ["fold_over_control_max", "argmax_sample"])
        for nm, bands in MARKERS.items():
            for b in bands:
                v = {s: float(np.median(feat[s][f"{nm}_{int(b)}"] / feat[s]["ink2137"]))
                     for s in SAMPLES}
                others = [s for s in SAMPLES if s != CONTROL]
                w.writerow([nm, int(b)] + [f"{v[s]:.4f}" for s in SAMPLES]
                           + [f"{max(v[s] for s in others)/v[CONTROL]:.2f}",
                              max(others, key=lambda s: v[s])])

    # ---- peel 제거율 — 잉크가 같이 떨어지므로 잉크 자신의 제거율과 나란히 본다 ----
    PEEL_PAIR = ("leavesmix", "leavesmix2peel")          # 같은 잎 같은 자리, 같은 격자
    if all(k in feat for k in PEEL_PAIR):
        a, b = PEEL_PAIR
        ia, ib = np.median(feat[a]["ink2137"]), np.median(feat[b]["ink2137"])
        with open(os.path.join(OUT, "peel_removal.csv"), "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["readout", "before_peel", "after_peel", "kept_fraction",
                        "removed_pct", "kept_relative_to_ink"])
            w.writerow(["ink2137", f"{ia:.6g}", f"{ib:.6g}", f"{ib/ia:.3f}",
                        f"{100*(1-ib/ia):.1f}", "1.000"])
            for nm in SUB:
                va, vb = np.median(feat[a][nm]), np.median(feat[b][nm])
                w.writerow([nm, f"{va:.6g}", f"{vb:.6g}", f"{vb/va:.3f}",
                            f"{100*(1-vb/va):.1f}", f"{(vb/va)/(ib/ia):.3f}"])
            # 잉크가 고른 두께로 걷혔나 — 분위수끼리 맞대 본다.  두 맵은 스테이지
            # 좌표계가 달라(재장착) 픽셀 단위 차감은 못 하고, 분포 비교까지가 한계다.
            for q in (10, 25, 50, 75, 90):
                qa = np.percentile(feat[a]["ink2137"], q)
                qb = np.percentile(feat[b]["ink2137"], q)
                w.writerow([f"ink2137_p{q}", f"{qa:.6g}", f"{qb:.6g}", f"{qb/qa:.3f}",
                            f"{100*(1-qb/qa):.1f}", ""])
        print(f"\npeel 제거율 ({a} -> {b})   잉크 {100*(1-ib/ia):.0f}% 제거")
        for nm in SUB:
            va, vb = np.median(feat[a][nm]), np.median(feat[b][nm])
            print(f"  {nm:>4}: {100*(1-vb/va):5.1f}% 제거   잉크 대비 잔존 "
                  f"{(vb/va)/(ib/ia):.3f}")

    with open(os.path.join(OUT, "SOURCES.txt"), "w") as f:
        for s in SAMPLES:
            st = store[s]
            f.write(f"{s}\t{st['src']}\t{len(st['ux'])}x{len(st['uy'])}\t"
                    f"step {st['step']:.0f} um\n")

    print(f"\n대조군 = {CONTROL}.  마커/잉크 중앙값, 괄호는 대조군 대비 배수")
    print(f"{'sample':>16} {'ink2137':>14} " + " ".join(f"{nm:>16}" for nm in SUB))
    for s in SAMPLES:
        ik = feat[s]["ink2137"]
        cells = [f"{det[(s,nm)]['med']:6.3f} ({det[(s,nm)]['fold']:4.1f}x)" for nm in SUB]
        print(f"{s:>16} {np.median(ik):8.0f} ({ik.std(ddof=1)/ik.mean()*100:3.0f}%) "
              + " ".join(f"{c:>16}" for c in cells))
    print(f"\n{'sample':>16} " + " ".join(f"{nm+' %px>3SD':>16}" for nm in SUB))
    for s in SAMPLES:
        print(f"{s:>16} " + " ".join(f"{det[(s,nm)]['pct']:14.0f}%" for nm in SUB))
    print("\nwrote ->", OUT)


if __name__ == "__main__":
    main()
