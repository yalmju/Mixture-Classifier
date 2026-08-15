"""잎 시료 4맵 — 전처리 + 픽셀별 판독값 추출 (그림 없음, 숫자만).

  입력: `*_corrected.csv` 10x10 맵 (앱이 쓰는 형식 — 1행 `X num`, 2행 `Y num`,
        3행 헤더 `X,Y,<파장수...>`, 이후 한 줄 = 한 픽셀).
  전처리: ALS 베이스라인 제거 (lam=1e5, p=0.01, 10회) — `analysis/dq9-sus-
        reproducibility/scripts/01_load_preprocess.py` 와 같은 설정.
  판독값:
        ink2137   2100-2180 cm-1 적분 — 잉크 리포터 (silent region)
        <성분>_<밴드>  마커밴드 +-10 cm-1 적분
        <성분>    그 성분 마커밴드 3개의 평균
        <성분>_over_ink   마커/잉크 — 기판 gain 이 상쇄되는 판독값

  출력: `data/pixels.npz` (다음 스크립트용) + `data/*.csv` (Origin 용).

  실행:  python3 -u 01_extract.py [데이터폴더]
"""
import os, sys, glob, csv
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "data")
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), "raw")

# 파일명(접두 해시는 무시) -> 시료 이름.  순서가 그림의 열 순서다.
SAMPLES = ["leaf_dq", "leaf_tb1", "leaf_thi2", "cov3"]

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
        store[s] = dict(B=B, XY=XY, rows=rows, cols=cols, ux=ux, uy=uy,
                        src=os.path.basename(p))
        print(f"loaded {s:>10}  {A.shape}  {len(ux)}x{len(uy)} grid  "
              f"step {np.median(np.diff(ux)):.0f} um", flush=True)

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

    # ---- 대표 스펙트럼 (맵 100픽셀의 중앙값 + 사분위) ----
    rep = {}
    for s in SAMPLES:
        B = store[s]["B"]
        ink = np.median(feat[s]["ink2137"])
        rep[s] = dict(med=np.median(B, axis=0),
                      q25=np.percentile(B, 25, axis=0),
                      q75=np.percentile(B, 75, axis=0),
                      ink=ink)
    np.savez_compressed(os.path.join(OUT, "representative.npz"), wn=wn0,
                        **{f"{s}__{k}": v for s in SAMPLES for k, v in rep[s].items()})
    with open(os.path.join(OUT, "spectra_representative.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        head = ["wavenumber_cm-1"]
        for s in SAMPLES:
            head += [f"{s}_median", f"{s}_q25", f"{s}_q75", f"{s}_median_over_ink"]
        w.writerow(head)
        for i, v in enumerate(wn0):
            row = [f"{v:.2f}"]
            for s in SAMPLES:
                r = rep[s]
                row += [f"{r['med'][i]:.6g}", f"{r['q25'][i]:.6g}", f"{r['q75'][i]:.6g}",
                        f"{r['med'][i]/r['ink']:.6g}"]
            w.writerow(row)

    # ---- 요약: 맵 대표값과 조제 내 산포 ----
    with open(os.path.join(OUT, "summary_stats.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["sample", "readout", "median", "within_prep_SD", "within_prep_CV_pct",
                    "p10", "p90", "n_pixels"])
        for s in SAMPLES:
            for k in ["ink2137"] + [nm for nm in SUB] + [f"{nm}_over_ink" for nm in SUB]:
                v = feat[s][k]
                w.writerow([s, k, f"{np.median(v):.6g}", f"{v.std(ddof=1):.6g}",
                            f"{v.std(ddof=1)/v.mean()*100:.1f}", f"{np.percentile(v,10):.6g}",
                            f"{np.percentile(v,90):.6g}", len(v)])
    # ---- 밴드별 선택성 (마커밴드 하나하나가 시료를 가르는가) ----
    with open(os.path.join(OUT, "band_selectivity.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["substance", "band_cm-1"] + [f"{s}_over_ink" for s in SAMPLES]
                   + ["max_over_min", "argmax_sample"])
        for nm, bands in MARKERS.items():
            for b in bands:
                v = [float(np.median(feat[s][f"{nm}_{int(b)}"] / feat[s]["ink2137"]))
                     for s in SAMPLES]
                w.writerow([nm, int(b)] + [f"{x:.4f}" for x in v]
                           + [f"{max(v)/min(v):.2f}", SAMPLES[int(np.argmax(v))]])

    with open(os.path.join(OUT, "SOURCES.txt"), "w") as f:
        for s in SAMPLES:
            f.write(f"{s}\t{store[s]['src']}\n")

    print("\n판독값 (맵 중앙값, 조제 내 CV%)")
    print(f"{'sample':>10} {'ink2137':>12} " + " ".join(f"{nm+'/ink':>14}" for nm in SUB))
    for s in SAMPLES:
        cells = []
        for nm in SUB:
            v = feat[s][f"{nm}_over_ink"]
            cells.append(f"{np.median(v):7.4f} ({v.std(ddof=1)/v.mean()*100:3.0f}%)")
        ik = feat[s]["ink2137"]
        print(f"{s:>10} {np.median(ik):8.0f} ({ik.std(ddof=1)/ik.mean()*100:3.0f}%) "
              + " ".join(f"{c:>14}" for c in cells))
    print("\nwrote ->", OUT)


if __name__ == "__main__":
    main()
