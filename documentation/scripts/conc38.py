"""저농도 세트의 절대농도 판독 — conc6.py의 두방향 적합을 38조건으로 확장.

  모델        s_im = g_m · r_i · C_im
              log s = log g_m + log r_i + log C
                g_m  맵 게인 (핫스팟 산포가 실리는 곳)
                r_i  성분 응답계수

  조성        s_i / r_i 정규화 → g_m 이 상쇄되므로 게인과 무관하게 복원된다
  절대농도    g_m 이 필요하다. 시험 맵의 게인은 알 수 없으므로 학습 평균 게인을 쓴다
              → 그 대체가 정확히 이 판독의 오차 원인이다

  검증은 **조건 단위** leave-one-out. 같은 조성의 반복맵은 함께 빠진다.

  실행:  python3 -u conc38.py
"""
import sys, os, re, glob, json, hashlib
import numpy as np
from scipy.optimize import nnls

REPO = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/github/Mixture Classifier"
DB = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB"
sys.path.insert(0, REPO)

from dl_model import _refs
from real_data import load_map
from sers_mixture import als_baseline

PURE = f"{DB}/Pure"
SUB = ["DQ", "TBZ", "THI"]
# `_corrected` 파일은 이미 baseline 보정본이다. conc6.py 는 원본 파일에 ALS 를 걸었으므로
# 그 코드를 그대로 옮기면 **이중 보정**이 된다. 두 모드를 다 돌려 대조할 수 있게 해둔다.
#   als        v - als_baseline(v)  후 clip   (conc6 와 같은 처리, 기본)
#   cliponly   clip 만               (메모리 sers-mixture-measurement-facts 의 권고)
MODE = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("als", "cliponly") else "als"
SUFFIX = "" if MODE == "als" else "_cliponly"
OUT = f"{REPO}/documentation/scripts/conc38_signals{SUFFIX}.json"
print(f"[전처리 모드] {MODE}")

subs, wn_t, mask, P, lo, hi = _refs(PURE, baseline=True, trim=None)
SI = [subs.index(s) for s in SUB]
print(f"템플릿 {subs} · 창 {lo:.0f}–{hi:.0f} cm⁻¹ · P {P.shape}")


def map_signal(path):
    """맵 한 장의 성분별 신호 = 픽셀별 NNLS 계수의 중앙값."""
    wn, cube, _m, _c = load_map(path)
    wn = np.asarray(wn, float); cube = np.asarray(cube, float)
    m = (wn >= lo) & (wn <= hi)
    wn_m, cube = wn[m], cube[:, m]
    out = []
    for v in cube:
        y = np.clip(v - als_baseline(v) if MODE == "als" else v, 0.0, None)
        if len(wn_m) != P.shape[1]:
            y = np.interp(np.linspace(wn_m[0], wn_m[-1], P.shape[1]), wn_m, y)
        a, _ = nnls(P.T, y)
        out.append(a)
    return np.median(np.asarray(out), axis=0)[SI]


# --------------------------------------------------------- 저농도 맵 수집 (조성별)
LO, _seen = {}, set()


def _add(key, p):
    """같은 내용(md5)의 파일이 두 폴더에 있으면 한 장으로만 센다."""
    h = hashlib.md5(open(p, "rb").read()).hexdigest()
    if h in _seen:
        return
    _seen.add(h)
    LO.setdefault(key, []).append(p)


for p in sorted(glob.glob(f"{DB}/Ratio/Baseline260808/DQ*/DQ*_corrected.csv")):
    if "DQ9-sus" in p:                      # 혼합 의심 배치 — 제외
        continue
    mm = re.match(r"DQ(\d+)-TB(\d+)-TH(\d+)", os.path.basename(p))
    if mm:
        _add(tuple(int(mm.group(i)) // 3 for i in (1, 2, 3)), p)
OLD = {(6, 6, 24): 1, (12, 12, 12): 2, (24, 6, 6): 3,
       (6, 6, 6): 4, (24, 24, 24): 5, (12, 12, 48): 6}
for key, n in OLD.items():
    for r in (1, 2, 3):
        p = f"{DB}/tert-new-baseline/{n}-{r}_corrected.csv"
        if os.path.exists(p):
            _add(key, p)

keys = sorted(LO)
print(f"조건 {len(keys)} · 맵 {sum(len(v) for v in LO.values())}")

# --------------------------------------------------------- 신호 측정 (캐시)
cache = {}
if os.path.exists(OUT):
    cache = {k: np.array(v) for k, v in json.load(open(OUT)).items()}
rows = []                                   # (조건키, 경로, 신호(3,), 참농도(3,))
todo = [(k, p) for k in keys for p in LO[k] if p not in cache]
for i, (k, p) in enumerate(todo, 1):
    print(f"  신호 측정 {i}/{len(todo)}  {os.path.basename(p)}", flush=True)
    cache[p] = map_signal(p)
if todo:
    json.dump({k: list(map(float, v)) for k, v in cache.items()}, open(OUT, "w"))
for k in keys:
    for p in LO[k]:
        rows.append((k, p, cache[p], np.array(k, float)))

S = np.array([r[2] for r in rows])          # (n_maps, 3) 신호
C = np.array([r[3] for r in rows])          # (n_maps, 3) 참 µM
G = np.array([str(r[0]) for r in rows], object)   # 조건 키
paths = [r[1] for r in rows]
print(f"신호 행렬 {S.shape}")

# --------------------------------------------------------- 두방향 적합 (마스킹 최소제곱)
# NNLS 계수가 정확히 0인 칸이 있다 (검출 실패). 1e-9로 클립해 로그를 씌우면 그 한 칸이
# log10 = -9 가 되어 평균 게인과 잔차 통계를 통째로 망가뜨린다. 유효한 칸만 쓴다.
VALID = S > 0
n_bad = int((~VALID).sum())
if n_bad:
    print(f"\n[주의] NNLS 계수 0 = 검출 실패 {n_bad}칸 — 적합·통계에서 제외")
    for m_, j_ in zip(*np.where(~VALID)):
        print(f"        {os.path.basename(paths[m_]):34s} {SUB[j_]}  (참 {C[m_, j_]:.0f} µM)")


def twoway(Smat, Cmat, valid):
    """log s − log C = a_m + b_i 를 유효 칸만으로 최소제곱. 제약 mean(b)=0.
    반환 (a: 맵 게인 로그, b: 응답 로그, 잔차행렬 nan 채움)."""
    mi, ii = np.where(valid)
    y = np.log10(Smat[mi, ii]) - np.log10(Cmat[mi, ii])
    nm, ns = Smat.shape
    A = np.zeros((len(y), nm + ns))
    A[np.arange(len(y)), mi] = 1.0
    A[np.arange(len(y)), nm + ii] = 1.0
    A = np.vstack([A, np.r_[np.zeros(nm), np.ones(ns)]])      # mean(b)=0
    yy = np.r_[y, 0.0]
    sol, *_ = np.linalg.lstsq(A, yy, rcond=None)
    a, b = sol[:nm], sol[nm:]
    R = np.full(Smat.shape, np.nan)
    R[mi, ii] = y - (a[mi] + b[ii])
    return a, b, R


gm, ri, res = twoway(S, C, VALID)
rv = res[np.isfinite(res)]
print("\n=== 두방향 적합 s ~ g·r·C  (유효 칸 최소제곱) ===")
print("  응답계수 r (DQ=1):  " + "  ".join(
    f"{SUB[j]} {10 ** (ri[j] - ri[0]):6.2f}" for j in range(3)))
print(f"  맵 게인 g 산포:  {10 ** gm.std():.2f}배 (1 SD)  ·  최대/최소 {10 ** (gm.max() - gm.min()):.1f}배")
print(f"  잔차:  전형 {10 ** np.sqrt((rv ** 2).mean()):.2f}배  ·  최악 {10 ** np.abs(rv).max():.2f}배")

# --------------------------------------------------------- 조건단위 LOO
print("\n=== 조건단위 LOO — 절대농도 ===")
uk = list(dict.fromkeys(G.tolist()))
fold_all, recs, miss = [], [], []
EST = np.full_like(S, np.nan)                          # 맵별 **held-out** 추정 µM
for ck in uk:
    te = np.where(G == ck)[0]; tr = np.where(G != ck)[0]
    a_tr, r_tr, _ = twoway(S[tr], C[tr], VALID[tr])
    g_tr = a_tr.mean()                                 # 시험 맵 게인은 모른다 → 학습 평균
    est = S[te] / 10 ** (g_tr + r_tr)
    EST[te] = est                                      # 그림도 이 값을 써야 통계와 일치한다
    fold = np.maximum(est / np.clip(C[te], 1e-12, None),
                      C[te] / np.clip(est, 1e-12, None))
    fold = np.where(VALID[te], fold, np.nan)           # 검출 실패는 배수오차에서 분리
    miss.append(int((~VALID[te]).sum()))
    fold_all.append(fold)
    recs.append((ck, C[te][0], est.mean(axis=0), np.nanmean(fold, axis=0)))
F = np.concatenate(fold_all)                           # (n_maps, 3), 검출 실패는 nan
print(f"{'조성 (µM)':>14} | {'참':>16} | {'추정':>18} | {'배수오차':>16}")
for ck, c, e, f in sorted(recs, key=lambda x: -x[3].mean()):
    print(f"{'/'.join(f'{v:.0f}' for v in c):>14} | " + "/".join(f"{v:4.0f}" for v in c) +
          " | " + "/".join(f"{v:5.1f}" for v in e) + " | " + "/".join(f"{v:4.1f}x" for v in f))
fv = F[np.isfinite(F)]
print(f"\n  유효 {fv.size}점 (검출 실패 {int(np.isnan(F).sum())}점 제외)")
print(f"  중앙 {np.median(fv):.2f}배  ·  2배 이내 {100 * (fv < 2).mean():.0f}%"
      f"  ·  3배 이내 {100 * (fv < 3).mean():.0f}%  ·  최악 {fv.max():.1f}배")
for j in range(3):
    c = F[:, j][np.isfinite(F[:, j])]
    print(f"    {SUB[j]:<4} 중앙 {np.median(c):.2f}배 · 2배내 {100 * (c < 2).mean():3.0f}%"
          f" · 3배내 {100 * (c < 3).mean():3.0f}% · 최악 {c.max():.1f}배")

# --------------------------------------------------------- 총량
print("\n=== 총 농도 ===")
tf = []
for ck in uk:
    te = np.where(G == ck)[0]; tr = np.where(G != ck)[0]
    a_tr, r_tr, _ = twoway(S[tr], C[tr], VALID[tr])
    g_tr = a_tr.mean()
    est = (S[te] / 10 ** (g_tr + r_tr)).sum(axis=1).mean()
    tot = C[te][0].sum()
    tf.append(max(est / tot, tot / est))
tf = np.array(tf)
print(f"  중앙 {np.median(tf):.2f}배  ·  2배 이내 {100 * (tf < 2).mean():.0f}%  ·  최악 {tf.max():.1f}배")

np.savez(f"{REPO}/documentation/scripts/conc38_result{SUFFIX}.npz",
         S=S, C=C, G=G.astype(str), paths=np.array(paths),
         fold=F, ri=ri, gm=gm, est=EST, valid=VALID)
print(f"\n저장: conc38_result{SUFFIX}.npz")
