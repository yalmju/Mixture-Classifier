"""저농도 64조건 + 고농도까지 **평가에 포함**. 블록을 나눠 따로 보고한다.

저농도 64조건 완전격자(Baseline260808 DQ 9/18/36/72 × TBZ × THI)로 재학습.
4-fold 조건단위 검증.

  `DQ9-sus` 는 혼합이 잘못된 것으로 의심되는 배치라 **제외**한다.
  같은 내용(md5 동일)인 파일이 폴더를 넘나들며 중복 저장돼 있어 그것도 한 장으로 친다.
"""
import os as _os, sys as _sys                 # paths 부트스트랩 — 기계마다 마운트가 다르다
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import sys, os, re, glob, hashlib
import numpy as np

REPO = paths.REPO
sys.path.insert(0, REPO)
import dl_model
from real_data import load_map

import mixtures as MX

DB = paths.DB
PURE = paths.PURE
CALIB = paths.calibration()          # 하드코딩하면 조용히 사전학습이 꺼진다
SUB = MX.SUB

# 라벨은 `260814_mixture_final` 파일명이 최종 µM 이다 (`mixtures.py`).
HI = MX.items(("high",))
LO = MX.groups(("grid",), replicates=True)

# 고농도도 조성 키로 묶어 평가 대상에 넣는다. 다만 **블록은 분리**해 따로 보고한다 —
# 고농도 34맵 중 27맵이 이원(한 성분이 진짜 0이라 문제가 쉽다)이고, 삼원 7맵은
# 성분당 333 µM 라 THI 가 표면을 독점하는 적용 밖 구간이다. 한 숫자로 합치면
# 평균이 "문제가 쉬워서" 좋아진다.
HIG = {}
for p, c, cm in HI:
    key = tuple(round(float(c.get(s_, 0.0)), 6) for s_ in SUB)
    HIG.setdefault(key, []).append((p, c, cm))

keys = sorted(LO)
hkeys = sorted(HIG)
print(f"저농도 {len(keys)}조건 {sum(len(v) for v in LO.values())}맵 · "
      f"고농도 {len(hkeys)}조건 {len(HI)}맵")


def items_for(ks):
    out = []
    for k in ks:
        d = dict(zip(SUB, map(float, k)))
        for p in LO[k]:
            out.append((p, d, {s: v * 1e-6 for s, v in d.items()}))
    return out


EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
FOLDS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
# 세 번째 인자로 blank 클래스를 켠다. INK 는 DQ 와 코사인 0.73 으로 가장 닮았고
# NNLS 기저에서 빠져 있어 그 몫이 DQ 계수에 얹혀 있다 (dqfloor.py). blank 를 켜면
# 조성 헤드가 4-way 가 되어 "분석물 아님" 을 따로 뱉을 수 있다.
# 3번째 인자로 실험 조건을 고른다.
#   base   기존 그대로 (blank 없음, 사전학습에 잉크 없음)
#   blank  blank 클래스 추가
#   ink    사전학습 시뮬레이션에 잉크·기판 주입  (dqfloor.py 진단의 처방)
#   both   둘 다
# 3번째 인자로 실험 조건을 고른다. 쉼표로 조합 가능.
#   base   기존 그대로
#   blank  blank 클래스 추가
#   ink    사전학습 시뮬레이션에 잉크·기판 주입
#   sips   DQ 만 Sips 지수로 시뮬레이션 (TBZ·THI 는 m=1 이라 Langmuir 그대로)
#   both   = blank,ink
ARM = sys.argv[3] if len(sys.argv) > 3 else "base"
_t = set(ARM.replace("both", "blank,ink").split(","))
BLANK = "blank" in _t
INK = "ink" in _t
ISO = "sips_dq" if "sips" in _t else None
print(f"실험 조건: {ARM}   include_blank={BLANK}  sim_nuisance={INK}  sim_iso={ISO}")
rng = np.random.default_rng(0)
order = rng.permutation(len(keys))
folds = [[keys[i] for i in order[f::FOLDS]] for f in range(FOLDS)]
horder = np.random.default_rng(1).permutation(len(hkeys))
hfolds = [[hkeys[i] for i in horder[f::FOLDS]] for f in range(FOLDS)]


def hitems_for(hks):
    out = []
    for k in hks:
        for p, c, cm in HIG[k]:
            out.append((p, c, cm))
    return out

print(f"\n{FOLDS}-fold 조건단위 검증 · epochs={EPOCHS}")
print(f"{'조성 (최종 µM)':>16} | {'참 %':>18} | {'예측 %':>18} | {'오차':>6} | 우세")
allerr, nnls_err, hierr, HPRED = [], [], [], {}
for fi, te in enumerate(folds):
    tr = [k for k in keys if k not in te]
    hte = hfolds[fi]; htr = [k for k in hkeys if k not in hte]
    m = dl_model.train_model(PURE, hitems_for(htr) + items_for(tr), calib_path=CALIB, baseline=True,
                             trim=None, method="mlp", epochs=EPOCHS, seed=0,
                             use_pretrain=True, nnls_screen=True, px_per_map=20,
                             include_blank=BLANK, sim_nuisance=INK, sim_iso=ISO,
                             progress=lambda m: print('   ', m) if 'isotherm' in m else None)
    for k in te:
        C = np.array(k, float); t = C / C.sum() * 100
        ps = []
        for p in LO[k]:
            wn, cube, _mm, _co = load_map(p)
            r = dl_model.apply_model(m, np.asarray(wn, float), np.asarray(cube, float))
            ps.append([r["composition"].get(s, 0.0) for s in SUB])
        pm = np.mean(ps, axis=0); pm = pm / pm.sum() * 100
        e = np.abs(pm - t).sum() / 2
        allerr.append(e)
        tie = t.max() - t.min() < 1
        ok = "—" if tie else ("✅" if int(np.argmax(pm)) == int(np.argmax(t)) else "❌")
        print(f"  {'/'.join(f'{v:.0f}' for v in C):>16} | " + "/".join(f"{v:5.1f}" for v in t) +
              " | " + "/".join(f"{v:5.1f}" for v in pm) + f" | {e:5.1f}% | {ok}")
    # ---- 같은 fold 에서 고농도 시험 조건도 채점 (블록 분리 보고) ----
    for k in hte:
        C = np.array(k, float); t = C / C.sum() * 100
        ps = []
        for p, c, cm in HIG[k]:
            wn, cube, _mm, _co = load_map(p)
            r = dl_model.apply_model(m, np.asarray(wn, float), np.asarray(cube, float))
            ps.append([r["composition"].get(s_, 0.0) for s_ in SUB])
        pm = np.mean(ps, axis=0); pm = pm / pm.sum() * 100
        e = np.abs(pm - t).sum() / 2
        nz = int((C > 0).sum())
        hierr.append((k, nz, e)); HPRED[k] = pm
        print(f"  HI {'/'.join(f'{v:.0f}' for v in C):>18} | " + "/".join(f"{v:5.1f}" for v in t) +
              " | " + "/".join(f"{v:5.1f}" for v in pm) + f" | {e:5.1f}% | {nz}성분")

print(f"\n  저농도 {len(allerr)}조건 조성 오차  평균 {np.mean(allerr):5.1f}%  중앙 {np.median(allerr):5.1f}%")
_h2 = [e for _k, n, e in hierr if n == 2]
_h3 = [e for _k, n, e in hierr if n == 3]
if _h2:
    print(f"  고농도 이원 {len(_h2)}조건        평균 {np.mean(_h2):5.1f}%  중앙 {np.median(_h2):5.1f}%")
if _h3:
    print(f"  고농도 삼원 {len(_h3)}조건        평균 {np.mean(_h3):5.1f}%  중앙 {np.median(_h3):5.1f}%")

# ---- 소수 성분 검출 계열: 두 성분 1000 µM 고정, 세 번째만 1000 → 100 → 10 ----
# 묻힌 성분의 참 분율이 33.3% → 4.8% → 0.5% 로 내려간다. 성분마다 대칭이라
# "누가 묻히면 못 찾는가" 를 직접 비교할 수 있다.
buried = [(k, n, e) for k, n, e in hierr if n == 3]
if buried:
    print("\n=== 소수 성분 검출 (두 성분 1000 µM 고정) ===")
    print(f"{'조성 (µM)':>20} | {'묻힌 성분':>8} | {'참 분율':>7} | {'예측':>7} | {'회수':>6} | {'조성오차':>7}")
    for k, _n, e in sorted(buried, key=lambda x: min(x[0])):
        C = np.array(k, float); t = C / C.sum() * 100
        j = int(np.argmin(C))
        pm = HPRED.get(k)
        if pm is None:
            continue
        print(f"{'/'.join(f'{v:.0f}' for v in C):>20} | {SUB[j]:>8} | {t[j]:6.1f}% | "
              f"{pm[j]:6.1f}% | {pm[j]/max(t[j],1e-9)*100:5.0f}% | {e:6.1f}%")
print(f"  [비교] 6조건 18.8% · 35조건 12.9% · 반복산포 하한 9.6%")
