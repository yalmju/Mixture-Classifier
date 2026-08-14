"""혼합물 목록 한 곳 — 라벨은 `Ratio/260814_mixture_final` 의 파일명이다.

  ## 왜 이 파일이 생겼나

  2026-08-14 이전에는 스크립트마다 세 벌의 수집 코드를 각자 들고 있었다.

    고농도  `Ratio/Ratio_mix/*orrected.csv` → `parse_hi()` 가 파일명 숫자를 그대로 µM 로
    저농도  `Ratio/Baseline260808/DQ*/…`    → 파일명 숫자 ÷ 3
    반복    `tert-new-baseline/{n}-{r}`     → 스크립트 안의 `OLD` dict

  **고농도 쪽 ÷ 가 빠져 있었다.** `Ratio_mix` 파일명은 혼합 전 원액 농도다. 등량으로
  섞으므로 최종은 이성분 ÷2 · 삼성분 ÷3 이고 (`TODO.md` §6 에 이미 적혀 있다:
  "`TBZ1000DQ1000THI1000`(equal-volume 기준 333 µM씩)"), 저농도 쪽은 ÷3 을 하는데
  고농도 쪽만 안 했다. 그래서 2026-08-12~13 의 export 는 **한 표 안에 최종농도와 원액농도가
  섞여** 있었다 — 조성 %는 스케일이 상쇄돼 멀쩡하지만 µM 축은 32조건 전부 2~3배 과대다.

  또 `Ratio_mix/DQ1000_corrected.csv` 는 순물질로 읽혔는데 실제로는 DQ:THI 1:1 이다
  (`Ratio/Binary/DQ1TH1_corrected.csv` 와 md5 동일 · 최종 `DQ500-TB0-TH500`).

  ## 규약

  `260814_mixture_final/` 의 **파일명이 최종 µM 이고, 그것이 유일한 라벨 권위**다.
  여기서는 파일명을 나눗셈 없이 그대로 읽는다. 나눗셈이 코드에 남아 있으면 어느 세트에
  적용되는지가 다시 흩어진다.

  세트 구분은 파일명만으로 결정된다.
    grid       세 성분 모두 {3,6,12,24} — 64조건 완전격자
    equimolar  등몰 희석 계열 (10 · 33 · 100 · 333 µM/성분)
    high       나머지 — 예전 `Ratio_mix` 34맵 (3.3–500 µM)

  > ⚠ 파일명은 반올림돼 있다. `DQ3-TB333-TH333` 의 실제값은 3.33/333.3/333.3 µM 다.
  >   조성 기여는 0.1% 미만이지만, DQ 3 vs 3.33 은 농도 회수율에서 11% 차이다.

  ## 최종 폴더에 없는 맵

  같은 조건을 **다시 조제**한 것들이라 조건 목록에는 안 들어가고 파일만 따로 있다.
  `replicates=True` 로 켜거나, 출처를 골라 켠다 (기본은 꺼짐).
    "baseline"   Baseline260808/DQ36/DQ36-TB36-TH36   12/12/12 의 첫 번째 조제
                 (최종 폴더는 `-2` 쪽을 담았다) → 격자가 64맵 → 65맵
    "tert-new"   tert-new-baseline/{1..6}-{1..3}      6조건 × 3반복, md5 중복 3장 제외
                 `12/12/48` 은 격자 밖 조건이다 → 격자가 65맵 → 80맵
    "tert-2"     Ratio/mis/tert-2/                    등몰 계열 2차 조제 4맵
  예전 스크립트가 서로 다른 조합을 읽고 있었다 — LOD·게인은 격자 폴더만(65맵),
  재학습·벤치마크는 tert-new 까지(80맵). 그 차이를 여기서 명시한다.
  `Ratio/mis/DQ9-sus-relabeled/` 12맵은 여기 없다. 재현성 계산 전용이고 학습에 넣지
  않는다 (`export_rsd.py` 가 따로 읽는다).
"""
import os, re, glob, hashlib, collections

import paths

DB = paths.DB
FINAL = f"{DB}/Ratio/260814_mixture_final"
SUB = ["DQ", "TBZ", "THI"]
GRID_LEVELS = {3, 6, 12, 24}

# 파일명 → 최종 µM. `TH` 뒤의 `I` 는 `DQ500-TB0-THI5.csv` 하나의 오타를 받아 주는 것이고,
# 끝의 ` (2)` 는 같은 조성을 다시 조제한 `DQ333-TB333-TH333 (2).csv` 다.
_NAME = re.compile(r"^DQ(\d+)-TB(\d+)-THI?(\d+)(?: \(\d+\))?$")

_TYPO = {"DQ500-TB0-THI5": "DQ500-TB0-TH5 로 고쳐 두면 이 예외가 필요 없다"}


def _tag(c, stem=""):
    """세트 이름. 333/333/333 은 두 세트에 다 있어서 파일명으로만 갈린다 —
    `DQ333-TB333-TH333.csv` 는 예전 Ratio_mix, `… (2).csv` 는 등몰 계열의 꼭대기다."""
    if stem.endswith(")"):
        return "equimolar"
    if all(v in GRID_LEVELS for v in c) and all(v > 0 for v in c):
        return "grid"
    if c[0] == c[1] == c[2] and c[0] in (10, 33, 100):
        return "equimolar"
    return "high"


class Map(tuple):
    """(path, conc, set) — conc 는 최종 µM 세 값."""
    __slots__ = ()

    def __new__(cls, path, conc, tag):
        return super().__new__(cls, (path, conc, tag))

    path = property(lambda s: s[0])
    conc = property(lambda s: s[1])
    set = property(lambda s: s[2])
    name = property(lambda s: os.path.basename(s[0])[:-4])


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


REP_SOURCES = ("baseline", "tert-new", "tert-2")


def _replicate_table(sources):
    """최종 폴더에 없는, 같은 조건의 다른 조제. 라벨을 추측하지 않고 적어 둔다.

    출처를 골라야 하는 이유: 어떤 스크립트는 격자 폴더만 읽었고(65맵) 어떤 것은
    tert-new 까지 읽었다(80맵). 여기서 뭉뚱그리면 라벨 수정과 무관하게 숫자가 움직인다.
    """
    t = []
    if "baseline" in sources:
        t.append((f"{DB}/Ratio/Baseline260808/DQ36/DQ36-TB36-TH36_corrected.csv",
                  (12, 12, 12), "grid"))
    if "tert-new" in sources:
        for key, n in {(6, 6, 24): 1, (12, 12, 12): 2, (24, 6, 6): 3,
                       (6, 6, 6): 4, (24, 24, 24): 5, (12, 12, 48): 6}.items():
            # 전부 저농도 세트로 친다. 12/12/48 만 격자 밖이다 — 48 µM 은 3/6/12/24
            # 어디에도 없어서, 격자를 64조건이라 부르면서 이 조건을 세면 65가 된다.
            t += [(f"{DB}/tert-new-baseline/{n}-{r}_corrected.csv", key, "grid")
                  for r in (1, 2, 3)]
    if "tert-2" in sources:
        for stem, key in (("003mM", (10, 10, 10)), ("01mM", (33, 33, 33)),
                          ("03mM", (100, 100, 100)), ("1mM", (333, 333, 333))):
            t.append((f"{DB}/Ratio/mis/tert-2/tert_{stem}_2_corrected.csv", key, "equimolar"))
    return t


def catalogue(sets=("high", "grid", "equimolar"), replicates=False, verbose=True):
    """조건 목록. 같은 내용(md5 동일)인 파일은 한 장으로만 센다."""
    if not os.path.isdir(FINAL):
        raise FileNotFoundError(f"최종농도 폴더가 없다: {FINAL}")
    out, seen, dropped = [], set(), []
    for p in sorted(glob.glob(f"{FINAL}/*.csv")):
        stem = os.path.basename(p)[:-4]
        m = _NAME.match(stem)
        if not m:
            raise ValueError(f"파일명을 최종농도로 못 읽었다: {stem}\n"
                             f"  규약은 DQ<µM>-TB<µM>-TH<µM>.csv 다.")
        c = tuple(int(m.group(i)) for i in (1, 2, 3))
        tag = _tag(c, stem)
        if tag not in sets:
            continue
        h = _md5(p)
        if h in seen:
            dropped.append(stem)
            continue
        seen.add(h)
        out.append(Map(p, c, tag))
    if replicates:
        srcs = REP_SOURCES if replicates is True else tuple(replicates)
        for p, c, tag in _replicate_table(srcs):
            if tag not in sets or not os.path.exists(p):
                continue
            h = _md5(p)
            if h in seen:
                dropped.append(os.path.basename(p))
                continue
            seen.add(h)
            out.append(Map(p, c, tag + "-rep"))
    if verbose and "high" in sets:
        print("[mixtures] 예전 BAD 목록(DQ500TBZ100 · DQ1000TBZ100)은 최종 폴더에 "
              "DQ250-TB50-TH0 · DQ500-TB50-TH0 로 들어 있어 **제외하지 않는다** "
              "— 고농도가 32 → 34조건이다.")
    if verbose:
        n = collections.Counter(m.set for m in out)
        print(f"[mixtures] {len(out)}맵 · " + " · ".join(f"{k} {v}" for k, v in sorted(n.items()))
              + (f" · md5중복 {len(dropped)}장 제외" if dropped else ""))
    return out


def items(sets=("high", "grid", "equimolar"), replicates=False, verbose=True):
    """`train_model` · `benchmark_loo` 가 받는 (path, µM dict, M dict) 목록."""
    out = []
    for m in catalogue(sets, replicates, verbose):
        d = dict(zip(SUB, (float(v) for v in m.conc)))
        out.append((m.path, d, {k: v * 1e-6 for k, v in d.items()}))
    return out


def groups(sets=("high", "grid", "equimolar"), replicates=False, verbose=True):
    """{최종 µM 세 값: [파일들]} — 조건단위 분할·held-out 에 쓴다."""
    g = collections.OrderedDict()
    for m in catalogue(sets, replicates, verbose):
        g.setdefault(m.conc, []).append(m.path)
    return g


def items_for(g, keys):
    """`groups()` 결과에서 조건 몇 개만 골라 학습용 목록으로."""
    out = []
    for k in keys:
        d = dict(zip(SUB, map(float, k)))
        for p in g[k]:
            out.append((p, d, {s: v * 1e-6 for s, v in d.items()}))
    return out


if __name__ == "__main__":
    for tag in ("high", "grid", "equimolar"):
        ms = catalogue((tag,), verbose=False)
        lo = min(min(v for v in m.conc if v) for m in ms)
        hi = max(max(m.conc) for m in ms)
        print(f"{tag:<10} {len(ms):>3}맵  성분농도 {lo}–{hi} µM")
    print()
    catalogue(replicates=True)
