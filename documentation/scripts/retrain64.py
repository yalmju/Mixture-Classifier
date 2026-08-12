"""저농도 64조건 완전격자(Baseline260808 DQ 9/18/36/72 × TBZ × THI)로 재학습.
4-fold 조건단위 검증.

  `DQ9-sus` 는 혼합이 잘못된 것으로 의심되는 배치라 **제외**한다.
  같은 내용(md5 동일)인 파일이 폴더를 넘나들며 중복 저장돼 있어 그것도 한 장으로 친다.
"""
import sys, os, re, glob, hashlib
import numpy as np

REPO = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/github/Mixture Classifier"
sys.path.insert(0, REPO)
import dl_model
from real_data import load_map

DB = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB"
PURE, CALIB = f"{DB}/Pure", f"{DB}/Ratio/results/calibration_spectra.csv"
BAD = {"DQ500TBZ100", "DQ1000TBZ100"}
SUB = ["DQ", "TBZ", "THI"]


def parse_hi(nm):
    c = {"DQ": 0.0, "TBZ": 0.0, "THI": 0.0}
    for k, v in re.findall(r"(DQ|TBZ|TH[I1])(\d+)", nm):
        c["THI" if k.startswith("TH") else k] += float(v)
    return c


HI = []
for p in sorted(glob.glob(f"{DB}/Ratio/Ratio_mix/*orrected.csv")):
    nm = os.path.basename(p).split("_corrected")[0].split("-corrected")[0]
    if nm in BAD:
        continue
    c = parse_hi(nm)
    if sum(c.values()):
        HI.append((p, c, {k: v * 1e-6 for k, v in c.items()}))

# ---- 저농도: 조성(최종 µM)을 키로 묶는다. 같은 조성의 다른 파일은 반복으로 취급 ----
LO, _seen = {}, set()


def _add(key, p):
    """같은 내용의 파일이 두 폴더에 있으면 한 장으로만 센다 (반복 수가 부풀지 않게)."""
    h = hashlib.md5(open(p, "rb").read()).hexdigest()
    if h in _seen:
        print(f"  [중복 제외] {os.path.relpath(p, DB)}")
        return
    _seen.add(h)
    LO.setdefault(key, []).append(p)


for p in sorted(glob.glob(f"{DB}/Ratio/Baseline260808/DQ*/DQ*_corrected.csv")):
    if "DQ9-sus" in p:                      # 혼합 의심 배치 — 제외
        continue
    m = re.match(r"DQ(\d+)-TB(\d+)-TH(\d+)", os.path.basename(p))
    _add(tuple(int(m.group(i)) // 3 for i in (1, 2, 3)), p)
OLD = {(6, 6, 24): 1, (12, 12, 12): 2, (24, 6, 6): 3, (6, 6, 6): 4, (24, 24, 24): 5, (12, 12, 48): 6}
for key, n in OLD.items():
    for r in (1, 2, 3):
        p = f"{DB}/tert-new-baseline/{n}-{r}_corrected.csv"
        if os.path.exists(p):
            _add(key, p)

keys = sorted(LO)
print(f"고농도 {len(HI)}맵 · 저농도 {len(keys)}조건 {sum(len(v) for v in LO.values())}맵")


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

print(f"\n{FOLDS}-fold 조건단위 검증 · epochs={EPOCHS}")
print(f"{'조성 (최종 µM)':>16} | {'참 %':>18} | {'예측 %':>18} | {'오차':>6} | 우세")
allerr, nnls_err = [], []
for fi, te in enumerate(folds):
    tr = [k for k in keys if k not in te]
    m = dl_model.train_model(PURE, HI + items_for(tr), calib_path=CALIB, baseline=True,
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
print(f"\n  저농도 {len(allerr)}조건 조성 오차  평균 {np.mean(allerr):5.1f}%  중앙 {np.median(allerr):5.1f}%")
print(f"  [비교] 6조건 18.8% · 35조건 12.9% · 반복산포 하한 9.6%")
