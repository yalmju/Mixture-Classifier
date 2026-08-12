"""방법 비교(패널 d)와 검출 ROC(패널 e) — 64조건 격자로 다시.

  디스크에 남아 있던 유일한 벤치마크 export 는 `code results/20260731-03/model bench/`
  로 2026-07-31 자다. 64조건 격자 이전이고 숫자도 논문 그림과 다르다. 자료가 없으니
  다시 돌린다.

  `retrain64.py` 와 **같은 데이터 수집**을 쓴다 (DQ9-sus 제외, md5 중복 제외, 파일명 ÷3).
  다만 `benchmark_loo` 는 방법 비교용이라 `include_blank` · `sim_iso` 를 받지 않는다 —
  **채택 구성이 아니라 base 구성에서의 방법 비교**다. 표에 그렇게 적을 것.

  실행:  python3 -u bench64.py [epochs=120] [px_per_map=20]
         5개 방법 × 조건단위 LOO 라 오래 걸린다 (1–3시간).
"""
import sys, os, re, glob, csv, json, hashlib, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
import dl_model

DB = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB"
PURE = f"{DB}/Pure"
import paths
CALIB = paths.calibration()          # 하드코딩하면 조용히 사전학습이 꺼진다
OUT = f"{DB}/260808_data interpret/panels"
BAD = {"DQ500TBZ100", "DQ1000TBZ100"}
SUB = ["DQ", "TBZ", "THI"]
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
PXPM = int(sys.argv[2]) if len(sys.argv) > 2 else 20
os.makedirs(OUT, exist_ok=True)


def parse_hi(nm):
    c = {"DQ": 0.0, "TBZ": 0.0, "THI": 0.0}
    for k, v in re.findall(r"(DQ|TBZ|TH[I1])(\d+)", nm):
        c["THI" if k.startswith("TH") else k] += float(v)
    return c


items, seen = [], set()


def add(path, conc):
    h = hashlib.md5(open(path, "rb").read()).hexdigest()
    if h in seen:
        return
    seen.add(h)
    items.append((path, conc, {k: v * 1e-6 for k, v in conc.items()}))


for p in sorted(glob.glob(f"{DB}/Ratio/Ratio_mix/*orrected.csv")):
    nm = os.path.basename(p).split("_corrected")[0].split("-corrected")[0]
    if nm in BAD:
        continue
    c = parse_hi(nm)
    if sum(c.values()):
        add(p, c)
for p in sorted(glob.glob(f"{DB}/Ratio/Baseline260808/DQ*/DQ*_corrected.csv")):
    if "DQ9-sus" in p:
        continue
    m = re.match(r"DQ(\d+)-TB(\d+)-TH(\d+)", os.path.basename(p))
    add(p, dict(zip(SUB, [int(m.group(i)) / 3.0 for i in (1, 2, 3)])))
OLD = {(6, 6, 24): 1, (12, 12, 12): 2, (24, 6, 6): 3, (6, 6, 6): 4, (24, 24, 24): 5,
       (12, 12, 48): 6}
for key, n in OLD.items():
    for r in (1, 2, 3):
        p = f"{DB}/tert-new-baseline/{n}-{r}_corrected.csv"
        if os.path.exists(p):
            add(p, dict(zip(SUB, map(float, key))))

print(f"{len(items)}맵 · epochs={EPOCHS} · px/map={PXPM}", flush=True)
t0 = time.time()
res = dl_model.benchmark_loo(PURE, items, calib_path=CALIB, baseline=True, trim=None,
                             epochs=EPOCHS, seed=0, use_pretrain=True, px_per_map=PXPM,
                             progress=lambda s: print("   ", s, flush=True))
subs = res["subs"]
nb = [s for s in subs if s in SUB]
idx = [subs.index(s) for s in nb]

METHODS = [("nnls", "NNLS"), ("pls", "PLS"), ("rf", "Random Forest"),
           ("cnn", "1D-CNN"), ("mlp", "MLP")]
THR = 0.05                                   # 존재 판정: 참 분율 5% 초과


def roc(T, P):
    """검출 ROC — 성분을 다 모아서(pooled) 존재/부재를 예측 분율로 점수 매긴다."""
    y = np.concatenate([[1 if T[i][j] > THR else 0 for i in range(len(T))] for j in idx])
    s = np.concatenate([[P[i][j] for i in range(len(P))] for j in idx])
    o = np.argsort(-s)
    y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    if not tp[-1] or not fp[-1]:
        return None, None, float("nan")
    tpr = np.concatenate([[0.0], tp / tp[-1]]); fpr = np.concatenate([[0.0], fp / fp[-1]])
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


metrics, rocrows, predrows = [], [], []
for key, label in METHODS:
    if key not in res:
        continue
    T = np.asarray(res[key]["true"], float); P = np.asarray(res[key]["pred"], float)
    err = float(np.mean([0.5 * np.abs(P[i][idx] - T[i][idx]).sum() for i in range(len(T))]))
    f, t, auc = roc(T, P)
    metrics.append([label, key, f"{100 * err:.3f}", f"{auc:.4f}", str(len(T))])
    if f is not None:
        for a, b in zip(f, t):
            rocrows.append([label, f"{a:.5f}", f"{b:.5f}"])
    for i in range(len(T)):
        predrows.append([label, str(i + 1)] + [f"{100 * T[i][j]:.3f}" for j in idx]
                        + [f"{100 * P[i][j]:.3f}" for j in idx])
    print(f"  {label:<14} 조성오차 {100 * err:5.1f}%   AUC {auc:.3f}", flush=True)


def write(name, head, body):
    with open(f"{OUT}/{name}", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(head); w.writerows(body)
    print(f"  ✓ {name}  {len(body)}행", flush=True)


write("panel_d_method_comparison.csv",
      ["method", "key", "composition_error_pct", "detection_auc", "n_conditions"], metrics)
write("panel_e_detection_roc.csv", ["method", "fpr", "tpr"], rocrows)
write("panel_de_predictions.csv",
      ["method", "condition"] + [f"true_{s}_pct" for s in nb] + [f"pred_{s}_pct" for s in nb],
      predrows)
print(f"\n{time.time() - t0:.0f}s · → {OUT}")
print("표에 적을 것: leave-one-condition-out · base 구성(blank/sips 없음) — "
      "benchmark_loo 가 그 인자를 받지 않는다.")
