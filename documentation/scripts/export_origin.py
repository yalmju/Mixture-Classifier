"""모든 그림의 숫자를 CSV 로 뽑는다 — Origin 에서 다시 그리기 위한 것.

  분석 결과가 `.npz` · `.pkl` · 콘솔 텍스트로만 남아 있어서, 패널을 Origin 으로 옮길 때
  자료가 없는 그림이 생겼다(농도 파리티 패널이 그래서 matplotlib 그대로 붙었다).
  여기서 한 번에 전부 tidy CSV 로 내린다. 한 파일 = 한 패널.

  긴 형식(long form)으로 쓴다 — 관측 하나가 한 줄. Origin 이 그룹 열로 바로 나눈다.
  히트맵만 행렬 형식으로 따로 낸다(행렬이 아니면 Origin 이 heat map 을 못 만든다).

  실행:  python3 -u export_origin.py [출력폴더]
"""
import os, re, sys, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
import grid_figs as GF

DB = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB"
INTERP = os.environ.get("INTERP_DIR") or f"{DB}/260808_data interpret"   # 테스트용 우회
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{INTERP}/origin_data"
SUB = ["DQ", "TBZ", "THI"]
os.makedirs(OUT, exist_ok=True)
made, missing = [], []


def write(name, header, rows, note=""):
    if not rows:
        missing.append((name, note or "행이 없다")); return
    import csv
    p = os.path.join(OUT, name)
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    made.append((name, f"{len(rows)}행 × {len(header)}열", note))


def need(path, name, note):
    if os.path.exists(path):
        return True
    missing.append((name, f"{note} — 없음: {os.path.relpath(path, DB)}"))
    return False


# ------------------------------------------------------------------ 조성: 조건별 (삼각형·파이·격자)
def load_retrain(path):
    rows, concs = [], []
    for l in open(path):
        if not re.search(r"\|\s+[\d.]+% \|", l):
            continue
        p = [x.strip() for x in l.split("|")]
        k = tuple(int(float(v)) for v in p[0].split("/"))
        rows.append(("/".join(str(v) for v in k) + " uM",
                     np.array([float(v) for v in p[1].split("/")]),
                     np.array([float(v) for v in p[2].split("/")])))
        concs.append(k)
    return rows, concs


RETRAIN = f"{INTERP}/64conditions/retrain64_blanksips.txt"
if need(RETRAIN, "composition_conditions.csv", "조건별 조성 (채택 구성)"):
    rows, concs = load_retrain(RETRAIN)
    h, b = GF.condition_table(rows, concs, SUB)
    write("composition_conditions.csv", h, b, "삼각형·파이·격자 패널 전부 이 표에서 나온다")
    for nm, mh, mb in GF.error_matrix_tables(rows, concs, SUB):
        write(f"composition_error_matrix_{nm}.csv", mh, mb, "히트맵용 행렬")

# ------------------------------------------------------------------ 농도: 파리티 + 배수오차 CDF
# 이 패널이 자료가 없어서 matplotlib 출력 그대로 들어갔던 것이다.
CONC = f"{HERE}/conc38_result_cliponly.npz"
if need(CONC, "concentration_parity.csv", "농도 파리티 (패널 h)"):
    z = np.load(CONC, allow_pickle=True)
    C, EST, F, V = z["C"], z["est"], z["fold"], z["valid"]
    paths = [str(p) for p in z["paths"]]
    design = np.array([48 not in C[i] for i in range(len(C))])   # THI 48 = 설계 밖
    body = []
    for i in range(len(C)):
        for j, s in enumerate(SUB):
            body.append([os.path.basename(paths[i]).replace("_corrected.csv", ""), s,
                         f"{C[i, j]:g}", f"{EST[i, j]:.4f}", f"{F[i, j]:.4f}",
                         str(int(bool(V[i, j]))), str(int(bool(design[i]))),
                         str(int(F[i, j] < 2))])
    write("concentration_parity.csv",
          ["map", "compound", "true_uM", "estimated_uM", "fold_error",
           "valid", "in_design", "within_2x"], body,
          "패널 h 왼쪽 3개 — 성분별로 나눠 그린다. in_design=0 은 THI 48 µM (설계 밖)")

    cdf = []
    for j, s in enumerate(SUB):
        # 파리티·CDF·요약이 모두 같은 행을 써야 한다. 예전엔 CDF 만 valid 를 안 걸러서
        # 그림의 곡선과 옆에 적힌 "2배 이내 %" 가 서로 다른 n 에서 나왔다.
        ok = design & np.asarray(V)[:, j] & np.isfinite(F[:, j])
        f = np.sort(F[ok, j])
        for i, v in enumerate(f):
            cdf.append([s, f"{v:.4f}", f"{100.0 * (i + 1) / len(f):.3f}"])
    write("concentration_fold_cdf.csv", ["compound", "fold_error", "cumulative_pct"], cdf,
          "패널 h 오른쪽 — step plot. 2× 세로선은 Origin 에서 그으면 된다")

    summ = []
    for j, s in enumerate(SUB):
        ok = design & np.asarray(V)[:, j] & np.isfinite(F[:, j])
        f = F[ok, j]
        summ.append([s, str(int(ok.sum())), f"{np.median(f):.4f}",
                     f"{100.0 * (f < 2).mean():.2f}"])
    write("concentration_summary.csv",
          ["compound", "n", "median_fold_error", "within_2x_pct"], summ, "패널 h 주석 숫자")

# ------------------------------------------------------------------ 등몰 삼원 농도 시리즈
NN = f"{HERE}/tert260805_nnls.npz"
TR = f"{HERE}/tert260805_result.npz"
if need(NN, "tert_series_composition.csv", "등몰 삼원 시리즈 (누적 막대 패널)"):
    n = np.load(NN, allow_pickle=True)
    per, comp = n["per"], n["comp"]
    body = []
    for i, p in enumerate(per):
        for j, s in enumerate(SUB):
            body.append([f"{p:.4g}", "NNLS", s, f"{comp[i, j]:.3f}"])
    if os.path.exists(TR):
        t = np.load(TR, allow_pickle=True)
        for i in range(len(t["per_uM"])):
            for j, s in enumerate(SUB):
                body.append([f"{t['per_uM'][i]:.4g}", "MLP", s, f"{100 * t['pred'][i, j]:.3f}"
                             if t["pred"][i, j] <= 1.0 else f"{t['pred'][i, j]:.3f}"])
    write("tert_series_composition.csv",
          ["per_compound_uM", "method", "compound", "composition_pct"], body,
          "참 조성은 모든 농도에서 33.3/33.3/33.3 — 점선 기준")

# ------------------------------------------------------------------ 학습곡선 (조건 수 효과)
CURVE = f"{HERE}/curve38_result.npz"
if need(CURVE, "learning_curve.csv", "조건 수 대비 오차"):
    z = np.load(CURVE, allow_pickle=True)
    write("learning_curve.csv", ["n_conditions", "mean_error_pct", "sd_pct"],
          [[str(int(s)), f"{100 * m:.3f}" if m <= 1 else f"{m:.3f}",
            f"{100 * d:.3f}" if d <= 1 else f"{d:.3f}"]
           for s, m, d in zip(z["sizes"], z["means"], z["sds"])],
          "35조건 시절 것 — 64조건판은 아직 없다")

# ------------------------------------------------------------------ NNLS 기준선
NNL = f"{HERE}/nnls_lowconc.npz"
if need(NNL, "nnls_baseline.csv", "NNLS 조건별 오차 (방법 비교 막대)"):
    z = np.load(NNL, allow_pickle=True)
    write("nnls_baseline.csv", ["condition", "nnls_error_pct"],
          [[str(k), f"{100 * e:.3f}" if e <= 1 else f"{e:.3f}"]
           for k, e in zip(z["keys"], z["errs"])],
          "방법 비교 막대의 NNLS 막대")

# ------------------------------------------------------------------ 신호(응답) 값
for tag, fn in (("clip-only", "conc38_signals_cliponly.json"),
                ("ALS", "conc38_signals.json"),
                ("ink-diagnostic", "dqfloor_signals.json")):
    p = f"{HERE}/{fn}"
    if not os.path.exists(p):
        missing.append((fn, "신호값")); continue
    d = json.load(open(p))
    body = []
    for path, v in d.items():
        nm = os.path.basename(path).replace("_corrected.csv", "")
        if isinstance(v, dict):
            for k2, v2 in v.items():
                body.append([nm, str(k2), f"{float(v2):.6g}"])
        else:
            for j, v2 in enumerate(np.atleast_1d(v)):
                body.append([nm, SUB[j] if j < len(SUB) else str(j), f"{float(v2):.6g}"])
    write(f"signals_{tag.replace('-', '_')}.csv", ["map", "channel", "signal"], body,
          f"{tag} 전처리의 성분별 신호")

# ------------------------------------------------------------------ 방법 비교 · 검출 ROC
#   bench64.py 가 `panels/` 에 이미 tidy 로 써 둔다. 여기서는 Origin 폴더 한 곳에 모이도록
#   같은 이름 규약으로 옮겨 담기만 한다 — 숫자는 손대지 않는다.
#   위의 `nnls_baseline.csv` 와 겹쳐 쓰지 말 것: 그쪽은 맵평균 특징의 옛 기준선이고,
#   표 1 의 NNLS 막대는 여기 `method_comparison.csv`(같은 픽셀·같은 fold) 가 정본이다.
import csv as _csv
for _src, _dst, _note in (
        ("panel_d_method_comparison.csv", "method_comparison.csv",
         "다섯 방법 조성오차·검출 AUC (표 1 · 패널 d)"),
        ("panel_e_detection_roc.csv", "detection_roc.csv",
         "검출 ROC — method 열로 그룹, fpr/tpr (패널 e)"),
        ("panel_de_predictions.csv", "method_predictions.csv",
         "방법별 조건 단위 참값 vs 예측 (파리티·잔차용)"),
        ("learning_curve64.csv", "learning_curve64.csv",
         "64조건판 학습곡선 — 위 learning_curve.csv(35조건판)와 **겹쳐 그리지 말 것**, "
         "데이터 규약이 다르다")):
    _p = f"{INTERP}/panels/{_src}"
    if not need(_p, _dst, f"{_note} — bench64.py 가 끝나야 생긴다"):
        continue
    with open(_p, newline="", encoding="utf-8-sig") as _fh:
        _rows = list(_csv.reader(_fh))
    write(_dst, _rows[0], _rows[1:], _note)

#   방법별로 따로도 낸다 — 다섯을 한 장에 겹쳐 그리는 것 말고 하나씩 보고 싶을 때.
#   ROC 는 방법마다 점 개수가 달라 긴 형식 한 장으로는 Origin 에서 나누기 번거롭다.
def _split_by_method(src, stem, head_note, keep=None):
    p = f"{INTERP}/panels/{src}"
    if not os.path.exists(p):
        return {}
    with open(p, newline="", encoding="utf-8-sig") as fh:
        rows = list(_csv.reader(fh))
    head, body = rows[0], rows[1:]
    by = {}
    for r in body:
        by.setdefault(r[0], []).append(r[1:])
    for meth, rs in by.items():
        write(f"{stem}_{meth.replace(' ', '_')}.csv", head[1:], rs, f"{meth} — {head_note}")
    return {"head": head, "by": by} if keep else {}


_split_by_method("panel_e_detection_roc.csv", "detection_roc", "검출 ROC (fpr, tpr)")
_pred = _split_by_method("panel_de_predictions.csv", "method_predictions",
                         "조건 단위 참값 vs 예측", keep=True)
if _pred:
    #   조건 단위 조성 오차 = 0.5·Σ|예측−참| — bench64 가 표 1 평균을 낼 때 쓰는 그 정의다.
    #   평균 하나가 아니라 조건별 값이 있어야 상자그림·짝지은 비교를 다시 그릴 수 있다.
    _h = _pred["head"]
    _ti = [i - 1 for i, h in enumerate(_h) if h.startswith("true_")]
    _pi = [i - 1 for i, h in enumerate(_h) if h.startswith("pred_")]
    _err = []
    for _meth, _rs in _pred["by"].items():
        for _r in _rs:
            _e = 0.5 * sum(abs(float(_r[b]) - float(_r[a])) for a, b in zip(_ti, _pi))
            _err.append([_meth, _r[0], f"{_e:.3f}"])
    write("method_error_by_condition.csv", ["method", "condition", "composition_error_pct"],
          _err, "표 1 평균의 원자료 — 상자그림·방법 간 짝지은 비교")

# ------------------------------------------------------------------ 정리
print(f"\n출력 → {OUT}\n")
print("만들어진 것")
for n, shape, note in made:
    print(f"  ✓ {n:<44} {shape:<16} {note}")
if missing:
    print("\n자료가 없어서 못 만든 것 — 해당 스크립트를 다시 돌려야 한다")
    for n, why in missing:
        print(f"  ✗ {n:<44} {why}")
