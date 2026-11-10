"""논문용 모델 — 배포 모델(260831 FINAL)과 같은 데이터·전처리·분할, 조성 단계만 PLS.

  배포 모델은 앱 Composition 탭에서 학습됐다(`page_compose._opts`). 여기서는 같은 인자를
  코드로 적어 `method="pls"` 한 가지만 바꾼다. 농도 단계(검량 + 잔차망)는 그대로다.

    학습 82맵 + 순수·blank 13맵 / 시험 20맵  (Pure/mixtures.json 의 role, 라벨은 파일명)
    전처리  Pure/preprocess.json  (baseline 끔 · 500–2500 cm⁻¹ · 맵당 대표 100픽셀)
    검량    배포 모델에 내장된 calibration_spectra_260821.csv 를 그대로 꺼내 쓴다
            (1. results/260821_Calib_144-9/ 의 같은 이름 파일은 내용이 다르다)

    배경    BLK·INK 만. Pure/LEAF-1~3 은 배포 모델 학습(08-31) 뒤인 09-04 에 들어온 파일이라
            뺀다 — 넣으면 학습 집합이 배포 모델과 달라진다(98맵 9708 ≠ 95맵 9498)

  배포 MLP 모델 파일은 건드리지 않는다. 새 모델은 별도 폴더에 저장한다.

  실행:  ../../.venv/bin/python3 -u train_paper_pls.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import os, sys, re, json
import numpy as np

sys.path.insert(0, paths.REPO)
import dl_model
import dataset
from dataset import load_mixture_list, load_mixture_roles, load_preprocess

SKIP_REF = {"LEAF"}                       # 배포 모델 학습 뒤에 추가된 배경 — 위 설명
_discover = dataset.discover_references
dataset.discover_references = lambda d: [
    (c, p) for c, p in _discover(d) if dataset.base_and_batch(c)[0].upper() not in SKIP_REF]

PURE = paths.PURE
FINAL = f"{paths.DATA}/Ratio/260814_mixture_final"
DEPLOYED = f"{paths.RESULTS}/260831_Model_FINAL/mlp_composition_260831_final.dlm"
OUT_DIR = f"{paths.RESULTS}/261110_Model_PLS_paper"
OUT = f"{OUT_DIR}/pls_composition_261110.dlm"
CALIB = f"{OUT_DIR}/calibration_spectra_260821.csv"
LOG_DIR = f"{paths.REPO}/documentation/results/paper_pls_2026-11-10"
N_COMPONENTS = 8
GRID = {3, 6, 12, 24}
_NAME = re.compile(r"DQ(\d+)-TB(\d+)-THI?(\d+)")


def conc_of(path):
    m = _NAME.match(os.path.basename(str(path).replace("\\", "/")))
    return tuple(int(m.group(i)) for i in (1, 2, 3)) if m else None


def comp_dev(ev):
    """조성 오차 ½Σ|Δ| (%p) — blank 열을 떼고 세 성분으로 다시 정규화한 뒤. 맵별."""
    t = np.asarray(ev["true"], float)[:, :3]; p = np.clip(np.asarray(ev["pred"], float)[:, :3], 0, None)
    cs = [conc_of(x) for x in ev["paths"]]
    keep = np.array([c is not None for c in cs])          # 순수·blank 맵은 뺀다
    t, p = t[keep], p[keep]; cs = [c for c in cs if c is not None]
    t = t / t.sum(1, keepdims=True); p = p / np.clip(p.sum(1, keepdims=True), 1e-9, None)
    d = 50 * np.abs(p - t).sum(1)
    grid = np.array([all(v in GRID for v in c) for c in cs])
    return d, grid


def um_within2(ev):
    t = np.asarray(ev["true_uM"], float); p = np.asarray(ev["pred_uM"], float)
    cs = [conc_of(x) for x in ev["paths"]]
    grid = np.array([c is not None and all(v in GRID for v in c) for c in cs])
    ok = (t > 0) & np.isfinite(p) & (p > 0)
    fold = np.where(ok, np.maximum(p / np.clip(t, 1e-12, None), t / np.clip(p, 1e-12, None)), np.nan)
    return fold, grid


def report(tag, m, lines):
    for key, name in (("loo_eval", "조건 held-out (학습 82맵)"), ("test_eval", "시험 20맵")):
        ev = m.get(key)
        if not ev:
            continue
        d, g = comp_dev(ev)
        lines.append(f"{tag:5s} 조성 {name:22s} 64격자 {d[g].mean():5.1f} %p (n={g.sum()})"
                     f" · 격자 밖 {d[~g].mean():5.1f} %p (n={(~g).sum()})")
    ev = (m.get("uM") or {}).get("loo_eval")
    if ev:
        f, g = um_within2(ev)
        fg = f[g]
        lines.append(f"{tag:5s} 농도 조건 held-out 64격자: 2배 이내 {100 * np.nanmean(fg <= 2):.1f} %"
                     f" · 3배 이내 {100 * np.nanmean(fg <= 3):.1f} % · 중앙 배수 {np.nanmedian(fg):.2f}×")


def _here(it):
    """mixtures.json 의 경로는 학습한 기계(S:/…) 것이다 — 파일명으로 이 기계의 최종 폴더에 붙인다."""
    p = f"{FINAL}/{os.path.basename(str(it[0]).replace(chr(92), '/'))}"
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    return (p,) + tuple(it[1:])


def main():
    cfg = load_preprocess(PURE)
    roles = load_mixture_roles(PURE)
    allm = load_mixture_list(PURE, filename_truth=True)      # 라벨 = 파일명, JSON 은 목록·role 만
    train = [_here(it) for it in allm if roles.get(it[0], "train") != "test"]
    test = [_here(it) for it in allm if roles.get(it[0], "train") == "test"]
    print(f"학습 {len(train)}맵 · 시험 {len(test)}맵 · 전처리 {cfg}", flush=True)

    dep = dl_model.load_model(DEPLOYED)
    base = lambda ps: sorted(os.path.basename(str(p).replace(chr(92), "/")) for p in ps)
    if base(dep["test_eval"]["paths"]) != base(it[0] for it in test):
        raise SystemExit("시험 분할이 배포 모델과 다르다 — Pure/mixtures.json 의 role 을 확인할 것")
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
    with open(CALIB, "w", encoding="utf-8", newline="") as fh:   # 배포 모델과 같은 검량
        fh.write(dep["calib_csv_text"])
    print("배포 모델과 시험 분할 동일 · 검량은 배포 모델 내장본", flush=True)

    m = dl_model.train_model(
        PURE, train, calib_path=CALIB, baseline=cfg["baseline"], trim=cfg["trim"],
        method="pls", n_components=N_COMPONENTS, seed=0, epochs=300, use_pretrain=False,
        px_per_map=int(cfg.get("sample_pixels", 100)),
        pixel_sampling=cfg.get("pixel_sampling", "representative"),
        sampling_seed=int(cfg.get("sampling_seed", 0)),
        include_blank=True, nnls_screen=False, equal_volume_mix=False,
        loo=True, test_items=test, progress=lambda s: print("  ", s, flush=True))

    if (m["n_maps"], m["n_train"]) != (dep["n_maps"], dep["n_train"]):
        raise SystemExit(f"학습 집합이 배포 모델과 다르다: {m['n_maps']}맵 {m['n_train']}스펙트럼"
                         f" ≠ {dep['n_maps']}맵 {dep['n_train']}스펙트럼")
    dl_model.save_model(m, OUT)
    lines =[f"논문용 PLS 조성 모델 — 잠재변수 {N_COMPONENTS}개, 학습 {m['n_maps']}맵 {m['n_train']}스펙트럼",
             f"저장: {OUT}", ""]
    report("PLS", m, lines)
    lines.append("")
    report("MLP", dep, lines)
    lines.append("  (MLP 줄은 배포 모델에 저장된 기록. 둘 다 분할 한 번의 결과라 우열 근거는 아니다 —"
                 " 우열은 learner_reeval_2026-11-10 의 반복 CV 를 볼 것)")
    txt = "\n".join(lines); print("\n" + txt)
    with open(f"{LOG_DIR}/summary.txt", "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    with open(f"{LOG_DIR}/eval.json", "w", encoding="utf-8") as fh:
        json.dump({"loo_eval": m.get("loo_eval"), "test_eval": m.get("test_eval"),
                   "uM_loo_eval": (m.get("uM") or {}).get("loo_eval")}, fh, ensure_ascii=False)


if __name__ == "__main__":
    main()
