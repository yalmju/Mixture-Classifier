"""데이터 경로 한 곳 — 특히 검량 파일.

  스크립트마다 `f"{DB}/Ratio/results/calibration_spectra.csv"` 를 하드코딩하고 있었는데
  그 폴더가 사라졌다. `train_model` 은 `if calib_path and use_pretrain:` 로 **문자열이
  비어 있지 않은지만** 보고 들어간 뒤, 파일이 없으면 `except: pre = None` 으로 **조용히
  물리 사전학습을 통째로 건너뛴다.** 그래서 2026-08-11 실행에는 사전학습이 들어갔고
  2026-08-12 재실행에는 안 들어갔는데 아무 경고도 없었다 (조성 12.9% → 13.7%).

  여기서는 **찾거나 죽는다.** 조용히 다른 결과를 내는 것보다 낫다.

  `DB` · `REPO` 도 같은 이유로 여기서 찾는다. 두 대(맥·윈도우)에서 같은 Google Drive 를
  보는데 마운트 경로가 다르다. 스크립트마다 맥 경로를 박아 두면 다른 기계에서는 glob 이
  0개를 돌려주고 — 예외 없이 — 세트가 통째로 빠진 결과가 나온다.
  `ACF_PEST_DB` · `MIXTURE_REPO` 환경변수로 덮어쓸 수 있다.
"""
import os

_MAC = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브"
_WIN = "S:/Google Drive/내 드라이브"


def _first(cands, what):
    for p in cands:
        if p and os.path.isdir(p):
            return p
    raise FileNotFoundError(f"{what} 를 못 찾았다.\n" + "\n".join(f"  찾아본 곳: {p}" for p in cands if p))


DB = _first([os.environ.get("ACF_PEST_DB"),
             f"{_MAC}/ACF_PEST_DB",
             f"{_WIN}/ACF_PEST_DB"], "ACF_PEST_DB")

REPO = _first([os.environ.get("MIXTURE_REPO"),
               os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
               f"{_MAC}/github/Mixture Classifier",
               f"{_WIN}/github/Mixture Classifier"], "Mixture Classifier 저장소")

PURE = f"{DB}/Pure"

# 결과가 나가는 곳. `INTERP_OUT` 로 딴 데를 가리키면 기존 export 를 덮어쓰지 않고
# 나란히 놓고 비교할 수 있다.
#
# 2026-08-14 에 `260808_data interpret` → `260814_data interpret` 로 옮겼다. 고농도
# 라벨 정정(원액 → 최종농도) 전후를 한 폴더에 섞으면 어느 판인지 구분이 안 된다.
# **08-08 폴더는 손대지 않았다** — 정정 전 판이 그대로 남아 있어 대조에 쓸 수 있다.
INTERP = os.environ.get("INTERP_OUT") or f"{DB}/260814_data interpret"

# 같은 md5(ce184a98…) 사본이 여러 폴더에 있다. 먼저 찾히는 것을 쓴다.
_CALIB = [
    f"{DB}/Ratio/results/calibration_spectra.csv",        # 문서·스크립트가 적어온 자리
    f"{DB}/STD/260727_Spec/calibration_spectra.csv",
    f"{DB}/STD/260727_VIP/calibration_spectra.csv",
    f"{DB}/Pest/Standard/260729/calibration_spectra.csv",
    f"{DB}/code results/260731-22/stand/calibration_spectra.csv",
]


def calibration(required=True):
    """검량 희석계열 CSV 경로. 못 찾으면 (기본값) 예외를 던진다."""
    for p in _CALIB:
        if os.path.exists(p):
            return p
    if not required:
        return None
    raise FileNotFoundError(
        "calibration_spectra.csv 를 못 찾았다. 이게 없으면 물리 사전학습이 조용히 꺼진다.\n"
        + "\n".join(f"  찾아본 곳: {p}" for p in _CALIB))


def check_pretrain(calib_path, use_pretrain=True):
    """사전학습이 실제로 가능한 상태인지 확인하고 무엇을 쓰는지 출력한다."""
    if not use_pretrain:
        print("물리 사전학습: 끔 (use_pretrain=False)")
        return None
    if not calib_path or not os.path.exists(calib_path):
        raise FileNotFoundError(f"사전학습을 켰는데 검량 파일이 없다: {calib_path!r}")
    from io_utils import load_calibration_csv
    _ax, names, dils = load_calibration_csv(calib_path)
    print(f"물리 사전학습: 켬 · {os.path.relpath(calib_path, DB)} · "
          f"{names} · 점 {[len(d[0]) for d in dils]}")
    return calib_path
