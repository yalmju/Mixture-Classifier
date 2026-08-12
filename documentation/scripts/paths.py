"""데이터 경로 한 곳 — 특히 검량 파일.

  스크립트마다 `f"{DB}/Ratio/results/calibration_spectra.csv"` 를 하드코딩하고 있었는데
  그 폴더가 사라졌다. `train_model` 은 `if calib_path and use_pretrain:` 로 **문자열이
  비어 있지 않은지만** 보고 들어간 뒤, 파일이 없으면 `except: pre = None` 으로 **조용히
  물리 사전학습을 통째로 건너뛴다.** 그래서 2026-08-11 실행에는 사전학습이 들어갔고
  2026-08-12 재실행에는 안 들어갔는데 아무 경고도 없었다 (조성 12.9% → 13.7%).

  여기서는 **찾거나 죽는다.** 조용히 다른 결과를 내는 것보다 낫다.
"""
import os

DB = "/Users/seungki2/Library/CloudStorage/GoogleDrive-seungki1015@gmail.com/내 드라이브/ACF_PEST_DB"

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
