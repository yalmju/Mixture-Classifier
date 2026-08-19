"""mixtures.json 의 µM 라벨을 파일명과 전수 대조한다 — 그리고 요청 시 수리한다.

  2026-08-19 라벨 오염 사후 점검용. 그날의 일회성 복구는 엄격한 정규식
  (DQ숫자-TB숫자-TH숫자)으로 파싱되는 100건만 고쳤고, 이름이 변칙인 2건은
  건너뛰면서 오염된 conc 를 그대로 남겼다:

    DQ333-TB333-TH333 (2).csv   ← " (2)" 복제 접미사      (conc 9/36/72 = DQ9-sus 것)
    DQ500-TB0-THI5.csv          ← 'TH' 대신 'THI' 표기    (conc 9/72/72, ratio 0 인 TBZ 에도 conc)

  여기서는 앱 저장 경로와 같은 validate.parse_mixture_label 을 쓴다 — 위 두
  이름 모두 처리한다. 원칙은 260814_mixture_final 독트린 그대로: 파일명이
  µM 의 유일한 진실이고, 캐시·행순서에서 온 값은 믿지 않는다.

  실행:  python3 verify_mixture_labels.py            # 대조만 (수정 없음)
         python3 verify_mixture_labels.py --fix      # 백업 만들고 수리
"""
import io
import json
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import paths
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
from validate import parse_mixture_label

MIXJSON = f"{paths.DB}/Pure/mixtures.json"
REF = ["DQ", "TBZ", "THI"]


def wanted(entry):
    """파일명에서 읽은 conc(M 단위). 파싱 불가면 None."""
    nm = os.path.splitext(os.path.basename(entry["path"]))[0]
    amt = parse_mixture_label(nm, REF)
    if not amt:
        return None
    return {k: float(v) * 1e-6 for k, v in amt.items() if float(v) > 0}


def same(a, b):
    return set(a) == set(b) and all(abs(a[k] - b[k]) < 1e-12 for k in a)


def main():
    fix = "--fix" in sys.argv
    d = json.load(io.open(MIXJSON, encoding="utf-8"))
    ok, bad, unparsed = 0, [], []
    for e in d:
        nm = os.path.basename(e["path"])
        w = wanted(e)
        if w is None:
            unparsed.append(nm)
            continue
        if same(w, e.get("conc", {})):
            ok += 1
        else:
            bad.append((nm, e.get("conc", {}), w))
    print(f"일치 {ok} / 불일치 {len(bad)} / 파싱불가 {len(unparsed)}  (전체 {len(d)})")
    for nm in unparsed:
        print(f"  파싱불가: {nm} — 손대지 않음, 이름부터 확인할 것")
    for nm, got, w in bad:
        print(f"  불일치: {nm}\n    지금: {got}\n    파일명: {w}")
    if not bad:
        return
    if not fix:
        print("수리하려면 --fix 를 붙여 다시 실행 (백업을 만들고 고친다)")
        sys.exit(1)
    bak = MIXJSON + ".bak-verify"
    shutil.copy2(MIXJSON, bak)
    for e in d:
        w = wanted(e)
        if w is not None and not same(w, e.get("conc", {})):
            e["conc"] = w
    io.open(MIXJSON, "w", encoding="utf-8").write(
        json.dumps(d, ensure_ascii=False, indent=1))
    print(f"수리 {len(bad)}건 → {MIXJSON}  (백업 {bak})")


if __name__ == "__main__":
    main()
