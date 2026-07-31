# UNMIXR — 작업 인계 (2026-07-31)

브랜치 `claude/todo-implementation-gquqt0`. 앱은 정상 빌드됨(헤드리스 확인).

---

## 1. 이 앱이 하는 일 (5단계 워크플로우)

```
Samples → Model → Recovery → Quantify → Real data
 (1)      (2)      (3)                    (5)
```

1. **Samples** — pure 레퍼런스 맵을 물질 클래스로 묶고, **known-ratio 혼합물 목록**을 준비
2. **Model** — 조성 모델을 **한 번** 학습 (Recovery/Real은 이걸 적용만)
3. **Recovery** — known-ratio 혼합물로 회수율·응답계수 확인
4. **Quantify** — 검량선
5. **Real data** — 미지 시료 맵 → 조성 + 대략 농도

대상 물질: **DQ**(diquat) · **TBZ**(thiabendazole) · **THI**(thiram), 배경 **BLK**.

---

## 2. 핵심 설계 원칙 (깨뜨리지 말 것)

- **한 번 학습, 뒷단은 적용만.** Model 탭에서 학습 → `MODEL_BUS`(ui_common)로 발행 →
  Recovery·Real이 자동 채택. 뒷단에서 재학습하지 않는다.
- **혼합물 목록은 Samples가 소유.** `<dataset>/mixtures.json`에 저장, `MIXTURE_BUS`로 통지.
  Model·Recovery는 읽기만 한다.
- **학습 노브는 학습부에만.** 뒷단엔 노브를 두지 않는다.
- **누수 금지.** 학습에 쓴 혼합물은 반드시 held-out 예측으로 채점.
  분할은 **맵 단위**(같은 맵의 픽셀이 train/test에 걸치면 누수).
- **그림은 figure-set용**: 제목·캡션 굽지 않기, 그래프+범례만, 설명은 README에.
  스타일은 `Canvas.new_ax()`(cnsplots) 사용. export는 투명 배경 300dpi.
  **모든 그림은 옆에 CSV를 같이 써서 재그리기 가능해야 한다.**

---

## 3. 현재 성능 (정직한 수치, 5-fold)

| 방법 | 조성오차 (held-out) |
|---|---|
| **MLP** | **16.7% ± 3.9%** |
| PLS | 24.1% ± 1.8% |
| NNLS (고전 baseline) | 28.1% ± 6.2% |

- 검출 AUC: MLP 0.912 vs NNLS 0.903 → **검출은 사실상 동률**, 차별점은 **정량**.
- 단일 분할 하나만 보면 11.6%까지 나오지만 **과대평가**. 5-fold 값을 쓸 것.
- 절대 µM은 **반정량(order-of-magnitude)** — 세션별 gain 차이로 크기는 전이되지 않음.
  조성(비율)은 전이됨.

---

## 4. 주요 파일

| 파일 | 역할 |
|---|---|
| `dl_model.py` | 조성 모델의 심장. `train_model`(4방법+µM+LOO+test배치), `benchmark_loo`, `kfold_stability`, `apply_model`, `apply_model_pixels`, `apply_recovery`, `_fit_predict`, `_map_spectra` |
| `page_compose.py` | Model 탭의 조성 패널 (학습은 **서브프로세스**로 → GUI 안 멈춤) |
| `page_model.py` | Model 탭 = `[Composition model | Pixel classifier]` 모드 전환 |
| `page_samples.py` | 클래스 표 + **Mixtures 표**(Role train/test, Split 20%) |
| `page_validate.py` | Recovery |
| `page_real.py` | Real data |
| `triangle_figs.py` | 논문용 삼각형 (accuracy / RGB / **compare 좌우 2패널**) — 앱과 스크립트 공용 |
| `experiments/dl/*.py` | 오프라인 벤치마크 스크립트 (docs 그림 생성) |

---

## 5. 해결한 함정들 (다시 만들지 말 것)

- **GUI 프리즈**: torch가 GIL을 안 놓음 → 학습을 **별도 프로세스**(spawn)로. Cancel은 자식 kill,
  메인스레드에서 `join()`/`wait()` 하지 말 것(그 자체가 프리즈였음).
- **mixtures.json이 빈 리스트로 덮어써짐**: 로드 중 체크박스 시그널이 저장을 발동시켰음.
  `_save_mix`는 `_loading` 중 early-return.
- **삼성분 7개를 4개로 셈**: 성분 수를 "분율 2% 초과"로 세면 THI1000TBZ1000DQ10(DQ 0.5%)가
  이성분으로 강등됨 → **투입됐으면 무조건 카운트**.
- **train_model이 모르는 method를 몰래 MLP로 바꿈**: NNLS·MLP 숫자가 소수점까지 같아서 발각.
- **Recovery 회수율이 100%로 수렴**: 학습한 혼합물에 그 모델을 적용해서. → held-out 예측 재사용.
- **맵당 학습행 1개**: 400픽셀 맵을 평균 1개로 뭉갬 → 34행뿐이라 과적합.
  `pixels/map` 노브로 픽셀 행 추가(단, LOO는 맵 단위 유지).

---

## 6. 지금 붙잡고 있던 문제 (다음 작업)

**Real 탭에서 실측 샘플의 배경 판정이 안 됨.**

증상 (`260723_mixture(bg).csv`, 2100픽셀):
- `hide low-R²`가 **1860픽셀(89%)을 제외** → hit 0%
- NNLS는 정답에 가까움: hit 픽셀 비율 `DQ 12 : TBZ 8 : **THI 80**`
- 반면 DL 맵 조성은 `TBZ 89%`로 **뒤집힘** (배경 오염 때문 — 방금 고침, 미검증)

이미 반영된 것:
- DL 조성 계산이 **맵 전체 평균 → 신호 픽셀(hit mask 또는 상위 20%)** 로 변경 (`1e47f3f`)
- `learn background (BLK)` 옵션 (blank 맵 픽셀을 4번째 클래스로) — **기본 꺼짐**,
  사용자 방침은 "학습은 깨끗하게, 배경은 Real에서 판정"

다음에 할 일:
1. `hide low-R²` 끄거나 0.2로 → 다시 Unmix → THI가 우세로 나오는지 확인
2. 여전히 안 되면 **Real에 "Load background…"** 를 추가해 실측 배경(`ACF_PEST_DB/Pest/BLk/`)을
   지정하고, 그 배경 대비로 픽셀을 판정하도록 배선
3. 픽셀 파이 / 맵 DL 조성 / DL 농도 **세 숫자가 서로 어긋나는 문제**도 같이 정리

---

## 7. 성능을 더 올리려면

에폭이나 픽셀 수가 아니라 **서로 다른 조성의 개수**가 병목이다. 현재 34개 중 삼성분은 7개뿐이고
대부분 `1000:1000:x` 형태라 조성 공간 내부가 비어 있다. **중간 조성(예: 500:300:200)** 을 몇 개
측정해 추가하면 held-out 오차가 눈에 띄게 내려갈 것.
