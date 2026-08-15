# 그림 재작업 목록 (2026-08-15)

고농도 라벨 정정(원액 → 최종농도, `FINDINGS_2026-08-12.md` 머리말) 이후 그림 세트에서
다시 그릴 것과 그대로 둘 것. **데이터는 전부 이미 뽑혀 있다** —
`ACF_PEST_DB/260814_data interpret/` 의 `panels/` 와 `origin_data/`.
Origin 에서 재-import 만 하면 된다.

## 그대로 (재작업 없음)

| 그림 | 근거 |
|---|---|
| Fig 1 전체 (단일성분 특성화·VIP·PCA) | 혼합물 라벨 무관 |
| Fig 2(a) 검량·LOD (1.56 / 1.74 / 0.37 µM) | `lod_loq.csv` 어제와 바이트 동일 |
| Fig 3(d)(e) SERS 글씨 merge·파이 | NNLS 는 라벨 무관 · MLP 파이는 260806 모델(비율 학습) 유지 |
| 글씨 맵 µM (10.1 / 14.2 / 2.7) | 질량수지 — 혼합물 라벨을 안 씀 |

## 다시 (6개)

| 그림 | 새 데이터 | 캡션에 들어갈 새 수치 |
|---|---|---|
| Fig 2(b) 등몰 조성 막대 | `panels/panel_c_tert_series.csv` | NNLS 2조제 평균으로: 10 µM **49/21/30** · 33 µM **49/32/18** · 100·300 µM 0/0/100 그대로 |
| Fig 2(c) 방법 비교 | `panels/panel_d_method_comparison.csv`<br>`origin_data/method_error_by_mixture_type.csv` | n **70**. MLP **16.3** · PLS 17.1 · RF 19.4 · NNLS 21.5 · CNN **37.1**%. AUC 0.92–0.93 (사실상 대등 — "MLP 가 검출도 최고" 문구 금지) |
| Fig 2(d) DQ 블록 히트맵 | `panels/panel_j_grid_error_DQ_{3,6,12,24}.csv` | 전체 평균 **13.1%** (중앙 13.3) |
| Fig 2(e) 농도 parity | ⚠ 아래 참고 | conc38 기준 중앙 배수오차 THI **1.33** · TBZ **1.36** · DQ **1.67**, 2배 이내 **89 / 87 / 65%** (≥6 µM) |
| Fig 3(a)(b) 조건별 막대 | `panels/panel_i_grid_true_vs_predicted.csv`<br>`64conditions/retrain_all.txt` | 조건별 held-out 예측이 갱신됨 — 쓰던 조건의 새 값을 위 파일에서 |
| Fig 3(c) 삼각형 + 회수율 표 | `panels/panel_a_triangle_nnls_high.png`<br>`panels/panel_a_nnls_conditions_high.csv`<br>`panel_g_triangle_accuracy.png` | 32 → **34점**, "순물질" 점이 DQ–THI 변 위로 이동(50/0/50). 1/K (33/57/222 µM) 는 검량이라 유지 |

## ⚠ Fig 2(e) 는 판단 보류 상태다

어제 판(중앙 1.32/1.35/1.46 · 2배 이내 85/95/77%)은 **미복원 hybrid 보정**(conc38 + K
등온선 보정, 다른 세션의 physics-rescue 작업 산출로 추정)에서 나왔다. 그 스크립트가
커밋되면 그 숫자를 되살릴 수 있고, 그 전까지 재현 가능한 값은 위의 **conc38** 이다.
MLP 농도 헤드는 출처가 아니며 µM 보고에 쓰지 않는다 (`FINDINGS` §2 2026-08-15 절).

## 캡션 문구 주의 (`FINDINGS_2026-08-12.md` 에서)

- 방법 비교: RF 100 trees·max_features=sqrt, **CNN 만 40 epoch**(불리한 비교) 명기
- 서사: "NNLS vs MLP" 가 아니라 **"NNLS 가 이미 잘 복원하고, physics-MLP 가 그 위에서
  NNLS 가 무너지는 자리(고농도 삼원·글씨 필름)를 더 배워 메운다"** (§5)
- 음성 대조를 쓰는 그림이 있으면: THI 는 2% 문턱에서 오탐 0%/검출 99% (45.2% 는 라벨
  오류였음), DQ 는 쓸 만한 문턱 없음
