# 검량선-residual µM 헤드 채택 (2026-08-26)

## 왜

통합 재평가(`documentation/results/integrated_final_20260824/full`, 생성 스크립트
`documentation/scripts/run_integrated_reevaluation.py`)에서 세 가지 농도 경로를
**같은 absolute-condition-grouped 5-fold**로 비교했다. fold마다 조성 MLP와 보정기를
학습셋만으로 재학습한 진짜 held-out 평가다 (경계 검증 완료 2026-08-26).

검량선 범위 내 64개 맵 (3–24 µM Grid64):

| 경로 | median recovery | MAE (µM) | within 2× |
|---|---:|---:|---:|
| 순수 검량선 역산 | 36.9% | 8.81 | 29.2% |
| Ridge Δlog10 보정 | 122.7% | 6.19 | 72.9% |
| **Neural Δlog10 보정** | **104.2%** | **5.01** | **78.1%** |

전 89맵(고농도 binary 포함)에서도 순서는 같지만 검량선 밖 binary는 여전히 나쁘다
(neural MAE 154 µM, within-2× 43%). **이 헤드의 신뢰 구간은 검량선 범위 내부다.**

## 무엇이 바뀌었나

`dl_model.train_model`: 검량선 CSV가 있으면 µM 헤드로
`calibration_residual_uM_v1`을 학습하고, 없으면 기존
`map_pooled_pixel_concentration_v1`(직접 log10 µM 픽셀 헤드)로 fallback.

구성 (검증된 run과 동일):

```
성분별 마커 밴드 (DQ 1570 / TBZ 1270 / THI 1367 cm⁻¹, 그 외 물질은 템플릿 최대 채널)
→ 조성확률 가중 밴드신호 Ieq (±10 cm⁻¹ 최댓값)
→ log-linear 역산 Ccal = 10^((Ieq−b)/a), 검량 범위로 클립
→ 맵 단위 12 feature [log10 Ccal ×3, ratio ×3, log1p Ieq ×3, 강도 P10/50/90]
→ 128→32 net → Δlog10(C) (±2 decade 클립) → C = Ccal·10^Δ
```

- **a, b**: 내장 검량선 CSV의 희석계열에서 같은 밴드 추출 방식으로 적합
  (`ab_source`에 기록). 적합과 역산이 같은 추출 규약을 쓰므로 규약 차이로 생기는
  성분별 상수 오프셋은 Δlog10이 흡수한다.
- **epoch 선택**: condition-grouped 20% holdout, 그 뒤 전체 재적합 (기존 관례 동일).
- **validated window**: 20% condition holdout에서 2-fold 이내로 회복된 레벨 구간
  (`validated_ranges_M`), 기존 의미 그대로.
- **LOO**: leave-one-absolute-condition-out으로 residual net만 재적합.
  조성 ratio는 배포 조성 헤드 것을 쓴다 — dict의 `ratio_source`에 명기.
  완전 중첩(fold별 조성 재학습) 수치는 오프라인 재평가 스크립트가 정답.
- **적용** (`apply_uM_pixels` → `_apply_calibration_residual`): 배치를 한 맵으로
  취급, analyte 확률 ≥0.5 픽셀만 맵 컨텍스트에 사용(배경 혼입 방지). 맵 추정치가
  보고값이고, 픽셀 맵은 픽셀별 역산을 성분별로 재스케일해 중앙값 = 맵 추정치가
  되게 함 — 하류의 median pooling이 그대로 검증된 값을 보고한다.

## 검증

- 합성 경쟁억제 데이터: 순수 역산 median 0.78 decade 오차 → residual LOO 0.070.
- 실데이터 스모크(12맵, 260821 검량선): DQ3-TB3-TH3 → DQ 3.07 / TBZ 3.23 / THI 1.45 µM.
- 검량선 없는 학습은 기존 픽셀 헤드로 정상 fallback.

## Known-total 제약 평가 (2026-08-26 추가)

같은 held-out 테이블(89맵, fold별 재학습)에서 총농도(라벨 합)를 선언된 메타데이터로
쓰는 두 변형을 총농도 구간별로 채점 (성분 단위, present만):

| 구간 (총농도) | 방법 | med rec% | within 1.5× | within 2× | MAE (µM) |
|---|---|---:|---:|---:|---:|
| ≤30 µM (29맵) | residual (spectrum-only) | 112.2 | 56.3% | 80.5% | 3.32 |
| | residual + total cap | 112.2 | 57.5% | 81.6% | 3.03 |
| | **known-total (pᵢ×C_total)** | **103.7** | **75.9%** | **92.0%** | **1.83** |
| ≤100 µM (64맵) | residual | 104.2 | 55.2% | 78.1% | 5.01 |
| | residual + cap | 104.2 | 56.2% | 79.2% | 4.81 |
| | **known-total** | **104.5** | **70.3%** | **90.6%** | **3.12** |
| ≤500 µM (76맵) | residual | 99.8 | 51.4% | 74.1% | 16.03 |
| | residual + cap | 99.8 | 53.7% | 75.9% | 12.48 |
| | **known-total** | **104.0** | **66.2%** | **87.5%** | **9.88** |
| 전체 89맵 | residual | 91.9 | 50.0% | 70.3% | 38.02 |
| | **known-total** | **103.6** | **63.4%** | **84.6%** | **26.21** |

- **cap 단독은 거의 무력**: ≤500 µM에서 216개 성분 예측 중 8개만 바뀜(총량 초과
  예측만 깎이므로). MAE 개선은 그 소수 폭주가 커서 생긴 것.
- **known-total 재구성이 실질 개선**: 조성 회복이 정확하므로 pᵢ×C_total이 모든
  구간에서 최선. 특히 <0.5× 저평가가 거의 사라짐(19%→7%).
- 이 수치는 **제약 적용(정보 추가) 결과**로 표기해야 하며 spectrum-only 벤치마크와
  같은 열에 섞으면 안 된다. 실제 미지 시료(총농도 미상)의 대표 수치는 여전히
  residual spectrum-only다.
- 앱: Real 탭 concentration options에 "known total µM" 입력 추가 — 설정 시 파란
  tick(known-total), 라벨 KT 값, cap 표시, `um_summary.csv`에
  known_total_uM/median_capped_uM/constraint_flag 열이 붙는다.

## 주의

- md(UNMIXR_DEVELOPMENT_SUMMARY_20260826)의 §7 수치는 이 프로토콜의 것이 맞다
  (재현 확인). 단 `S:/…/ACF_PEST_DB/260824/260824_figure3d_concentration_predictions.csv`는
  다른(후속) 변형 실행의 산출물로 수치가 다르다 — 혼용 금지.
- 검량선 범위 밖(고농도 binary 등)은 여전히 부정확: 범위 클립 + validated window로
  표시되지만, 보고는 범위 내로 한정할 것.
