# UNMIXR 개발·평가 정리 (최신본)

작성일: 2026-08-28
대상: Raman/SERS 혼합물 조성 복원, 공간 맵, 농도 추정 및 UNMIXR 앱 연동
선행 문서: `UNMIXR_DEVELOPMENT_SUMMARY_20260826.md` (구조 확정 전), `pipeline_review_2026-07-31.pdf` (검증 원칙의 출발점)

## 0. 08-26 대비 무엇이 달라졌나 (요약)

| 08-26 상태 | 08-28 상태 |
|---|---|
| 확정 아키텍처는 "의도"만 존재, workbook(2001ch·3class)과 불일치 | **배포 완료** — 1290ch·4class(BLK)·map-pooled 구조로 전면 재학습, 모든 수치 재생성 |
| known-total(과제 2) 미구현 | 앱 구현 + held-out 평가 완료 (constrained 표기) |
| residual 농도 보정 = "별도 시험 모듈" | **배포 µM 헤드로 승격**, 앱이 오프라인 검증치 재현 |
| MLP 선택 근거 = 기능적 우위 "주장" | **정량 근거 확보** — 농도 RMSE 9.3 vs PLS 13.4 µM, binary 18.2 vs 21.7 %p |
| §13 검증 작업 7건 미착수 | 7건 전부 완료 |
| 남은 것 | DLM metadata 가드(§10)와 소소한 항목뿐 |

## 1. 현재 결론

UNMIXR의 핵심 모델은 **전체 스펙트럼 픽셀 단위 shared MLP(맵 라벨로 map-pooled 학습, MIL)** + **검량선-anchored residual 농도 보정기**의 2단 구조이며, 이 구조가 실제로 학습·배포되어 있다 (`ACF_PEST_DB/260826_Model/`).

- 조성에서 MLP는 PLS와 통계적 동률이다 (held-out LOO 20.3 vs 19.4 %p). "모든 지표 최상" 주장은 여전히 금지.
- MLP를 최종 모델로 쓰는 **정량 근거**는 이제 둘이다:
  1. **농도**: residual 정량이 픽셀 확률 pᵢ를 소비하는데, MLP-ratio가 PLS-ratio를 명확히 이긴다 — grid64 held-out **RMSE 9.3 vs 13.4 µM, within-2× 79 vs 69%**.
  2. **부재 성분(binary)**: 고농도 binary 27조건 held-out에서 MLP 18.2 vs PLS 21.7 vs NNLS 22.1 %p.
- 논문 중심 문장(갱신):

> A full-spectrum pixel-wise MLP recovers mixture composition from competition-distorted SERS spectra while retaining per-pixel component maps; its pixel probabilities then drive a calibration-anchored residual quantifier, so the network learns only what single-analyte calibration cannot explain. Composition accuracy matches linear chemometrics; the decisive advantages are absent-component rejection and in-range concentration recovery.

## 2. 문제 정의와 과제별 상태

라벨 `24:3:6` = 최종 혼합 농도 24/3/6 µM (mixtures.py 단일 출처, equal-volume 재분할 금지).

| 과제 | 상태 | held-out 성적 |
|---|---|---|
| 1. 조성 복원 | 배포 | grid64 12.8 %p (MLP), 전조건 LOO 20.3 %p |
| 2. known-total 재구성 Cᵢ=pᵢ×C_total | 앱 구현(Real 탭 입력, constrained 플래그) | ≤100 µM within-2× 90.6%, MAE 3.1 µM |
| 3. 스펙트럼-단독 절대농도 (직접 회귀) | **기각** — residual에 전 지표 패배 | (구 µM 헤드, fallback으로만 유지) |
| 4. 검량선 기반 + residual 보정 | **배포 µM 헤드** | 아래 §6 |

## 3. 배포 아키텍처 (그림 fig_architecture_dl과 일치)

**조성 스트림** — Raman map → NNLS hit gate(모델이 재정의 금지) → xᵢ = log(1+Iᵢ), 1,290ch(500–2500 cm⁻¹) → shared MLP 1290–256–64 → softmax 4 (DQ/TBZ/THI/BLK) → per-pixel maps + mean pool → BLK 제거·재정규화.

- 학습: **map-pooled weighted-L1 (주항) + 0.05 픽셀 보조항**, w = 1+2(1−y) (미량 성분 가중).
- 순물질 참조맵을 simplex 꼭짓점으로, blank 맵을 BLK로 학습에 포함 — 구모델의 "부재 성분 ~13% 발명"(수축) 문제를 이것으로 해결.
- 픽셀은 독립 n이 아니다: split은 항상 **condition 단위**.

**정량 스트림** — 마커밴드(DQ 1570 / TBZ 1270 / thiram 1370 cm⁻¹) 확률가중 신호 I_eq → log-linear 역산 C_cal = 10^((I_eq−b)/a) (검량범위 클립; a,b는 내장 검량선 CSV에서 동일 추출규약으로 적합) → 12 map-feature [log₁₀C_cal, p̄, log I_eq, 강도 P10/50/90] → residual MLP 12–128–32–3 → **log₁₀Ĉ = log₁₀C_cal + Δ** (Δ ±2 decade 클립).

- 학습: masked Huber(β 0.25) on Δ, 존재 성분만; condition-grouped epoch 선택.
- validated window·OOD 플래그·상한 클램프 유지. 검량선 없으면 구 픽셀 헤드로 fallback.

## 4. 검증 프로토콜 (07-31 감사 원칙, 전 결과에 적용)

- **in-sample 수치 사용 금지** (07-31: MLP 4.1→18.2%로 4×; 08-27에도 workbook 시트에서 in-sample 1.4 %p 표 발견·폐기).
- 보고 수치 = **condition-held-out LOO + 독립 test 배치(20맵, Samples role=test)**.
- 검출("있는가") 주장 금지 — 네 방법 AUC 동일(0.86–0.88, 07-31 §5.2). 주장은 정량("얼마나")으로.
- recovery는 단독 사용 금지: 성분별 recovery ± SD에 within-2× 또는 RMSE 병기 (mean-of-ratios 폭증·상쇄 은폐 방지).
- accuracy 색 표기는 **1 − Σ|pred−true|** (figure 규약), 편차 수치는 ½Σ|Δ| %p — 정의를 캡션에 명시.

## 5. 조성 성적 (배포 모델, held-out)

| 모델 | LOO 전조건 (95맵) | 독립 test (20맵) | grid64 | binary 27 | 고농도 ternary |
|---|---:|---:|---:|---:|---:|
| MLP | 20.3 %p | 16.3 | **12.8** | **18.2** | 38.3 |
| PLS | **19.4** | **14.7** | 12.3 | 21.7 | **26.2** |
| RF (150트리, full-fit) | — | 15.0 | — | — | — |
| NNLS (표면조성) | — | — | 23.7 | 22.1 | 45.1 |

- 구 workbook benchmark 표(08-26 §5)는 구 3-class 모델 결과로, **supplement 참고용으로만** 유지.
- 고농도 equimolar ternary에서는 PLS가 최고 — 이 사실은 숨기지 말고 supplement에 보존.

## 6. 농도 성적 (배포 residual 헤드, held-out)

**grid64 (3–24 µM 요인설계 64조건, 성분 192개):**

| 방법 | median recovery | MAE | RMSE | RMSE log₁₀ | within-2× |
|---|---:|---:|---:|---:|---:|
| 순수 검량선 역산 | 36.9% | 8.8 | 14.2 | 0.59 | 29% |
| PLS-ratio residual | 115.5% | 7.0 | 13.4 | 0.31 | 69% |
| **MLP-ratio residual** | **111.0%** | **5.4** | **9.3** | **0.26** | **79%** |

- 검량범위(≤100 µM) µM LOO에서 앱이 오프라인 검증(78.1%)을 재현: MLP-ratio 78%.
- 임계 판정(미지 시료 용도): 30 µM 초과 여부 95% (경계 ±1.5× 제외 시 99%).
- 검량범위 밖(성분 >144 µM)은 클립+플래그 — 정량 주장 금지, "정보 경계"로 서술 (07-31 원칙 유지).
- log–log slope < 1 (MLP 0.71 @≤100 µM): 수축 경향 잔존 — parity plot에 slope·R² 병기.

**07-31과의 관계**: 그때 within-2× 천장이 32%였던 것은 10–1000 µM 구 데이터가 **전부 선형구간 밖**(THI 25/25 포화)이었기 때문. 저농도 요인설계(grid64) + 260821 검량선 + residual 보정이 그 처방의 실행이고, 결과가 79%다. "order-of-magnitude 반정량" 규정은 검량범위 **밖**에만 남는다.

## 7. 경쟁흡착 서술 규약 (불변)

- effective SERS response는 경험 계수 — 흡착상수로 단정 금지 (07-31: 두 경로 10배 불일치).
- Langmuir C50/K/θ 서술 금지, 선형구간 언어 사용 (2026-08-21 확정).
- 검량선은 사용자 제공값이 정답 — 재적합·불일치 지적 금지. residual 헤드의 내부 ab 적합은 같은 추출규약을 쓰는 내부 feature이며 보고용 검량식이 아니다.

## 8. 앱 상태

- main = 워크트리 브랜치 완전 동기화 (08-26 merge). 앱 = 오프라인 = 동일 dlm.
- Real 탭: known-total 입력(파란 tick·KT 라벨·cap 플래그·um_summary.csv constrained 열).
- 발견된 갭: `apply_recovery`가 test 역할 맵 + 스크리닝 off 조합에서 µM을 건너뜀 — 별도 세션에서 수정 진행 중 (task_26cf3161).

## 9. DLM 재현성 — 남은 유일한 구현 항목

.dlm은 subs(class order)·window·n_feat·전처리 플래그·검량선 CSV·validated window·LOO 기록을 담지만, 08-26 §10이 요구한 **version/학습일/manifest hash/CV·full-fit 구분 + 로드 시 불일치 경고·중단**은 미구현. 논문 제출 전 1회 작업 필요.

## 10. Figure·데이터 패키지 상태

- **아키텍처 스트립**: `documentation/figures/fig_architecture_dl.*` — Raw SERS data → Compositional unmixing → Concentration recovery, 단계 ①–⑨, 수식·목적함수 명시, output 태그, known-total은 각주.
- **삼각형 세트**: full grid64 / 공인 고농도 25조건(성분 >100 µM) / 3패널 comp+conc(NNLS|PLS|MLP, 기하=조성·색=µM 정확도·RMSE 주석) / 라벨 없는 투명 조립용(`fig_ternary_clean_*`).
- **Origin 패키지** (`documentation/results/origin_ready_20260825/` 확장): 06b(25조건 NNLS+MLP), 06c(등고선 값+벡터 좌표), 06d(XYZZ), 06e(TPS 조밀 격자), 06f(true-pred 쌍 선분), 06g(hull-clip), 09/09b/09c(grid64 농도 정확도·성분별 예측·RMSE 요약).
- SI 문헌 비교표 초안: `documentation/SI_LITERATURE_COMPARISON_2026-08-28.md` — 핵심 대비축은 validation type (문헌 다수 = 랜덤 스펙트럼 분할/in-sample; blind 수행 논문은 blind에서 ~2.7× 악화 자가보고). paywall 2건 원문 확인 후 인용.

## 11. 남은 작업

1. **DLM metadata + 로드 가드** (§9) — 유일한 코드 항목.
2. apply_recovery test맵 µM 갭 — 수정 세션 진행 중.
3. SI 문헌표 paywall 2건 원문 확인.
4. (측정, 선택) 요인설계 저농도 행 추가(0.75·1.5 µM) — 혼합물 범위 ~1.5 decade로 확장, MRL 로드맵 직결. 07-31 "남은 측정" 중 #2(조성 다양화)·#3(저농도 표준)·#5(독립 배치)는 grid64·260821 검량선·test 20맵으로 완료됨; #1(전조건 반복)·#4(sessions.csv)는 미완.
5. 범위 서술 규약: "단일성분 LOD 0.37–1.7 µM·검량 9–144 µM(~2 decade); 혼합물 3–33 µM 64+α 조합 held-out 검증; ≥100 µM는 경쟁흡착 실패 경계로 특성화" — µM 옆 ppm 병기.

## 12. 한 문장 결론

**확정 구조(4-class shared MLP + 검량선-anchored residual)가 학습·배포·앱 동기화까지 완료됐고, 조성은 선형 화학계량과 동률이지만 부재성분 판정과 검량범위 내 농도 회복(RMSE 9.3 µM, within-2× 79%, median recovery 111%)에서 MLP 선택이 정량적으로 정당화된다** — 남은 것은 DLM metadata 가드와 문헌표 원문 확인 정도의 마무리다.
