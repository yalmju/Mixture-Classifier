# 2026-09-01 세션 — Real 탭 전면 정비 + µM 판독 3모드

내일 앱을 켜면 이 문서 하나로 재개할 수 있게 정리한다.

## 내일 아침 체크리스트 (5분)

1. **앱 실행** → Real 탭으로 바로 시작, 좌측에 `DL: mlp_composition_260831_final.dlm · BLK class ✓ · µM head ✓` 자동 표시.
   폴더 기본값 = `ACF_PEST_DB\Pure` (Samples 안 거쳐도 참조 로드됨).
2. **정상 동작 확인** — `Ratio\260814_mixture_final\DQ12-TB12-TH12.csv` 로드 → Unmix:
   - true µM / declared total 자동 채움 (파일명 파싱: 12,12,12 / 36)
   - model head 모드 raw: **DQ 13.2 · TBZ 11.9 · THI 10.9** (참 12/12/12)
3. **글씨맵** — `Pest\260812_12 trio_THI_TBZ_DQ.csv` 로드 → Unmix:
   - 조성 38 : 18 : 44 (NNLS 93% THI에서 복원)
   - model head: **무응답** + 하단 빨간 domain-guard 경고 (측정 영역 밖, d≈3.9)
   - µM readout을 **pixel k-NN**으로 바꾸면: 12.0 / 3.6 / 6.0 µM-등가 (표면 로딩)
   - declared total 36 입력: reported **13.4 / 6.8 / 15.8 µM**
4. 레일 폭은 드래그로 조절(스플리터). auto background gate 기본 OFF → fraction 0.15 스핀 조작 가능.

## µM 판독 3모드 (concentration options의 "µM readout" 콤보)

| 모드 | 값의 출처 | 성적(검증창 3–24) | 쓰임 |
|---|---|---|---|
| model head (기본) | 검량+보정넷 log₁₀Ĉ=log₁₀C_cal+Δ | **RMSE 7.5 µM · within-2× 78%** | 논문 보고 경로 |
| map k-NN | 최근접 학습맵 3장 실측 농도 가중평균 | LOO 21.8 µM · 75% | 감사 가능(이웃 이름 표시) |
| pixel k-NN | 픽셀=액적: 픽셀 서명(밴드3+총강도+MLP조성3) → 라이브러리 2,311px 조회 | 조건별 예시: 24/6/3→24.0/5.4/4.1 (전체 LOO 미채점) | 글씨맵류 로딩-등가 판독 |

**공통 무응답 규칙**: 라이브러리 z-거리 > 3 → 농도 숫자 없이 "outside library" (외삽 구조적 금지).
declared total은 어느 모드에서든 오버레이(reported 줄) — supervised임을 라벨에 명시.

## 오늘 앱/모델 변경 전부 (main 머지 완료)

- 앱 시작 = Real 탭, FINAL dlm 자동 로드 (`UNMIXR_DLM` 환경변수로 교체 가능)
- 기본 데이터 폴더 = `ACF_PEST_DB\Pure` (빈 legacy 폴더 버그 해결 — "no reference classes"의 원인)
- Real µM 맥락을 게이트(0.15) 통과 픽셀로 구성 (학습 모집단과 정합; 글씨맵 raw 126→81)
- 번들 validated_ranges = 3–24 µM 통일 (DQ150/TBZ500은 근거 없는 라벨 범위; 백업 .bak-260901)
- 배치/OOD 감지기: residual 피처 z, max|z|>3 → 경고+무응답 (held-out 험지 ≤2.6 / 글씨맵 3.9)
- 판독 = 픽셀별 [MLP 조성 × 픽셀 µM총량] (조성-순서 정합), 분포 산점(violin+dots), 라벨 최소화("reported/ signal ⚠"), 총량 두 개(신호추정 vs 선언)를 제목에 병기
- 배치 앵커(기지 맵 1장 원포인트, 포화 성분 자동 제외), true/total 파일명 자동 채움
- 사이드카: `.knn.json`(맵 102), `.pxknn.npz`(픽셀 2,136 + 검량 시리즈 175 — 재학습 없이 append로 확장)

## 글씨맵(12/12/12 잉크) 최종 판독 카드

조성 38:18:44 복원(경쟁흡착 시연) · 신호 단독 절대정량은 무응답(로딩 1/6~1/20, 광학 아님 —
조용구간 배경 동일로 증명) · pixel k-NN = 로딩-등가 12/3.6/6 · 제조총량 36 → 13.4/6.8/15.8
(TBZ만 잉크 억압으로 낮음). 논문 자리: 경쟁흡착 복원 + 표면효과 + domain guard 시연.

## 미결 (다음 세션)

1. **논문 기본 보고 경로 확정** — 현 권고: model head(7.5/78) + domain guard 서술.
2. pixel k-NN 89맵 전체 LOO 정식 채점 (스크립트 `run_knn_um_readout.py` 확장이면 됨).
3. page_validate/held-out 캐시 경로엔 3모드 미반영 (Real 전용) — 필요 시 확장.
4. 측정 캠페인 시트(표적 12맵 + 세션당 표준 1장 행) 요청 시 작성.
