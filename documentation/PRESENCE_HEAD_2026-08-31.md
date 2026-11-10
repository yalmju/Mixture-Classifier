# Presence 헤드 — 부재 성분을 0으로 말하기 (2026-08-31)

## 왜

회귀 출력은 0을 찍지 못한다. 조성 헤드는 softmax 라 구조적으로 양수만 내고
(부재 성분 예측 중앙값 3.4%, µM 헤드는 0.9–6.6 µM), 앱 apply 경로에는 이를
0으로 바꿔주는 보고 게이트가 없었다 (`dl_model.apply_recovery` — NNLS screen 은
픽셀 선별에만 쓰임).

고정 문턱 게이트는 원리적으로 실패한다. NNLS 분율 5% 게이트는 고농도 binary
(부재 겉보기 최대 4.4%)에서는 부재 오보 0/27 이지만, 저농도 8:1 혼합물에서는
실존 소수 성분(TBZ 3 µM under THI 24 → NNLS 1.2%)을 ND 로 지운다 — 부재의
겉보기 신호가 파트너 맥락의 함수라 "몇 % 밑이면 0" 이 성립하지 않는다.

검량계열이 이 문제를 못 푸는 이유: 검량선은 p(신호 | C, 파트너 없음)만 주고,
0 판정에 필요한 것은 p(신호 | C=0, 파트너 있음)이다. 후자는 혼합물에서만
측정된다. 단, **검량 단일성분 맵 자체가 "파트너 1개 있는 0"의 실측**이라는
점이 해법의 열쇠였다 (THI 단독 24 µM 맵 = TBZ·DQ 의 저농도 부재 표본).

## 부재 바닥값 곡선 (14_absence_floor_curves.csv)

260821 검량 단일맵 35장을 전체 파이프라인(배경 BLK/INK 템플릿 + hit 게이트)으로
재측정한 부재 성분 겉보기 NNLS 분율:

| 부재 | 파트너 | 9 µM | 18 | 36 | 144 |
|---|---|---|---|---|---|
| THI | DQ | 7.4% | 6.4% | 2.1% | 0.0% |
| TBZ | DQ | 10.8% | 5.9% | 1.1% | 0.1% |
| TBZ | THI | 0.3% | 0.5% | 0.7% | 0.0% |
| DQ | any | ≤0.5% | | | |

- 오염 경로는 사실상 **약한 DQ 파트너** 하나 (잉크 템플릿이 DQ 와 cosine 0.73).
- ⚠ 배경 템플릿 없이 3-템플릿 NNLS 로 계산하면 바닥이 26–36%로 과대된다 —
  바닥값 계산은 반드시 배경 포함 파이프라인으로.

## 무엇: 성분별 presence 분류기 + 3-상태 보고

존재/부재를 회귀에서 분리해 성분별 로지스틱 분류기로 판정한다.

- **Feature (누출 없음 — 학습형 조성모델 출력 배제)**: 배경-인지 NNLS 자기
  분율, 최대 파트너 분율, 자기 마커밴드 세기(log1p), 최대 파트너 밴드 세기,
  총 신호(log1p), 분석물/배경 비. 마커 밴드 = DQ 1570 / TBZ 1270 / THI 1367 cm⁻¹.
- **학습 데이터 (전부 기존 측정)**: 89맵 번들(grid64 + binary) + 검량 단일맵
  35장 = 124맵 × 3성분. 음성 91 (고농도 binary 21·검량 부재 70), 양성 281.
- **보고**: p > 0.8 **Detected** / 0.2–0.8 **Indeterminate** / p < 0.2 **ND**.
  Indeterminate 가 핵심 — "이 신호 수준에서는 0과 미량을 구분할 수 없다"를
  말할 수 있는 상태.

## 검증 (조건-grouped 5-fold, seed 701)

| 성분 | AUC | 부재→Detected | 존재→ND |
|---|---|---|---|
| THI | 0.997 | 0/27 | 0/97 |
| TBZ | 0.940 | 2/32 | 5/92 |
| DQ | 0.910 | 4/32 | 5/92 |

- **저농도 파트너 옆의 0** (검량 부재 70건): Detected 오보 **0건**, ND 62, Ind 8.
- **8:1 소수 성분** (3 µM under 24 µM, 48건): 거짓 ND **1건** (고정 게이트는
  이들을 대량 ND 처리). Detected 22 / Indeterminate 25.
- 남은 오보: 부재→Detected 6건 전부 **고농도 binary** (150–500 µM 파트너
  crosstalk — 검량범위 밖 플래그 영역과 겹침). DQ 채널이 최약(잉크 유사성).

## 논문 문구 (초안)

> Because the composition regressor is a simplex-constrained model, it cannot
> emit exact zeros; presence is therefore adjudicated by a dedicated per-analyte
> logistic gate operating on background-aware NNLS abundances and marker-band
> intensities. Under condition-grouped cross-validation the gate attains
> AUC 0.91–0.997 per analyte, never promotes an absent analyte to "detected"
> in low-concentration single-analyte controls (0/70), and preserves genuinely
> present minor components at 8:1 imbalance (1/48 false negatives), reporting
> borderline signals as "indeterminate" rather than forcing a binary call.

## 산출물

- 프로토타입: `documentation/scripts/run_presence_head_prototype.py`
- 예측 전표: `documentation/results/presence_head_20260831/presence_predictions.csv`
- 바닥값 곡선: `documentation/results/14_absence_floor_curves.csv`
- 배포 통합: `train_presence_head.py` 가 사이드카
  (`<dlm>.presence.json`)를 만들고, `dl_model.load_model` 이 있으면 싣고
  `apply_recovery` 결과에 `presence` (성분별 prob·state)를 붙인다.

## 남은 일

1. 고농도 binary 오보 6건 — 검량범위 플래그와 연동해 "out-of-range" 상태로
   흡수하는 안을 검토.
2. 저농도 binary 8–12맵 측정 — 이제 설계 필수가 아니라 **검증용** (게이트
   커버리지의 마지막 외삽 칸을 실측으로 닫는다).
3. UI 노출 (Real/Recovery 탭에 3-상태 뱃지).
