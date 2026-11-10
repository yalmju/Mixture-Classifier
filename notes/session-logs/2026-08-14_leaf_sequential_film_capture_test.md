# 잎 잔류 농약 순차 박막 포집 실험 메모

날짜: 2026-08-14

## 목적

금-셀룰로오스 나노섬유 잉크가 잎 위에서 건조 박막을 형성하면서 표면의 잔류 농약을 박막으로 포집·전이하는지 확인한다.

박막 제거 후 잎에서 농약 신호가 사라지는 결과만으로는 실제 농약 제거와 표면증강 라만 산란 잉크 제거에 따른 검출능 상실을 구분할 수 없다. 따라서 동일 영역에 새로운 blank sensing film을 순차적으로 적용하여 잔류 농약을 다시 조사한다.

## 핵심 실험 설계

1. 잎의 지정 영역에 농약을 알려진 양으로 처리한다.
2. 농약을 완전히 건조한다.
3. 농약이 포함되지 않은 blank gold-cellulose nanofiber sensing ink를 도포한다.
4. Film A를 완전히 건조한 뒤 동일 조건으로 Raman mapping한다.
5. Film A를 잎에서 분리한다.
6. 분리한 Film A를 깨끗한 유리 기판에 올리고 다시 Raman mapping한다.
7. Film A를 제거한 잎의 동일 영역에 fresh blank Film B를 도포한다.
8. Film B를 건조한 뒤 동일 조건으로 Raman mapping한다.
9. 가능하면 Film B도 분리한 후 같은 위치에 fresh Film C를 적용하여 순차적인 신호 감소를 측정한다.
10. 마지막으로 동일 영역에 농약을 다시 일정량 처리하고 fresh film을 적용하여 re-spike 신호 회복을 확인한다.

## 기대되는 핵심 결과

포집·전이가 실제로 발생했다면 다음 패턴이 예상된다.

- 잎 위 Film A: 농약 신호 양성
- 분리하여 유리 위에 놓은 Film A: 동일한 농약 신호 양성
- 동일 잎 영역에 새로 적용한 Film B: Film A보다 농약 신호 감소
- Film C: 추가 감소 또는 배경 수준
- 농약 재첨가 후 fresh film: 농약 신호 재출현

이 조합은 Film B의 낮은 신호가 단순히 잉크 제거 또는 잎 손상으로 측정이 불가능해진 결과가 아니라, 첫 번째 박막으로 농약이 이동한 결과임을 지지한다.

## 필수 대조군

1. 무처리 잎 + blank gold-cellulose nanofiber film
2. 농약 처리 잎 + 아무 처리 없음
3. 농약 처리 잎 + 물 도포·건조
4. 농약 처리 잎 + cellulose nanofiber-only film
5. 농약 처리 잎 + gold-cellulose nanofiber film
6. 농약을 처음부터 잉크에 혼합한 positive control
7. Film A를 제거하지 않고 동일 시간 동안 유지한 대조 영역
8. Film A 제거 후 농약을 다시 첨가한 re-spike control

## 측정 조건

Film A, Film B, Film C와 분리막의 비교에서는 다음 조건을 고정한다.

- laser power
- integration time
- objective
- accumulation 횟수
- Raman mapping step
- spectral preprocessing
- marker-band integration window
- map color scale

정확히 같은 잎 영역을 다시 찾기 위해 잎맥, 잎 가장자리, 미세한 표시 또는 현미경 좌표를 landmark로 기록한다. 각 단계에서 optical image와 mapping 영역을 함께 저장한다.

## 분석 지표

Marker band의 baseline-corrected integrated area를 사용한다. 가능하면 잉크 내부의 안정적인 Raman reference band로 정규화한다.

### Sequential depletion index

\[
R_n = \frac{I_{pesticide,n}}{I_{reference,n}}
\]

\[
Sequential\ depletion = 1-\frac{R_B}{R_A}
\]

이 값은 질량 기준 회수율이 아니므로 다음 표현을 사용한다.

- Raman-based depletion
- sequential depletion index
- relative film-transfer signal
- apparent capture signal

다음 표현은 독립적인 질량분석 검증 전에는 사용하지 않는다.

- absolute recovery
- mass balance
- extraction efficiency
- quantitative removal

## 주요 혼동 요인

### 잉크 제거에 따른 검출능 소실

해결: 동일 위치에 fresh Film B를 다시 적용한다.

### 박막 분리 과정에서 잎 표면 손상

해결: re-spike 후 fresh film에서 농약 신호가 다시 나타나는지 확인한다. Blank leaf에도 동일한 film 부착·분리 과정을 적용하여 잎 background와 wax-layer 변화도 기록한다.

### 물에 의한 단순 재용해 또는 세척

해결: 동일 부피의 물 처리 대조군을 포함한다.

### 셀룰로오스 포집과 금 표면 흡착의 구분

해결: cellulose nanofiber-only film과 gold-cellulose nanofiber film을 비교한다.

### 표면증강 hotspot 변동

해결: 내부 reference band 정규화, 독립 film 반복, 동일 측정 조건과 고정 color scale을 사용한다.

## 반복 설계

최소 권장 반복:

- 농약 3종
- 농약별 독립 잎 3장 이상
- 잎당 최소 2개 독립 처리 영역
- Film A와 Film B는 모든 시료에서 수행
- Film C와 re-spike는 대표 조건에서 수행 가능

픽셀은 하위 표본이며 독립 반복으로 계산하지 않는다. 통계 단위는 독립 잎 또는 독립 처리 영역으로 설정한다.

## Figure 6 권장 패널

- (a) 농약 처리 → Film A → peel → Film B → re-spike 실험 개념도
- (b) Film A가 붙은 상태의 optical image와 농약 map
- (c) 분리하여 유리 위에 놓은 Film A의 map
- (d) 동일 잎 영역에 적용한 Film B의 map
- (e) re-spike 후 fresh film에서 회복된 농약 map
- (f) Film A, Film B, Film C의 정규화 marker-band intensity 또는 sequential depletion index

## 논문용 안전한 결론

강한 결과가 확보되면 다음 수준으로 기술한다.

> The pesticide signal was retained in the detached sensing film but markedly decreased when a fresh film was reapplied to the original leaf region. Recovery of the signal after re-spiking further indicated that the decrease was attributable to residue depletion rather than loss of local detectability, supporting transfer of surface-deposited pesticide into the self-formed plasmonic film.

액체크로마토그래피 기반 질량수지가 없으므로 정량적 추출효율이 아니라 박막으로의 포집·전이와 Raman 기반 depletion으로 제한한다.

## 다음 단계

1. 박막이 잎에서 손상 없이 분리되는지 blank ink로 예비시험한다.
2. Film A와 Film B의 동일 조건 측정 재현성을 확인한다.
3. 가장 신호가 강한 thiram으로 순차 박막 실험을 먼저 수행한다.
4. 실험이 성립하면 thiabendazole과 diquat으로 확장한다.
5. Figure 6 최종 패널과 통계 단위를 결정한다.
