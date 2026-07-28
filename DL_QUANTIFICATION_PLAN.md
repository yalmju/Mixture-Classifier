# DL 정량화 계획 — 순수 스펙트럼 → 조성 + 농도 (로컬 작업용 핸드오프)

> 지금까지의 설계 논의 완결판. 로컬 세션에서 이 문서 하나로 이어간다.
> 대상: **DQ(diquat) / THI(thiram) / TBZ(thiabendazole)** SERS 혼합물.
> 리포: `mixture-classifier` (UNMIXR). 추가 코드: `unmix_net.py`.

---

## 0. 한 줄 요약

**DL을 "분류기"가 아니라 "예측기"로.** 순수물질만 학습해 임의 혼합을 분해하되,
피크를 판독하는 게 아니라 **패턴 전체로 조성·농도를 예측**한다. 비율은 데이터로
(gain 약분), 농도는 검량선으로(gain anchor), **DL은 그 사이 `spectrum→B` 적합을
instrument drift에 안 속게** 만든다. NNLS를 없애는 게 아니라 그 약점만 메꾼다.

---

## 1. 네 기존 파이프라인 + DL이 들어가는 자리 ★

현재 프로세스:
`샘플링 → RF 분류 → VIP 선별 → 검량선(series) 획득 → real 활용`

DL은 **마지막 "real 활용"의 `Y→B` 단계 하나에만** 들어간다. 나머지는 유지.

| 단계 | 지금 | DL 적용 후 | 바뀜? |
|---|---|---|---|
| 1. 샘플링 | Samples (전처리·class/batch/role) | 그대로 | — |
| 2. **RF 분류** | 어떤 성분 present (검출) | 그대로 (RF 유지) | — |
| 3. **VIP 선별** | 판별 밴드 (해석/검출 보조) | 그대로 (단, 검량엔 안 씀 ↓) | — |
| 4. **검량선 획득** | 희석 series → NNLS **B** → K, gA | 그대로 (NNLS B) | — |
| 4.5 | — | **★ ResNet 1회 학습** (P + K,gA로 합성) | ✅ 추가 |
| 5. **real 활용** | NNLS로 `Y→B` unmix | **★ NNLS ↔ ResNet 선택** | ✅ |

**두 축이 따로 흐른다** — 헷갈리지 말 것:
- **검출 축** (2 RF, 3 VIP): *어떤* 성분? → DL 안 건드림.
- **정량 축** (4 검량, 5 unmix): *얼마 / 몇 M*? → **5번의 `Y→B`가 DL 자리.**

### VIP vs 전체스펙트럼 B (3→4 함정, 이미 넘음)
"VIP 선별 → 검량"은 암묵적으로 **단일 marker 밴드 검량**을 뜻함. 근데:
- **VIP 단일밴드 검량**: DQ@1572가 THI/TBZ와 겹쳐 오염 → **DQ R²=0.69, LOD 693 nM**.
- **전체 스펙트럼 B 검량**: 겹침 분리 → **DQ R²=0.92, LOD 62 nM** (11× ↓).

→ **VIP는 검출/해석용으로만, 검량·정량은 전체 스펙트럼 B로** (이미 그렇게 함).
DL quantifier가 예측하는 것도 이 **B**라 4·5번 정량 축과 정확히 붙는다.

---

## 2. 물리 배경 (설계 근거)

경쟁흡착(Langmuir) 사슬:
```
농도 C_i ──경쟁──▶ 표면덮힘 θ_i = K_i·C_i / (1 + Σ_j K_j·C_j)   (포화, Σθ<1)
θ_i     ──신호──▶ Y = g · Σ (A_i·θ_i)·P_i  =  Σ B_i·P_i ,  B_i ≡ g·A_i·θ_i
```
`K` 친화도, `A` 밝기, `P` 순수지문, **`g` 기질 gain(칩·스팟마다 변동)**.
**Y는 B에 선형** → `fit_B(Y,P)`=NNLS로 B 복원.

핵심 두 문장:
1. **비율 쉬움** — `C_i:C_j = (B_i/R_i):(B_j/R_j)`, gain `g` 약분 → 기질 변동에 불변.
2. **절대 농도 어려움** — `B=g·A·θ`에서 `g` 못 떼면 절대값 불가 → **anchor(검량선/내부표준) 필수.**

### 농도 어려움 4분해 (DL 되는 것/안 되는 것)
| 어려움 | DL 극복? | 비고 |
|---|---|---|
| ① gain 미지 (절대 스케일) | ❌ | 정보이론적 벽. anchor로만 깸. DL은 그 위에 얹힘 |
| ② 비선형 포화 (Langmuir) | ✅ | 실응답이 이상 Langmuir와 다르면 DL이 실곡선 학습 (옵션 B) |
| ③ 경쟁 결합 (θ_i가 모든 C_j 의존) | ✅ | 결합 역함수 학습 (옵션 B) |
| ④ drift (shift/baseline) | ✅ | augmentation 불변성 (검증됨, 지금 구현) |

**①에 힘 쓰지 말 것.** ②③④가 DL 영역. 지금 구현은 ④, ②③은 옵션 B.

---

## 3. NNLS vs ResNet — 뭐가 바뀌고 뭐가 남나 + 학습

- **바뀌는 건 5번 `Y→B` 하나.** ①검량 적합·②θ→C 역산(Langmuir 공식)은 그대로.
- `ResNet1DQuantifier.predict_B`는 `fit_B`와 **인터페이스 동일 → drop-in**.
- **무조건 교체 아님, 조건부**: clean 스펙트럼엔 NNLS가 최적(못 이김), drift엔 ResNet 승.
  → 실전: 검량=NNLS(항상), 실측 맵=ResNet(또는 둘 다 돌려 재구성 R² 낮은 쪽 버림).

**ResNet은 학습 필요 (NNLS는 학습 없음).** 단 **실측 라벨 혼합이 아니라 합성으로 학습** —
순수 P + 검량 K·gA로 (Y, B) 페어 무한 생성. **네 RF가 augmented 순수로 학습하는 것과 동일 원리.**

| | 학습 | 추론(픽셀당) | 재학습 |
|---|---|---|---|
| NNLS | 없음 | 매번 solve (대량 맵 부담) | — |
| ResNet | 1회 (몇 분/CPU) | forward 1방 (amortized, 대량 맵엔 빠름) | 템플릿/검량 바뀔 때만 |

---

## 4. 지금까지 만든 것 — `unmix_net.py`

기존 `resnet1d.ResNet1D` 백본 재사용, head/loss/데이터만 뒤집음. 순수 합성만으로 학습. torch 필요.

### `ResNet1DUnmixer` — 조성(비율)
- 입력 **L2 정규화**(모양만) → gain 약분. head **softmax** → 비음수·합1 조성.
- 학습: Dirichlet 혼합 + 검출기 augmentation.
- `fit(pure_spectra)` / `predict_abundance(spectra) -> (n,K)`.

### `ResNet1DQuantifier` — 농도(B)
- 입력 **절대세기**(정규화 X → magnitude 보존). head **softplus** → `B=gA·θ`.
- 학습: **fitted 검량(K,gA)만으로** 합성 (true physics 불필요). magnitude 보존 corruption
  (shift/baseline/noise, **intensity jitter 없음** = gain 고정).
- `fit(pure_spectra)` / `predict_B` / `quantify(spectra) -> (C, θ, B)`.
- θ→C는 `calibration.concentration_from_coverage` 재사용 → **농도 가정 안 늘림**.

### 벤치마크 (`python unmix_net.py`, synthetic)
**조성 RMSE:** none NNLS **0.003**/DL 0.015 · shift2/0.05 NNLS 0.135/DL **0.013** · shift6/0.10 NNLS 0.220/DL **0.075**
**농도 logMAE:** none NNLS **0.139**/DL 0.258 · shift2/0.05 NNLS 1.122/DL **0.345** · shift5/0.10 NNLS 1.232/DL **0.882**
→ 이상 선형=NNLS 최적, drift 끼면 DL이 조성 10×·농도 자릿수 앞섬. "안 속아서" 이김.

---

## 5. 검량선 실측 결과 (제공됨) — DQ 구제 확인
농도 그리드 **1e-7~1e-3 M, 데케이드당 1·5 (9점)**. marker: DQ@1572, TBZ@1011, THI@1366.

| | 단일 밴드 | 전체 스펙트럼 B |
|---|---|---|
| DQ | R²=0.69, LOD 693 nM | **R²=0.92, LOD 62 nM** |
| TBZ | R²=0.98, LOD 519 nM | R²=0.95, LOD 459 nM |
| THI | R²=0.98, LOD 272 nM | R²=0.98, LOD 209 nM |

DQ 겹침 분리로 R² 0.69→0.92. 이 **B**가 DL quantifier의 예측 대상 → 설계 정합성 확인.

---

## 6. 다음 단계 (로컬 TODO)

### 6.1 실측으로 진짜 숫자 (최우선)
필요 데이터 (편한 것):
1. **각 (성분×농도) 스펙트럼 맵 CSV** (예: `DQ_1e-7.csv … DQ_1e-3.csv`) — 최선.
2. 아니면 **(성분, 농도, B) 표 + 순수 템플릿 P**.

절차:
1. 로드 (`load_calibration_csv`는 `compound,conc,spectrum…` 평탄테이블 기대 →
   맵이면 **맵→평균 스펙트럼 로더** 추가).
2. `calibrate(dilutions, P, names)` → K, gA (B 검량 재현).
3. `ResNet1DQuantifier(names, K=calib.K, gA=calib.gA).fit(P)` 학습 (= 4.5단계).
4. **실측 held-out 혼합**에서 `NNLS B` vs `DL B` → θ→C 비교, drift 유무로 나눠 리포트.
   → 네 실제 drift 수준에서 DL이 진짜 이득인지 데이터가 판정.

### 6.2 (선택) 옵션 B — DL이 비선형 정복
- ②③(포화·경쟁)까지: DL이 **spectrum → C 직접** 학습(검량선은 gain anchor만).
- 이득 조건: 실응답이 이상 Langmuir에서 벗어날 때만 공식 역산을 넘음.

### 6.3 GUI 통합 ✅ (완료)
- **Real-data 탭**: method 콤보에 `ResNet1D (DL)` 추가 → NNLS/MCR와 나란히 조성 백엔드 선택
  (`unmix.unmix_map(method='dl')` → `unmix_net.unmixer_from_templates`).
- **Real-data + Quantify 탭**: `DL spectrum→B` 토글 → 단일 `Y→B` 스텝을
  `ResNet1DQuantifier.predict_B_one`로 라우팅 (`calibration.quantify(..., B_predictor=…)`,
  `unmix_net.quantifier_from_calibration`). **검량 적합·θ→M 역산은 항상 NNLS 유지.**
- 두 DL 경로 모두 기존 워커 스레드(`RealWorker`/`QuantWorker`)에서 학습 → UI 안 멈춤.
- torch 미설치 시 `pip install torch` 안내 메시지로 graceful fail.

### 6.4 데이터 캠페인 (일반화·정직 검증)
- **replicate 맵** (조건당 ≥3, 다른 날/스팟/재조제) — 없으면 hold-out이 "안 본 비율"만
  테스트(안 본 시료 아님). DL은 픽셀 near-duplicate 외우기 강함 → leakage 위험.
- **대칭 triple** — THI만 내린 1:1:1/1:1:0.1/1:1:0.01은 편향. 0.1:1:1, 1:0.1:1 추가.
- **matrix blank** (농약 없는 banana/leaf) — 기질 오검 방지.
- **총 농도 vs 조성비 분리** — 각 맵 실제 mol 메타데이터 기록.

---

## 7. 설계 결정 기록
- **분류기→예측기**: `resnet1d`는 one-hot BCE 검출기였음. 조성/농도는 NNLS만 함 → DL을 회귀기로.
- **농도 3안 중**: (A) end-to-end µM ❌ gain 굳음 · **(B) DL→B, calibration→C ✅ 채택** ·
  (C) hybrid → magnitude 결국 NNLS.
- **비율기=L2정규화(모양), 농도기=절대세기(magnitude)** — 정규화가 농도 정보 지움.

## 8. 한계 (정직하게)
- **gain 고정 가정** — 검량/측정 gain 동일 전제. 심한 drift엔 내부표준 필요.
- **signal ≠ molar** — θ는 signal 기준, 진짜 molar는 response factor 보정 별도.
- **현재 검증 synthetic** — 진짜 숫자는 실측 필요.
- **DQ** — B로 R²0.92까지 왔으나 여전히 가장 어려운 성분(약신호·겹침).

## 9. 인터페이스 + 파일
```python
from unmix_net import ResNet1DUnmixer, ResNet1DQuantifier
u = ResNet1DUnmixer(names).fit(P);            comp = u.predict_abundance(X)   # (n,K) 합1
q = ResNet1DQuantifier(names, K=calib.K, gA=calib.gA).fit(P)
C, theta, B = q.quantify(X)                    # C: 절대 M
# python unmix_net.py → 조성+농도 NNLS-vs-DL 표 (synthetic)
```
전처리는 `sers_mixture.preprocess`(ALS+L2). `ResNet1D`는 32× 다운샘플+GAP → 파수 길이 무관.

- `unmix_net.py` (이 작업) · `resnet1d.py`(백본+검출기) · `competitive.py`(coverages/fit_B) ·
  `calibration.py`(calibrate/concentration_from_coverage/build_synthetic_lab) ·
  `sers_mixture.py`(preprocess/augment/AugmentConfig) · `io_utils.py`(load_calibration_csv) ·
  `dataset.py`(맵 폴더→dataset).
