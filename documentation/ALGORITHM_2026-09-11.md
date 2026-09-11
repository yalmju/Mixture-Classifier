# UNMIXR 알고리즘 상세 — DQ / TBZ / THI 삼성분 SERS 언믹싱과 농도 판독

작성 2026-09-11. 모든 수치는 배포 번들 `ACF_PEST_DB/260831_Model_FINAL/mlp_composition_260831_final.dlm`(+ 사이드카)과 코드(`dl_model.py`, `dl_quantify.py`, `unmix.py`, `dataset.py`, `page_real.py`)에서 직접 읽은 값이다. 추정치는 없고, 추정이 필요한 곳은 그렇게 표시했다.

---

## 0. 한눈에 보는 흐름

```
SERS map (픽셀 스펙트럼)
  → 전처리 (500–2500 cm⁻¹ 절단, 1,290 점, log1p)
  → NNLS 게이트 (순물질 + 배경 템플릿에 대한 비음수 최소제곱; 분석물 몫 ≥ 0.15인 픽셀만 통과)
  → 조성 MLP f(x): 1,290 → 256 → 64 → 4 softmax (DQ · TBZ · THI · BLK)
  → 맵 조성: BLK 제거 후 세 성분 재정규화, 신호가중 평균
  → 존재 판정(presence head): 성분별 로지스틱, P < 0.2 → 부재(0), P > 0.8 → 검출, 사이는 미결정
  → 농도 g(map): 12 맵 피처 → 128 → 32 → 3 (Δlog10) ; C = Ccal · 10^Δ
  → 적용영역 검사: 학습맵 라이브러리 최근접 z-거리 ≤ 3 이면 보고, 아니면 무응답
  (선택) 선언 총량이 있으면 Ĉᵢ = pᵢ · C_total
  (배포 대체) 픽셀 라이브러리 k-NN: 검사 실패 시 픽셀 서명 7개로 라이브러리 픽셀 15개 조회
```

두 개의 학습 모델(f, g)과 두 개의 비학습 참조 단계(NNLS 게이트, 라이브러리 조회), 그리고 한 개의 작은 로지스틱(presence)로 구성된다.

---

## 1. 입력과 전처리

### 1.1 맵 파일 형식
- CSV 3행 헤더: `X num,N` / `Y num,1` / `X,Y,wn1,wn2,…` 뒤에 픽셀 행 `x,y,I(wn1),I(wn2),…`.
- 원 파장수 축은 100–2600 cm⁻¹ 부근 2,001점(장비 축). 좌표(x, y)는 µm 단위 격자.

### 1.2 절단·정규화
- **절단 창**: 500–2500 cm⁻¹ (`trim=(500, 2500)`, 번들 `lo/hi`). 창 안의 축 점수가 **1,290** = 모델 입력 차원 `n_feat`.
- **베이스라인 제거**: 이 번들은 `baseline=False` — 장비에서 이미 보정된 `*_corrected.csv`를 쓰므로 앱은 ALS를 다시 걸지 않는다. (레퍼런스 폴더 설정과 무관하게 모델의 `baseline` 값이 우선한다.)
- **조성 입력 피처**: `feature_mode = "log1p_raw"` — 픽셀 스펙트럼을 0 이하 절단 후 `log1p`. 행 단위 L2 정규화는 하지 않는다(픽셀 간 세기 차이를 보존).

### 1.3 순물질 참조(Pure)
- `ACF_PEST_DB/Pure/`의 각 물질 CSV(`DQ-*.csv`, `TBZ-*.csv`, `THI-*.csv`)와 배경류(`BLK-*.csv`, `INK-*.csv`, `LEAF-*.csv`).
- 배경류 이름은 `dataset.BLANK_ALIASES = {"blk","blank","background","bg","none","ink","leaf"}` 로 판정된다. 배경류는 조성 클래스에 들어가지 않고(`subs = 이름 중 is_blank가 아닌 것`), NNLS 게이트의 "분석물이 아닌 것" 템플릿으로만 쓰인다.
- 템플릿 P = 각 물질의 평균 스펙트럼을 절단 창에서 L2 정규화한 것.

---

## 2. NNLS 게이트 — "이 픽셀에 분석물이 있는가"

목적: 배경(맨 기판, 잉크, 잎)만 있는 픽셀을 **학습 모델보다 먼저** 걸러낸다. 모델은 이 마스크를 바꾸지 못한다.

1. 각 픽셀 스펙트럼 x를 템플릿 행렬 [P_DQ, P_TBZ, P_THI, P_BLK, P_INK, P_LEAF …]에 대해 **비음수 최소제곱(NNLS)** 으로 분해 → 계수 A(픽셀 × 템플릿). 앱에서 이 A가 `A_evidence`(스펙트럼 증거)로 보관된다.
2. 분석물 몫 `ls_frac[:, nonbg] = A[:, nonbg] / ΣA`.
3. **통과 규칙(`hit_mode="threshold"`)**: `hit = Σ_nonbg ls_frac ≥ min_frac`, **min_frac = 0.15** (앱 기본값, "min substance fraction").
   - 대안 `hit_mode="auto"`: 그룹 최소제곱 증거로 임계값 없이 판정(옵션, 기본 OFF).
   - 배경 맵(`bg_map`)이 주어지면 배경 부분공간 점수 `bg_score`로 추가 판정(실험적; 잎맵에서는 효과가 확인되지 않음).
4. 통과 픽셀만 f(x)의 입력이 되고, 조성·농도·파이·분포 전부 이 마스크 위에서 계산된다.

같은 게이트가 학습 데이터 구성(`nnls_hit_spectra`), 벤치마크(`benchmark_loo`, `nnls_screen=True`), 앱(Real 탭)에 동일하게 적용된다. 벤치마크에서 모든 비교 방법(NNLS·PLS·RF·CNN·MLP)이 같은 게이트 통과 픽셀을 받는다.

---

## 3. 조성 MLP f(x)

### 3.1 구조 (`dl_quantify._spec_net`, 번들 `comp_hidden=(256, 64)`)
| 층 | 크기 | 구성 |
|---|---|---|
| 입력 | 1,290 | log1p 스펙트럼 |
| 은닉 1 | 256 | Linear → BatchNorm1d → ReLU → Dropout 0.15 |
| 은닉 2 | 64 | Linear → ReLU |
| 출력 | 4 | Linear → softmax (DQ · TBZ · THI · BLK) |

파라미터 수 ≈ 1,290×256 + 256×64 + 64×4 ≈ 3.5 × 10⁵.

### 3.2 학습 데이터
- 조건: `Pure/mixtures.json`의 준비 혼합물 102조건(파일명 = 최종 µM, 예 `DQ12-TB24-TH6.csv` = 12/24/6 µM). 라벨은 파일명 농도의 **비율**(`_ratio`)이다. 번들 `n_maps = 95` (혼합물 + 배경 참조맵), `n_train = 9,498` 픽셀.
- **픽셀 표집**: 맵당 100픽셀(`px_per_map=100`), `pixel_sampling="representative"` — 스펙트럼 형태(32구간 평균, L2 정규화)와 log 총세기를 합친 피처에 MiniBatchKMeans(k=100)를 돌려 각 군집의 **메도이드(실측 스펙트럼)** 를 뽑는다. 같은 맵·같은 시드(0)면 항상 같은 픽셀. 학습 픽셀에는 NNLS 게이트를 걸지 않았다(`nnls_screen=False`): 혼합물 맵의 어두운 픽셀도 그 맵의 조성 라벨로 학습되어 "세기와 무관하게 조성을 읽는" 성질을 갖는다.
- **BLK 클래스**: `include_blank=True` — 배경 참조맵(BLK)의 픽셀을 "100 % BLK"로 라벨링해 네 번째 클래스로 함께 학습. 이 덕에 배경·잉크 신호가 BLK 출력으로 흡수되고 세 성분 몫이 오염되지 않는다.
- 배경 참조가 3종(BLK/INK/LEAF)이어도 학습 클래스는 하나(BLK)다.

### 3.3 학습 절차 (`dl_quantify.train_composition`)
- 손실: softmax 출력에 대한 **L1 손실**(조성 벡터 절대오차 합).
- 최적화: Adam, 학습률 3 × 10⁻⁴, weight decay 10⁻³.
- 에폭: **350 고정**(`epoch_rule = "fixed-map-pooled"`, `selected_epochs = 350`). 물리 사전학습(`pretrain`)은 사용하지 않음(2026-08-19에 해석용으로만 강등).
- 시드 0.

### 3.4 평가 프로토콜
- **조건 단위 held-out**(leave-one-condition-out 또는 조건군 5-fold). 같은 조건의 픽셀은 절대 학습·검증에 나뉘어 들어가지 않는다. in-sample 조성 오차는 held-out의 2–4배 낙관적이므로 논문 수치는 전부 held-out이다.
- 결과(92조건 FINAL, 조성 겹침률 1 − ½Σ|Δ몫| 평균): MLP 0.83, PLS-R 0.84, NNLS 표면 0.74. 균형 삼성분에서는 PLS와 동률급이고, MLP의 우위는 부재 성분(binary)과 불균형(8:1: 농도 within-2× 50 % vs PLS 17 %)에 있다.

---

## 4. 맵 조성과 표면 편향 복원

- 픽셀 조성 `ratio_nb = A_model[:, nonbg] / Σ` — softmax 4벡터에서 BLK를 빼고 세 성분을 재정규화(합 = 1). 앱의 "Map composition" 파이의 "After MLP"가 이것.
- **맵 조성(보고값)**: 게이트 통과 픽셀의 `ratio_nb`를 픽셀 총세기(베이스라인 제거 후 합)로 가중 평균(`_mean_ratio`). 어두운 가장자리 픽셀이 조성을 희석하는 것을 막는다(글씨맵에서 THI 40 → 32 %로 끌려 내려가던 문제).
- **표면 조성("Before spectral/NNLS")**: 같은 픽셀의 `A_evidence[:, nonbg]` 정규화. 표면(NNLS)은 THI를 과대(+17 %p 수준), MLP는 이를 되돌린다 — 논문의 "surface → restored" 서사.
- 부재 성분: presence head가 ND로 판정한 성분은 0으로 보고(§5).

---

## 5. 존재 판정 — presence head (`*.presence.json`, `presence_v1_20260831`)

softmax 몫은 "얼마나"를 말하지 "있는가"를 확정하지 못한다. 성분별 로지스틱 회귀가 그 역할을 한다.

- 입력 6 피처(성분별 z-정규화, `mu/sd` 사이드카 저장):
  `nnls_self_frac` (자기 NNLS 몫), `nnls_partner_max` (다른 분석물 NNLS 몫 최대), `log1p_band_self` (자기 마커 밴드 신호), `log1p_band_partner_max`, `log1p_total` (총세기), `analyte_share` (분석물 전체 몫).
- 출력 P(존재). **3-상태**: P < 0.2 → ND(부재, 농도 0으로 보고) · 0.2–0.8 → Indeterminate · P > 0.8 → Detected.
- 학습: 124맵(번들 89맵 × 24px + 검량 35), grouped 5-fold(seed 701). AUC THI 0.997 / TBZ 0.940 / DQ 0.910. P ≥ 0.2 이진화 시 F1 THI 0.98 / TBZ 0.94 / DQ 0.93 (전체 0.95).

---

## 6. 농도 g(map) — 검량선 + 잔차 신경망 (`uM` 사이드카, `calibration_residual_uM_v1`)

### 6.1 검량 농도 Ccal
- 검량 스펙트럼: 번들에 내장된 `calibration_spectra_260821.csv` (DQ/TBZ/THI 각 9·18·36·72·144 µM 단일성분).
- 마커 밴드 `bands_cm = [1570 (DQ), 1270 (TBZ), 1367 (THI)]`, 반폭 ±10 cm⁻¹. 밴드 신호 = 창 안 최대(베이스라인 제거 후, 0 이하 절단).
- 성분별 **로그-선형 적합** `signal = a·log10(C) + b` (`loglinear_ab`, `ab_source`: 검량 CSV의 밴드 적합).
- 역산 `Ccal = 10^((signal − b)/a)`, **9–144 µM로 클립**(`cal_range_uM`). 클립 때문에 창 밖에서는 Ccal이 상수가 되고, 그 몫을 잔차 넷이 맡는다.

### 6.2 12개 맵 피처 (`_residual_context`, 맵 단위 풀링 `pool="map"`)
게이트 통과 픽셀에 대해:
1. `log10 Ccal` ×3 (성분별, 픽셀 밴드 신호 중앙값 기준)
2. 모델 조성 ×3 (`ratio_nb` 평균)
3. `log1p` 밴드 신호 ×3 (Ieq)
4. log 총세기의 10 / 50 / 90 백분위 ×3
→ 12차원, 사이드카 `mu/sd`로 z-정규화.

### 6.3 잔차 신경망 (`_residual_net_torch`)
| 층 | 크기 | 구성 |
|---|---|---|
| 입력 | 12 | z-정규화 맵 피처 |
| 은닉 1 | 128 | Linear → BatchNorm1d → ReLU → Dropout 0.25 |
| 은닉 2 | 32 | Linear → ReLU |
| 출력 | 3 | Linear = **Δlog10** (성분별) |

- 후처리: Δ를 **±2 decade로 클립**, `C = Ccal · 10^Δ`, 최종 [10⁻³, 5×10³] µM 클립.
- 학습(`_fit_residual_net`): 목표 `log10(true) − log10(Ccal)`, **마스크드 Huber** 손실(진실 0인 부재 성분은 손실에서 제외), Adam. 에폭 예산은 **조건군 20 % holdout**에서 고르고(선택 122 에폭, `selection_level = condition`), 그 에폭 수로 전체 맵에 재적합. 학습 82맵 / 8,198픽셀.
- 검증(`integrated_v1_20260824`, abs-condition-grouped held-out): **검증 창 3–24 µM**(`validated_ranges_M`)에서 2배 이내 회수. 창 밖은 부정확(THI는 ~18 µM 위에서 밴드가 평탄해져 신호가 농도를 구별하지 못한다 — 데이터로 해결되지 않는 물리 한계).

### 6.4 적용영역 검사 (double check)
- 맵 12-피처 z를 학습맵 라이브러리(`*.knn.json`, 102맵; 100:1 불균형 맵 `imb100`은 제외)와 비교해 **최근접 학습맵까지의 유클리드 z-거리 `dmin`** 을 잰다.
- **`dmin > 3` 이면 g(map)의 값을 보고하지 않는다(무응답)**. 라이브러리가 없을 때는 피처 z 최대 절댓값 > 3(`batch_mismatch`)이 같은 역할. 값을 평균하거나 고치지 않는다 — 영역만 지킨다.
- 같은 라이브러리로 "library k-NN" 옵션 경로도 제공: 최근접 3맵의 실측 µM을 거리 역수 가중 평균한 값으로 g(map) 픽셀맵의 **성분별 중앙값만 재조정**(패턴은 유지). LOO RMSE 21.8 µM, 논문 수치에는 쓰지 않음.

### 6.5 선언 총량(known total, 선택)
시료 총농도 C_total이 주어지면 `Ĉᵢ = p̄ᵢ · C_total` (p̄ = 게이트 통과 픽셀의 조성 평균). 조성만으로 절대량을 되살리는 제약 경로이며, ≤100 µM 맵에서 held-out within-2×를 78 → 91 %로 올린다(추가 정보 덕이지 모델 덕이 아님을 명시).

---

## 7. 픽셀 라이브러리 k-NN (배포 대체 경로, `*.pxknn.npz`)

- 픽셀 서명 7개: `log1p` 밴드 신호 ×3, `log1p` 총세기, 모델 조성 ×3. 라이브러리 전체 `mu/sd`로 z-정규화.
- 라이브러리: 학습맵 **게이트 통과 픽셀 전부**(맵당 상한 없음; 2026-09-04 재구축, `build_pxknn_library.py`) + 검량 스펙트럼(파일당 5 반복 전부). 로더에서 **성분 농도 > 100 µM 조건의 픽셀은 제외**(`PXKNN_UM_MAX = 100`; 검량 스펙트럼은 예외) → 6,828픽셀 / 79조건 + 검량. 고농도 맵의 밝은 픽셀이 실 시료의 밝은 픽셀을 끌어가 폭주시키던 문제(글씨맵 71/14/53 µM)의 사용자 결정.
- 조회: 픽셀마다 유클리드 거리 최근접 **k = 15**, 가중치 1/d, **로그공간 가중 기하평균**(`exp(Σw·log(Y+0.25)) − 0.25`). 픽셀별 최근접 거리의 중앙값 `d_med > 3` 이면 라이브러리 밖.
- **LOO 스위치**: 라이브러리에 있는 맵을 열면 자기 조건 픽셀을 제외할지 선택(기본 OFF = 배포 동작; 같은 조건이 라이브러리에 있으면 정확히 그 값 — 33:33:33 확인).

### 7.1 `auto` 라우팅 (앱 기본)
```
if 맵 최근접 학습맵 z-거리 ≤ 3:      → g(map)      (route "model")
elif 픽셀 라이브러리 d_med ≤ 3:        → pixel k-NN  (route "pxknn")
else:                                   → 무응답
```
그 외 수동 경로: `model head`, `library k-NN`, `pixel k-NN`, `raw VIP band signal`(µM 대신 밴드 카운트), `MLP-corrected signal`(총 밴드 카운트를 MLP 조성으로 배분).

---

## 8. 실측 맵(잎·잉크)에서의 처리 옵션 — Real 탭

- **signal floor**: 게이트 통과 픽셀 중 세 VIP 밴드 신호합이 hit 픽셀 p99의 n % 미만이면 제외(잎 바깥·기판에 깔린 픽셀). 맵·파이·분포·KPI·export가 같은 마스크를 쓴다. 잎맵에서 15 %가 만족스러웠다(2,821 → 688 px).
- **ink area only**: 1000 cm⁻¹ 밴드(raw)의 Otsu로 SERS 잉크 도포 영역을 잡고 바깥을 null. 액적 맵에서는 OFF.
- **saturation quarantine / low-R² drop**: 포화 픽셀·재구성 R² 낮은 픽셀 제외.
- **ROI**: 아무 맵에서나 드래그하면 그 영역만으로 조성·µM·KPI·export를 재계산(상태줄에 ROI vs 전체 조성 병기).
- **표시**: 조성 맵은 신호 밝기로 음영(`shade by signal`), 비-hit 픽셀에 조성 색 밑그림(`colour underlay`), 픽셀 격자(`pixel grid`); 농도 맵은 성분별 map max, merged R/G/B.
- **export**: 선택 대화상자(표 / 합성 그림 / 패널 이미지), `hit` 열 = 화면의 유효 hit, `pixel_distribution.csv`(strip plot 값), `matrix/`(ny×nx 표, Origin 히트맵), README에 필터·ROI·signal floor 기록.

---

## 9. 검증 요약 (논문에 쓰는 수치의 출처)

| 항목 | 프로토콜 | 값 | 파일 |
|---|---|---|---|
| 조성 겹침률(92조건) | 조건 held-out | MLP 0.83 · PLS 0.84 · NNLS 0.74 | `results/27_ternary_truepred_FINAL_*.csv` |
| 선언 총량 하 농도 정확도(64격자) | 조건 held-out | 2배 이내 NNLS 44 · PLS 62 · MLP 61 / 64 | `results/27g_conditions_mean_kt_grid64_po_*.csv` |
| µM within-2× (모델 헤드, 92) | abs-condition 5-fold | MLP 78 % · PLS 84 % · NNLS 75 % | `results/um_benchmark_allmethods_20260831/` |
| µM within-2× by imbalance (64격자) | held-out | 8:1에서 MLP 50 % vs PLS 17 % | `results/09f_grid64_within2x_by_imbalance.csv` |
| 존재 판정 AUC / F1 | grouped 5-fold | THI 0.997/0.98 · TBZ 0.94/0.94 · DQ 0.91/0.93 | `results/presence_head_20260831/` |
| 픽셀 라이브러리 LOO within-2× (≤100 µM 67맵) | LOO | DQ 90 · TBZ 87 · THI 93 % | 2026-09-04 세션 |

주의: "회수율"은 관례대로 검량 범위 안(진실 몫 ≥ 20 % 또는 3–24 µM 창) 조건의 평균 ± SD로 보고하고, 범위 밖 미량 성분까지 넣은 평균은 비율의 꼬리 때문에 과장된다(VIP band THI 215 % → 20 % 하한에서 128 ± 49 %).

---

## 10. 파일과 재현

| 파일 | 내용 |
|---|---|
| `mlp_composition_260831_final.dlm` | f(x) 가중치, 전처리 설정, 검량 CSV 내장, g(map) 가중치·mu/sd·ab·검증 메타 |
| `…presence.json` | 성분별 로지스틱(피처 mu/sd, 계수, 임계값 0.2/0.8) |
| `…knn.json` | 학습맵 12-피처 z 라이브러리(102맵, imb100 플래그) — 적용영역 검사·library k-NN |
| `…pxknn.npz` | 픽셀 라이브러리(Fz, Y, cond, mu, sd, F) — pixel k-NN |
| `Pure/mixtures.json` | 조건 목록(경로, 비율, 농도) — 라벨의 단일 출처 |
| `Pure/colors.json` | 성분 색(DQ #008cf7, TBZ #27a679, THI #f35376) |

재현 스크립트(`documentation/scripts/`): `run_knn_um_readout.py`(맵 라이브러리·LOO), `build_pxknn_library.py`(픽셀 라이브러리), `run_um_benchmark_allmethods.py`(8-방법 µM 벤치마크), `fig27g_ternary_rb_background.py` / `fig27h_digitize_ternary_png.py`(삼각도·배경), `fig_architecture_main_v2.py`(아키텍처 도식).

---

## 11. 설계상 확정된 결정과 그 이유 (요약)

- 게이트는 NNLS(비학습)가 맡고 모델은 마스크를 못 바꾼다 — 배경 처리가 모델 성능과 섞이지 않게.
- BLK를 softmax 클래스로 함께 학습 — 배경 신호가 세 성분 몫을 오염시키지 않게.
- 학습 픽셀은 게이트 없이 대표 표집 — 어두운 픽셀에서도 조성을 읽도록.
- 농도는 "검량 + 잔차" — 검량 창(9–144 µM) 밖은 Δ가 맡되 ±2 decade로 제한.
- 학습 영역 밖은 답하지 않는다(z-거리 3) — 외삽 대신 무응답.
- 픽셀 라이브러리는 ≤100 µM 조건만 — 고농도 밀도 편향 방지(사용자 결정 2026-09-04).
- THI ≥ ~18 µM은 신호가 평탄 — 희석 또는 검열 표기가 정직한 처리이며 모델 보정으로는 해결되지 않는다.
