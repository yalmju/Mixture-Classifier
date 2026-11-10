# 논문에 쓸 내용 — 정리 (2026-11-10)

> 브랜치 `paper/pls-composition` 기준. 왜 이렇게 정했는지와 근거 상세는 `FRAMING_AFM_2026-11-10.md`,
> Fig 3 구성은 `FIG3_LOGIC_2026-11-09.md`. 이 문서는 **본문에 무엇을, 어떤 수치로 쓰는가**만 모은다.
> 앱 배포 모델(MLP, `260831_Model_FINAL`)은 그대로 두고, 논문 수치는 전부 논문용 PLS 모델
> (`1. results/261110_Model_PLS_paper/pls_composition_261110.dlm`)에서 나온다. 둘을 섞어 쓰지 않는다.

## 1. 한 줄

**혼합물 SERS 의 한계는 아래(감도)가 아니라 위(자리 경쟁)에 있다.** 쓸 수 있는 CNF@Au 잉크에서 그 경쟁을
재고(응답곡선), 설명하고(DFT), 읽을 수 있는 구간(성분당 3–24 µM) 안에서 조성과 겉보기 농도를 차례로 읽는다.

## 2. 본문 주장과 수치

| # | 주장 | 쓸 수치 | 그림 |
|---|---|---|---|
| 1 | 손으로 쓰고 픽셀 단위로 맵핑하는 SERS 잉크 | 전단 박화(1 → 100 s⁻¹ 에서 점도 40배 감소) | Fig 1–2 |
| 2 | 감도는 혼합물에서도 유지된다 | 혼합 기울기가 단독의 76–118 %, 단독 LOD 0.37–1.74 µM | Fig 3a |
| 3 | 표면이 포화되면 가장 센 흡착종이 신호를 가져간다 | 등몰 100 · 333 µM 에서 신호 판독 0/0/100(THI), 33 µM 은 아직 읽힘 | Fig 3c |
| 4 | 이원·삼원 응답이 한 곡선으로 겹치고 순서는 DFT 와 같다 | THI 와 50:50 이 되려면 DQ 16배 · TBZ 8배, 삼각 닫힘 0.04 dex, 흡착 순서 THI > DQ > TBZ | Fig 3b · d |
| 5 | 밴드 세기만으로는 혼합물 조성을 못 읽는다 | 64 조건 조성 오차 29.1 %p (밴드 높이와 자기 농도의 r² DQ 0.08 · TBZ 0.63 · THI 0.57) | Fig 4b 왼쪽 |
| 6 | 측정 혼합물로 학습한 모델은 용액 조성을 읽는다 | 64 조건(전부 held-out) **12.3 %p**, 평균 편향 DQ −0.5 · TBZ +1.6 · THI −1.1 %p | Fig 4b 오른쪽 |
| 7 | 이어서 성분별 겉보기 농도를 읽는다 | 조건 held-out 2배 이내 **80.8 %** (DQ 75 · TBZ 77 · THI 90), 3배 이내 94.2 %, 중앙 배수 1.39× | Fig 4 (패널 미정) |
| 8 | 신호가 THI 하나로 보이는 글씨에서 세 성분을 모두 푼다 | 게이트 통과 803 픽셀: 신호 분해 0/0/100 → 모델 41/23/36 (참 33/33/33) | Fig 4c |
| 9 | 구간 밖은 희석으로 다룬다 | 같은 1:1:1 이 333 → 33 → 10 µM 로 내려오면 모델 판독 29/17/54 → 28/43/29 → 42/20/39 | Fig 3c · SI `eq_series` |
| 10 | 잎 위에서 그대로 읽는다 | (이 모델로 아직 다시 뽑지 않음) | Fig 5 |

비교 사다리(같은 64 조건 · 같은 픽셀): **밴드 세기 + 검량선 29.1 → NNLS 22.4(SI) → PLS 모델 12.3 %p.**
NNLS 는 SI 로 내리고 템플릿 구성(순수 3 + 배경 BLK · INK · LEAF)을 방법란에 적는다. BLK 하나만 넣은 판은 16.7 %p.

## 3. 판독 절차 — 방법란 서술

1. **픽셀 게이트.** 각 픽셀 스펙트럼을 순수 3성분 + 배경 템플릿으로 NNLS 분해하고, 분석물 신호 몫이
   15 % 이상인 픽셀만 남긴다(`unmix.py` 373–381행, `hit_mode="threshold"`). 코드가 보는 것은 "잉크 유무"가
   아니라 "분석물 신호 몫"이다. 우리 시료는 분석물이 잉크에 들어 있으므로 통과 픽셀 = 잉크가 쓰인 픽셀.
   본문 표현: **"잉크가 쓰인 픽셀(분석물 신호 몫 ≥ 15 %)"**.
2. **조성 — PLS(잠재변수 8).** 측정 혼합물 82맵 + 순수·blank 13맵(95맵 9,498 스펙트럼)으로 학습.
   입력은 전처리한 전체 스펙트럼, 출력은 세 성분의 몫.
3. **농도 — 작은 신경망(128–32).** 맵 단위 특징 12개(특성 밴드 세기 3, 조성 3, 조성으로 가중한 밴드 신호 3,
   총세기 분위수 3) → 성분별 겉보기 µM. 조성과 달리 여기서는 신경망이 선형보다 낫다(§5-2).
   **"단일성분 검량선을 신경망이 보정한다"고 쓰지 않는다** — 구간 안에서는 검량 역산이 64격자 52맵 × 3성분
   전부 하한 9 µM 에 잘려 상수다. 쓸 말: "밴드 세기 · 조성 · 총세기로부터 겉보기 농도를 회귀한다".

조성 단계가 PLS 이므로 본문의 "network" 는 조성에 대해 "model" 로 쓴다. 이름 DUET-Net 은 두 단계 전체의 이름으로 유지.

## 4. 오늘 정한 것 (11-10)

- 조성 단계는 **PLS**, 농도 단계는 **신경망**. 조성에서 "MLP 가 PLS 보다 낫다"는 문장은 쓰지 않는다
  (반복 CV 에서 MLP 가 이긴 횟수 0/20).
- Fig 4b 본문은 **Band intensity | PLS** (`fig4b_ternary`). NNLS 삼각도 두 판은 SI.
- **"총량을 안다"는 전제는 쓰지 않는다.** 세 물질이 서로 다른 화합물이라 실제 시료에서 합계 농도를 안다는
  가정이 성립하지 않는다. 따라서
  - 농도 주장은 농도 신경망의 성분별 겉보기 µM(80.8 %)만 쓴다. "판독 비율 × 선언 총량" 의 91.0 % 는 버린다.
  - Fig 4b 막대의 "2배 이내 34 → 62 / 64" 는 수식상 Ĉ/C = (판독 몫)/(참 몫) 이라 총량이 약분된다. 그림은 그대로
    두되 **정의를 "몫의 배수 오차"로 적고 농도라고 부르지 않는다.** 본문 대표 수치는 조성 오차 29.1 → 12.3 %p.
- 구간은 **성분당 3–24 µM** 로 쓴다. 합계 9–72 µM 는 격자가 덮은 범위로만 언급한다.
- 이원 추가 측정은 하지 않는다. 이원은 기존 고농도 계열로 설명한다(§5-3). 구간 수치는 세 성분이 모두 있는
  격자에 대한 것으로만 쓴다.

## 5. 근거 표

### 5-1. 구간 — 조성 (PLS 모델, held-out 102맵, `results/fig4_paper_pls_2026-11-10/heldout_maps.csv`)

| 맵 | n | 조성 오차 |
|---|---|---|
| 64격자 전체 | 64 | 12.3 %p |
| 격자 안, 합계 9–20 µM | 11 | 9.8 |
| 격자 안, 합계 21–36 µM | 28 | 13.1 |
| 격자 안, 합계 37–54 µM | 21 | 12.7 |
| 격자 안, 합계 55–72 µM | 4 | 11.7 |
| 격자 밖 전체 | 38 | 22.7 |
| 삼원, 최대 성분 ≤ 33 µM (등몰 10 · 33) | 2 | 13.5 · 9.6 |
| 삼원, 등몰 100 µM | 1 | 32.0 |
| 삼원, 최대 성분 > 100 µM | 8 | 27.8 |
| 이원, 최대 성분 > 100 µM | 27 | 21.8 |

구간 안에서는 합계가 늘어도 나빠지지 않고, 구간 밖에서 22–32 %p 로 무너진다. 33 과 100 µM 사이에는 측정이 없다.

### 5-2. 농도 단계 — 신경망 대 선형 (`results/conc_head_2026-11-10/`)

논문용 모델의 학습 82맵, 같은 12개 특징 · 같은 목표 · 같은 fold. 64격자 52맵 · 성분 156개, 조건 묶음 5-fold × 20회.

| 농도 단계 | 2배 이내 | 1.5배 이내 | 3배 이내 | 중앙 배수 | MAE |
|---|---|---|---|---|---|
| **신경망 (128–32)** | **81.3 ± 3.1 %** | 59.3 % | 93.7 % | 1.38× | 5.5 µM |
| Ridge | 70.5 ± 1.2 % | 49.3 % | 88.5 % | 1.51× | 7.9 µM |
| PLS | 69.3 ± 1.8 % | 47.3 % | 87.7 % | 1.55× | 8.3 µM |
| 검량 역산만 | 48.7 % | 48.7 % | 100 % | 2.67× | 6.8 µM |

신경망이 20회 중 20회 낫다. 차이는 DQ(76 대 64) · TBZ(76 대 58)에서 나고 THI 는 같다(92 대 89).
조건 하나씩 빼기로 돌리면 80.8 % · 94.2 % · 1.39× 로 모델 파일 기록과 일치한다.
참 농도별 2배 이내(신경망): 3 µM 76 % · 6 µM 81 % · 12 µM 88 % · 24 µM 79 % — 50 µM 이상은 28–67 % 로 떨어진다
(`level_and_binary.txt`). 조성 입력은 fold 마다 다시 맞추지 않았다(모델 기록과 같은 조건).

### 5-3. 이원 — 기존 데이터

- 7월 비율 계열(`Ratio/Binary`, 28맵, 세 쌍 100:1 → 1:100, 합계 200–1000 µM): 픽셀 파이 맵과 이원 응답곡선(Fig 3b).
- 최종 세트의 고농도 이원 27맵(성분 50–500 µM, 전부 held-out). 없는 성분을 읽은 몫(중앙 / 최대):

  | 없는 성분 | NNLS (파이프라인) | PLS 모델 |
  |---|---|---|
  | DQ | 0.1 / 0.8 % | 14.9 / 21.7 % |
  | TBZ | 0.1 / 0.3 % | 6.1 / 16.6 % |
  | THI | 0.1 / 0.2 % | 2.5 / 13.6 % |

  유무는 NNLS 로, 비율은 PLS 로 읽는다고 나눠 쓴다. 구간(3–24 µM) 안의 이원은 측정한 적이 없으므로
  "구간 안에서 이원도 읽는다"는 쓰지 않는다.

## 6. 본문 문장 초안 (영문)

- Window: "Within 3–24 µM per component, the composition read by the model deviates from the dispensed
  composition by 12.3 percentage points on average over 64 held-out conditions, with no trend in total
  loading; at 100 µM per component and above the reading collapses towards thiram."
- Baseline: "Reading the same pixels from characteristic band heights and single-component calibration
  curves gives 29.1 percentage points, because in mixtures the band height no longer follows the
  concentration of its own analyte (r² = 0.08, 0.63 and 0.57 for diquat, thiabendazole and thiram)."
- Concentration: "A second stage regresses the apparent concentration of each component from band
  intensities, the read composition and the overall intensity; 81 % of held-out components fall within a
  factor of two of the dispensed value."
- Lettering: "In a hand-written pattern of an equimolar mixture, spectral unmixing assigns every written
  pixel to thiram, whereas the model reads 41 : 23 : 36 (diquat : thiabendazole : thiram)."
- Upper bound: "Mixtures become harder, not easier, to read as they become more concentrated; the remedy
  is dilution rather than enrichment."

## 7. 쓰지 않는 것

- nM · ultrasensitive · trace-level · quantitative(→ semi-quantitative, apparent concentration).
- "physics-pretrained / physics-informed" — 최종 모델에 사전학습이 없다.
- "LOD 는 혼합물에서 무의미하다", "포화 시료를 거부/감지한다"(적용영역 검사는 포화를 못 거른다 — 위 경계는 희석 규칙).
- "단일 배치 (single batch)".
- 조성에서 "신경망이 PLS 보다 낫다", 분할 한 번의 수치로 하는 우열 주장.
- "총량을 알면 …" 식의 농도 보고, 글씨맵으로 하는 농도 주장(THI 픽셀이 3배 높게 읽힘).
- "3–90 µM", "고농도까지 복원", "구간 안 이원", 박리에 대한 "non-destructive"(데이터 없이).

## 8. 남은 것

- 잎(Fig 5)을 PLS 모델로 다시 뽑기. LEAF 배경을 학습에 넣을지(`train_paper_pls.py` 의 `SKIP_REF`).
- 존재 판정 · kNN 적용영역 보조 파일은 PLS 모델용으로 만들지 않았다.
- INK 템플릿이 무엇을 잰 것인지 확인해 방법란에 적기.
- `eq_series` 왼쪽 패널은 아직 NNLS(파이프라인). 밴드 세기로 바꿀지는 미정.
- 농도 신경망을 검량 단계 없이 다시 학습할지(지금은 모델 그대로, 서술만 §3 처럼). 시험 12맵의 µM 은 따로 뽑지 않았다.
- Fig 2c 재측정, 잔류 허용 기준 인용값 확인(`FRAMING_AFM` 의 표), 명칭 통일.
- (선택) 48 µM 탐색 7조건 — 위 경계가 성분별인지 합계인지 가르는 실험. 하지 않아도 본문 주장은 선다.

## 9. 파일

| 내용 | 위치 |
|---|---|
| 논문용 모델 학습 | `scripts/train_paper_pls.py` → `results/paper_pls_2026-11-10/` |
| 농도 단계 입력 불일치 수정 | `dl_model.py` (`train_model` 의 PLS/RF 분기, `apply_uM_pixels`) |
| 학습기 재평가(조성) | `scripts/learner_reeval_run.py` · `learner_reeval_analyze.py` → `results/learner_reeval_2026-11-10/` |
| 고농도 · 구간 확장 검토 | `results/window_highconc_2026-11-10/` |
| 농도 단계 비교 · 이원 표 | `results/conc_head_2026-11-10/` (실행한 그대로의 스크립트 포함) |
| Fig 4 | `scripts/fig4_paper_data.py` → `fig4_paper_figs.py` → `results/fig4_paper_pls_2026-11-10/` |
| Fig 3 · 3f | `scripts/fig3_final.py` · `fig3f_network.py` |
