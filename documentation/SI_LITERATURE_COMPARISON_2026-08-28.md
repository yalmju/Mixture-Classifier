# SI 비교표 초안 — SERS 다성분 정량 문헌 대비 (2026-08-28)

웹 조사 기반 초안. 각 행은 접근 가능한 본문/초록에서 확인된 내용만 기재; 접근 불가
항목은 "미확인"으로 남겼고 **본문 인용 전 원문 확인 필수**. 직접 수치 비교보다
"validation type" 열의 대비가 핵심 — 대부분의 문헌은 랜덤 스펙트럼 분할 또는
in-sample이라 우리의 condition-held-out과 같은 자로 잴 수 없다.

| Study | Analytes (n) | Range | Model | **Validation** | Reported | Mixture quant. | Spatial map |
|---|---|---|---|---|---|---|---|
| **This work (UNMIXR)** | diquat, thiabendazole, thiram (3) | 3–24 µM factorial(64) + binary/고농도 | pixel-wise shared MLP + calibration-anchored residual | **condition-held-out LOO + 독립 20맵** | 조성 12.8 %p · µM RMSE 9.3 · within-2× 79% | ✓ ternary | **✓ per-pixel** |
| MultiplexCR (virus co-infection, PMC11877629, 2025) | 11 viruses + 2/3-혼합 | 50–10⁵ PFU/mL | multitask CNN | 랜덤 8:1:1 스펙트럼 분할 + blind 734 검체 | MAE 0.028 log₁₀ · R²>0.99 (분할) / blind acc 95.8% | ✓ | ✗ |
| Axillary thiols (PMC11262063, 2024) | 4 thiols | 0.05–1 ppm | PLS-1/PLS-2 | 부트스트랩 내부 + **blind 15검체(타 분석자)** | 내부 RMSE<0.046 ppm → **blind <0.123 ppm (~2.7× 악화)** | ✓ (120 혼합) | ✗ |
| Handheld SERS+QuEChERS rice (npj Sci Food 2021) | 4 pesticides | 혼합 0.25–2.5 ppm | 단일밴드 선형회귀 | 삼중반복 + 벤치탑 대조 (held-out ML 없음) | 회수율 83–115% (단일성분) | △ — **"다른 농약 존재 시 정량 곤란"(신호 억제) 명시** | ✗ |
| SERSFormer-2.0 (ACS AMI 2025) | 5 pesticides (spinach 등) | 0–10 ppm, 5 level | multitask transformer | 분할 방식 미확인 (acc 0.999 → 랜덤 스펙트럼 분할 추정) | acc 0.999 · F1 0.992 (multilabel+회귀) | ✓ (multilabel) | ✗ |
| Multi-target fruit juice (Food Chem 2025, S030881462504395X) | TBZ + thiram 포함 | 미확인 | CARS-ELM 등 비선형 회귀 | 미확인 (paywall) | R²≈0.99 (초록) | ✓ | ✗ |
| Triazole tobacco (ACS AgST 2024) | 구조유사 triazole 3종 | 미확인 | SVR | 미확인 (paywall) | 회수율 94.2–106.1% | ✓ | ✗ |

## 본문/디스커션에 쓸 수 있는 관찰

1. **랜덤 스펙트럼 분할의 낙관**: MultiplexCR·SERSFormer류의 R²>0.99, acc≈1.0은
   같은 시료의 스펙트럼이 train/test 양쪽에 들어가는 분할에서 나온다. blind를 별도
   수행한 thiols 논문은 **blind에서 RMSE가 ~2.7배 악화**됨을 스스로 보고 — 우리가
   내부적으로 확인한 in-sample 낙관(≈10×)과 같은 현상의 문헌 근거.
2. **경쟁 신호 억제로 인한 혼합물 정량 실패의 선례**: npj Sci Food 2021은 혼합물에서
   지배 성분(tricyclazole)이 다른 농약 정량을 방해한다고 명시 — 우리의 문제 설정
   (경쟁흡착 구간 정량)이 문헌에서 미해결로 남아 있음을 보여주는 근거.
3. **공간 맵 + 혼합물 정량 동시**는 조사 범위에서 확인되지 않음 — UNMIXR의 차별점.
4. 농도 자릿수: 문헌 ppb–ppm ≈ 우리 µM(수 ppm)과 동일 오더 — 범위 비교는 성분
   분자량 명시 후 ppm 환산으로 제시할 것.

## 원문 링크 (인용 전 확인)

- MultiplexCR: https://pmc.ncbi.nlm.nih.gov/articles/PMC11877629/
- Thiols: https://pmc.ncbi.nlm.nih.gov/articles/PMC11262063/
- npj Sci Food: https://www.nature.com/articles/s41538-021-00117-z
- SERSFormer-2.0: https://pubs.acs.org/doi/10.1021/acsami.4c17777 (repo: https://github.com/BioinfoMachineLearning/SERSFormer)
- Fruit juice multi-target: https://www.sciencedirect.com/science/article/abs/pii/S030881462504395X
- Triazole SVR: https://pubs.acs.org/doi/10.1021/acsagscitech.4c00757
- 리뷰(단일→혼합 농약 SERS): https://pubs.acs.org/acsodf/article/10/24/25158/3654647/
