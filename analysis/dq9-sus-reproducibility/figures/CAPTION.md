# Fig_reproducibility — 캡션

**Fig. X | Reproducibility of the SERS-ink composition readout.**
Independent preparations of identical mixtures (dried analyte spot, then a 2 µL ink
droplet), *n* = 12 conditions.
(**a**) Preparation-to-preparation CV by measurement layer: ink reporter 7.7 % (dashed
line), analyte recovery 24.5 %, composition 25.0 % (DQ), 21.9 % (TBZ), 20.0 % (THI).
The ink reporter is a controlled variable — identical stock and volume in every
preparation.
(**b**) Mass fractions from two independent preparations; dashed line, 1:1. Error bars
are the SD across the 100 spectra acquired within one preparation. *r* = 0.78,
RMSE = 0.127. Their magnitude does not account for the scatter about the 1:1 line.
(**c**) Within-preparation SEM (SD/√100) 0.005–0.008, within-preparation SD
0.048–0.083, between-preparation SD 0.083–0.100. The within-preparation SEM understates
the true error 15-fold: the 100 spectra come from a single preparation and are
subsamples of it, not independent replicates.
(**d**) Discriminability *d′* for a two-fold concentration step versus the number of
preparations averaged; circled point, single preparation (*d′* = 1.13); shaded region,
*d′* < 2.

---

## 색 지정

| 계열 | hex | 출처 |
|---|---|---|
| DQ | `#1a73e8` (BLUE) | `ui_common.py` |
| TBZ | `#4a9e2a` (GREEN) | `ui_common.py` |
| THI | `#c85a8f` (PINK) | `ui_common.py` |
| 중립 램프 (패널 c) | `#c7ccd3` → `#98a1ac` → `#5b6673` | `ui_common.py` |
| 강조 링 (패널 d) | `#d8542a` (CORAL) | `ui_common.py` |

색각이상 검증(all-pairs, surface `#fafbfc`): 최악 쌍 THI↔TBZ ΔE 11.3 (deutan),
일반시야 최악 ΔE 26.0, 대비 3종 모두 ≥ 3:1 — 전 항목 통과.
**TBZ에 `TEAL #0f9d6b`를 쓰면 PINK와 ΔE 4.2로 FAIL**하므로 쓰지 말 것.
산점도에는 색 외 2차 부호화로 마커 모양(DQ ○ · TBZ □ · THI △)을 넣었다.

## 스타일 규칙

- 축 내부에 텍스트 없음. 모든 글자는 축 라벨·눈금·범례(축 바깥 상단)·패널 문자뿐.
- 수치는 전부 캡션으로 이동.
- 출력: 벡터 PDF 7.2 × 6.5 in, Type42 폰트 임베드 + 400 dpi PNG.
- 재생성: `python3 ../scripts/fig_main.py` (색은 `SER` 딕셔너리 한 줄만 고치면 전 패널 반영)
