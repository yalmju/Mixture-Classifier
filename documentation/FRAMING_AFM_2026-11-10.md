# 논문 프레이밍 확정 — Advanced Functional Materials (2026-11-10)

> 추가 실험 없음을 전제로, 지금 있는 데이터가 버티는 주장만으로 세운다.
> Fig 3 구성은 `FIG3_LOGIC_2026-11-09.md` 그대로 — 이 문서는 그 위의 "말"을 정한다.

## 한 줄

**혼합물 SERS 의 한계는 아래(감도)가 아니라 위(자리 경쟁)에 있다.**
표면이 포화되면 가장 센 흡착종이 신호를 가져가 1:1:1 용액이 단일 성분으로 읽힌다.
쓸 수 있는 CNF@Au 잉크에서 그 경쟁을 재고(곡선 하나), 설명하고(DFT), 읽을 수 있는
구간 안에서는 조성을 복원하고 밖에서는 스스로 거부하는 판독을 붙였다.

## 판독 구간(window)이 이야기의 뼈대다

```
아래 경계  ~1–2 µM   흡착 친화도가 정함 (재료·분자 물성)   혼합물에서도 유지 (기울기 76–118%)
구간       3–24 µM   조성 13.1% (반복 하한 9.6%) · 농도 2배 이내 76%
위 경계    33–100 µM 사이  자리 경쟁이 정함                33 µM 은 아직 읽힘(21/39/41) · 100 µM 은 0/0/100
```

- 문헌은 아래 경계(LOD)에서 경쟁한다. **이 논문의 새 내용은 위 경계다** — 왜 생기는지
  (DFT: THI > DQ > TBZ, DQ 는 눕기→세움으로 0.7 eV 손해), 얼마나 휘는지(마스터 곡선:
  THI 와 50:50 이 되려면 DQ 16배·TBZ 8배, 삼각 닫힘 0.04 dex), 어떻게 다루는지
  (구간 안 = 학습 역변환, 구간 밖 = 가드 → 희석).
- 직관을 뒤집는 문장: **혼합물은 진할수록 읽기 어렵다. 필요한 조작은 농축이 아니라 희석이다**
  (같은 1:1:1 조성이 333 → 10 µM 로 내려오면 0/0/100 → 33/21/46).

## nM 에 대한 입장 — 쫓지 않고, 필요 없다는 근거를 댄다

**불가능한 이유**: 검량 최저점이 0.1 µM 이고 LOD 는 0.37–1.74 µM (블랭크 7맵의 맵간 σ,
맵 중앙값 기준). DQ 의 바닥은 흡착 물성(지수 0.52)이라 학습으로 못 넘는다. 실험 추가 없이
nM 을 쓸 방법은 없다.

**필요 없다는 근거 (TBZ·THI 에만 성립)**: 잔류 허용 기준이 ppm 대라 µM–수십 µM 에 해당한다.

```
            허용 기준 (찾은 값)              ≈ µmol/kg      우리 구간 3–24 µM
TBZ  감귤   5–10 mg/kg (EU 5–7·Codex 7·US 10)   25–50         = 0.6–4.8 mg/L
THI  사과·딸기  7 ppm (US, 2005 FR 기준)          29           = 0.7–5.8 mg/L
DQ   감자·곡류  0.02–0.1 ppm (US)               0.1–0.5       = 0.55–4.4 mg/L  ← 구간 밖(아래)
```

즉 TBZ·THI 는 규제 수준 시료가 구간의 **위쪽 끝**에 걸린다 — 문제는 신호 부족이 아니라
경쟁 과잉이다. **DQ 는 규제 수준이 LOD 아래라 이 논거를 쓸 수 없다** — DQ 는 "규제 대응"이
아니라 "흡착 방식이 다른 경쟁자(정전기·눕기→세움)"로서 등장한다고 쓴다.

> ⚠ 인용 전 확인: 위 수치는 2차 자료에서 찾은 값이다. EU 감귤 TBZ 는 자료마다 5/6/7 로
> 갈리고, thiram 7 ppm 은 2005년 고시 기준, 딸기 20 ppm 상향은 확정 여부 미확인.
> EU MRL DB · eCFR 40 CFR 180.132/180.226/180.242 · Codex 에서 현재 값으로 바꿔 쓸 것.
> mg/kg(시료) ↔ µM(측정액) 은 추출 비율에 달려 있어 **자릿수 논거로만** 쓴다.

## 철회 — 2026-11-09 답변의 과장

"단독 LOD 는 혼합물에서 성립하지 않는다 / TBZ 는 THI 5배면 소멸" 은 **쓰면 안 된다.**
- 이원 응답곡선은 전부 총 200–1000 µM(포화 구간) 데이터다.
- 작업구간에서는 우리 데이터가 반대를 말한다: 혼합 기울기가 단독의 76–118%
  (`FINDINGS_2026-08-12.md` "혼합 LOD"), 경쟁 계수 ~0.
- 옳은 문장: "감도는 혼합물에서 유지된다. 무너지는 것은 **충실도**이고, 그것은 포화에서 온다."

## AFM 용 3요소와 그림 배치

| 요소 | 내용 | 그림 |
|---|---|---|
| 재료 | 쓰고 그리는 CNF@Au SERS 잉크 | Fig 1–2 (기존 재료·단일성분 그림) |
| 통찰 | 표면은 경쟁 흡착체다 — 판독 구간의 위 경계를 재고 DFT 로 설명 | **Fig 3 a–e** |
| 기능 | 구간 안 복원 + 구간 밖 거부(희석) — 글씨 맵: 신호 87% THI → 36/30/34 | Fig 3f · **Fig 4** |

벤치마크(NNLS·PLS·RF·CNN)는 SI. 본문에서는 "선형 판독은 구간 안에서 대등하다(PLS p=0.46),
학습의 이점은 범용성"까지만.

## 초록 뼈대 (영문)

> Surface-enhanced Raman sensors are benchmarked by how little of one analyte they detect,
> yet real residues are mixtures, and a SERS surface is a finite set of sites for which
> analytes compete. Here, a writable cellulose-nanofibre/gold (CNF@Au) ink is used to show
> that mixture readout is bounded from above rather than from below: sensitivity is retained
> in mixtures, but once the surface saturates (between 33 and 100 µM per component) site
> competition hands the signal to the strongest adsorber, and an equimolar ternary solution
> reads as a single compound. Binary
> and ternary responses of thiram, diquat and thiabendazole collapse onto one curve displaced
> by about one decade, in the order predicted by DFT adsorption energies. A physics-pretrained
> network inverts this response within the readable window (composition error 13%, close to
> the droplet repeatability floor) and flags saturated samples for dilution, recovering a
> 1:1:1 solution from a hand-written pattern whose raw signal is 87% thiram.

## 제목 후보

1. Reading Pesticide Mixtures with a Writable SERS Ink: Site Competition Sets the Window, Learning Reads Within It
2. Competitive Adsorption Bounds, and Physics-Informed Learning Recovers, Mixture Composition on a Writable Plasmonic Ink

## 예상 질문과 답

- **"LOD 가 µM 인데 문헌은 nM 이다"** → 맵 중앙값·블랭크 맵간 σ 기준의 값이다. TBZ·THI 의
  허용 기준은 ppm(수십 µM)대라 규제 수준 시료는 구간 위 끝에 온다. 기여는 위 경계다.
  DQ 의 저농도 한계는 물성으로 인정한다.
- **"구간 밖은 못 읽지 않느냐"** → 그렇다, 그래서 가드가 거부하고 희석을 지시한다.
  희석으로 회수된다는 근거가 등몰 계열이다.
- **"왜 선형 분해로 안 끝내나"** → 구간 안에서는 대등하다고 먼저 인정. 학습은 표면→용액
  역변환과 픽셀 적용·불확도를 한 망에 묶고, 어려운 조건일수록 격차가 커진다.
- **"DFT 에너지와 곡선의 정량 대응은"** → 순서·부호만. eV ↔ dex 환산 없음.

## 쓰면 안 되는 것 (프레이밍 한정 추가분)

- nM · ultrasensitive · trace-level — 어디에도.
- "LOD 는 혼합물에서 무의미" — 위 철회 참고.
- "quantitative" — 준정량(semi-quantitative).
- 가드를 구현된 기능처럼 — **아직 미구현**(순수 THI 코사인 문턱 규칙). 본문에 넣으려면 구현 필요.

## 미결 (사용자 판단)

- CNF@Au 잉크가 이 논문에서 처음 보고되는가 — AFM 적합성의 최대 변수. 선행 보고가 있으면
  재료 비중을 낮추고 "통찰 + 기능"으로 무게를 옮긴다.

관련: `FIG3_LOGIC_2026-11-09.md` · `DFT_BINARY_RESPONSE_2026-11-09.md` ·
`FINDINGS_2026-08-12.md` · `CLAIM_nnls_vs_learned_2026-08-12.md`
