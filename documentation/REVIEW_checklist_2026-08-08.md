# 체크리스트 검토 — SERS 농약 정량 파이프라인 (2026-08-08)

검토 대상은 드라이브 작업본 `github/Mixture Classifier`. `archive/` 는 제외했다.

| # | 항목 | 판정 |
|---|---|---|
| 1 | 데이터 분할 | 문제 있음 → **수정 완료 (2026-08-08)**. 18% vs 4%의 원인은 아니었다 |
| 2 | 전처리 누수 | 문제 없음 (1곳 명시 필요) |
| 3 | 응답인자 | **문제 있음** — 단 전제는 정정 |
| 4 | 벤치마크 공정성 | **문제 있음** |
| 5 | 검증 실험 | **대부분 미구현** |
| 6 | 라벨·수치 일관성 | **문제 있음** |

---

## 1. 데이터 분할 — 문제 있음 (심각)

### 현재 동작

분할 그룹의 단위가 **어디에서나 "맵 파일"** 이고, 조성 조건이 아니다.

| 위치 | 코드 | 그룹 |
|---|---|---|
| `dl_model.py:246` | `paths.append(it[0])  # group key: the MAP, not the row` | 맵 |
| `dl_model.py:679` | `gkey.append(it[0])   # group = the MAP` | 맵 |
| `dl_model.py:193` | `_group_validation_indices` — `unique = dict.fromkeys(groups)` | 맵 |
| `dl_model.py:640` | `kfold_stability` — `items` 를 basename 정렬 후 `pos % folds` | 맵 |

같은 조성을 여러 번 찍은 반복맵이 실제로 있다.

- `Ratio/Baseline260808/DQ36/DQ36-TB36-TH36.csv` 와 `…-TH36-2.csv` (같은 12/12/12)
- `tert-new-baseline/{n}-1, {n}-2, {n}-3` (조성당 3반복)
- `tert-new` 6조건 중 3개는 `Baseline260808` 과도 같은 조성
  (`#2`=12/12/12, `#3`=24/6/6, `#5`=24/24/24)

따라서 한 맵을 test로 빼도 **같은 조성의 형제 맵이 train에 남는다.** 모델은
그 조성의 정답을 이미 본 상태로 시험을 본다.

`kfold_stability` 는 더 나쁘다. basename 정렬 뒤 `pos % folds` 스트라이드라
`1-1, 1-2, 1-3` 이 정렬상 인접하고, 그래서 **반드시** 서로 다른 fold로 흩어진다.
무작위 분할이 우연히 누수를 만드는 게 아니라, 체계적으로 매번 만든다.

### 18% 와 4% 의 정체 — 분할 차이가 아니다

**확인 결과, 4%는 이 누수 때문이 아니다.** 4%가 나온 고농도 `Ratio_mix` 세트를
직접 세어 보면

```
맵 32 · 고유 조성비 32          ← 반복 조성이 하나도 없다
```

조성이 전부 서로 다르므로 그 세트에서는 **맵 단위 분할 = 조건 단위 분할**이다.
형제 맵이 train에 남을 수가 없다. 즉 고농도 4%는 프로토콜 때문에 부풀려진 값이 아니다.

두 숫자를 가르는 것은 **분할이 아니라 데이터**다.

| | 세트 | 농도 | 조성 오차 |
|---|---|---|---|
| 4% | `Ratio_mix` 고농도 32맵 | 강신호 | 4% |
| 12.9% | 저농도 35조건 51맵 | 최종 3–24 µM, LOD 근처 | 12.9% |

저농도가 나쁜 것은 당연하고, 논문에서 대조해야 할 축도 이쪽이다.

다만 **누수 자체는 여전히 실재한다** — 반복맵이 있는 쪽, 즉 저농도 세트에서다.
`Baseline260808` 의 `-2` 중복과 `tert-new` 3반복, 그리고 두 세트에 겹치는 3조성이
그 대상이다. 오늘 `retrain38.py` 는 조성 튜플을 fold 키로 써서 이미 막고 있다.
막지 않은 것은 **앱 안의 경로들** (`benchmark_loo`, `kfold_stability`,
`_group_validation_indices`) 이고, 저농도 데이터를 거기에 태우는 순간 부풀려진다.

앱이 출력하는 "composition error" 가 세 종류라는 점도 그대로 문제다.
`page_compose.py:527-528` 은 라벨을 정직하게 붙이지만

```python
kind = ("leave-one-map-out" if model.get("loo_eval")
        else "independent test batch" if model.get("test_eval") else "train-set")
```

보고서·CSV로 빠져나가면 이 `kind` 가 떨어진다. 슬라이드에 옮겨 적힌 숫자가
어느 프로토콜이었는지 사후 특정이 안 되는 것은 그래서다.

### 수정안

그룹 키를 파일 경로가 아니라 **조성 튜플**로 바꾼다.

```python
# dl_model.py — 맵 경로 대신 조성을 그룹 키로
def _group_key(it, subs):
    """같은 조성(정규화 비율)이면 같은 그룹. 반복맵·중복조성이 자동으로 묶인다."""
    v = _ratio([float(it[1].get(s, 0.0)) for s in subs])
    return tuple(np.round(v, 6))
```

- `dl_model.py:246,249` → `paths.append(_group_key(it, subs))`
- `dl_model.py:679` → `gkey.append(_group_key(it, subs))`
- `kfold_stability` → `items` 를 basename이 아니라 `_group_key` 로 묶고,
  **그룹 목록을 셔플한 뒤** fold에 배정 (정렬 후 스트라이드 금지)

### 적용된 수정 (2026-08-08, `dl_model.py`)

`paths` 는 **맵**으로 그대로 두고 (풀링 키·맵 수·경로 역참조가 이걸 쓴다),
분할용 키 `conds` / `abs_conds` 를 나란히 만들어 **경계를 긋는 곳에서만** 쓴다.

| 곳 | 전 | 후 |
|---|---|---|
| `train_model` 행 적재 | `paths` 만 | `conds` (조성) · `abs_conds` (조성+절대농도) 추가 |
| 에폭 진단 분할 | `_group_validation_indices(paths)` | `(conds)` |
| 조성 LOO | leave-one-**map**-out | leave-one-**condition**-out |
| µM 용량 선택 분할 | `(gp)` = 맵 | `(gcond)` = 조건 |
| µM LOO | 맵 단위 | 조건 단위 |
| `benchmark_loo` | `gkey = it[0]` | `gkey = ckey` |
| `kfold_stability` | basename 정렬 후 스트라이드 | 조성으로 묶고 **셔플** (`seed`) |

설계상 짚어둘 두 가지.

- **풀링과 분할을 분리했다.** µM 헤드의 손실은 "맵당 중앙값 = 라벨" 이라 `gp` 는
  맵이어야 한다. 여기까지 조건으로 바꾸면 한 조성의 반복맵이 하나의 가짜 맵으로
  뭉쳐져 **헤드가 배우는 대상 자체가 달라진다.** 그래서 `gcond` 를 따로 뒀다.
- **농도 키에는 절대농도를 포함**했다 (`akey`). 같은 비율이라도 100 µM와 1000 µM는
  농도 헤드에게 다른 조건이고, 한쪽으로 다른 쪽을 맞히는 것은 정당한 일반화다.
  조성 헤드에는 비율만 쓴다.
- LOO 결과는 **맵당 한 행**으로 보고한다. 한 조건에 맵이 여럿일 수 있고,
  `_annotate` 와 Recovery 탭이 파일 경로로 역참조하기 때문이다.

검증 (`nnls`, 중복 조성 포함 6맵):

```
level condition · 보고 행 6 (= 맵 수) · fold 4 (= 고유 조성 수)
12/12/12 세 맵이 같은 fold로 함께 빠짐
```

MLP + 농도 헤드 (7맵 / 5조건): 조성·µM 모두 `level=condition`, 행 7, `n_maps` 7 유지.

### 덤으로 나온 데이터 문제

두 폴더에 **같은 측정이 중복 저장**되어 있다 (md5 동일, 51파일 중 고유 48).

```
Ratio/Baseline260808/DQ36/DQ36-TB36-TH36-2_corrected.csv == tert-new-baseline/2-1_corrected.csv
Ratio/Baseline260808/DQ72/DQ72-TB72-TH72_corrected.csv   == tert-new-baseline/5-1_corrected.csv
Ratio/Baseline260808/DQ72/DQ72-TB18-TH18_corrected.csv   == tert-new-baseline/3-1_corrected.csv
```

조성 키로 묶이므로 학습·검증 누수는 없다. 다만 **반복 산포를 이 파일들로 계산하면
안 된다** — 독립 반복이 아니라 같은 파일이라 산포가 0으로 깔린다.
하한 9.6%가 어떤 파일 집합에서 나왔는지 확인이 필요하다.

논문에는 **leave-one-mixture-condition-out** 숫자를 쓰고, 맵 단위 숫자는
같이 싣되 "반복맵이 train에 남는 낙관적 프로토콜" 이라고 밝힌다.

측정일·배치 정보는 현재 코드가 갖고 있지 않다 (`Baseline260808` 처럼 폴더명에만
날짜가 있다). leave-one-batch-out을 하려면 폴더명을 배치 키로 파싱해 넣어야 한다 —
지금은 불가능하지 않고, 그룹 키 함수 하나만 바꾸면 된다.

---

## 2. 전처리 누수 — 문제 없음 (1곳 명시 필요)

- **조성 특징**: `_composition_features` = `np.log1p(x)` (`dl_model.py:31-38`).
  행별 연산이고 데이터셋 통계를 쓰지 않는다 → 누수 없음.
- **ALS baseline / NNLS 스크리닝**: 맵마다 독립으로 돈다 (`dl_model.py:236-244`) → 누수 없음.
- **µM 헤드 표준화**: 평가 경로는 train에서만 fit하고 test에 transform한다
  (`dl_model.py:546-547` `fmu/fsd` ← `ftrain`, `567` 에서 test에 적용) → **올바름**.
  `503` 의 전체 데이터 `mu/sd` 는 배포용 최종 적합이라 test가 없다 → 정당.
- PCA는 쓰지 않는다.

**명시할 것 한 가지.** 물리 사전학습(`dl_model.py:300-312`)은 검량 CSV 전체로
5,000개 시뮬레이션을 만들어 **모든 fold에 공통으로** 투입된다. 검량은 단일성분
희석계열이고 test는 혼합물이라 라벨 누수는 아니다. 다만 체크리스트 3번의
"검량 스펙트럼이 학습셋에 있으면 제외" 요구와 맞물리므로, 논문에는
"사전학습은 단일성분 검량에서만 유래하며 혼합물 test 조성을 보지 않는다" 를 적어야 한다.

---

## 3. 응답인자 — 문제 있음 (전제는 정정)

### 전제 정정: MLP에서 나오지 않는다

라이브 경로의 응답인자는 **NNLS** 에서 나온다.

```
validate.py:200  r = unmix_map(data_dir, path, method="nnls", ...)
validate.py:205  obs = {nb[k]: float(r.mean_ratio[k]) ...}
validate.py:215  response, ref, response_se = _response_factors(rows, names)
```

`dl_quantify.py:204 fit_response_factors` 는 `archive/experiments/` 에서만 쓰이고
라이브에서는 호출되지 않는다. 그러니 "MLP 순환논리" 는 아니다.

### 그래도 순환은 있다 — 종류가 다르다

`_response_factors` 는 **보정할 혼합물 자신** 으로 보정계수를 적합한 뒤
같은 혼합물에 되먹인다 (`validate.py:215-217`). held-out이 없다.
`page_validate.py:1082` 의 `e0 → e1` 개선분은 전부 in-sample 이다.
논문에 그대로 실으면 "보정 후만 보고" 로 읽힌다.

### 작성한 코드

`documentation/scripts/rf_calib.py` — MLP도 혼합물도 거치지 않고 단일성분
희석계열에서만 응답인자를 낸다. 세 경로를 나란히 뽑는다.

- **(a)** 대표 밴드 피크 높이(국소 선형 baseline 차감) vs 농도 선형회귀
- **(b)** NNLS 계수 B의 Langmuir 적합 → 초기기울기 `gA·K` (C→0 외삽)
- **(c)** 세 성분 공통 저농도 창에서 B 직접 선형회귀 (적합범위 교란 제거)

대표 밴드는 배타성(그 파수에서 해당 성분이 차지하는 비율) ≥ 0.80 을 강제해 고른다.
피크 높이는 `h = B·P[i,w]` 이므로 템플릿 진폭 `P[i,w]` 로 나눠야 (b)·(c)와 같은
축(dB/dC)에 놓인다 — 이 보정을 빼면 밴드가 뾰족한 성분이 무조건 과대평가된다.

### 결과

```
대표 밴드     DQ 1175.7 (배타성 0.93) · TBZ 1009.9 (0.91) · THI 1367.9 (0.88)

응답인자 (anchor DQ = 1)
          (a) 피크    (b) gA·K 외삽    (c) 공통창 B
  TBZ        2.69            4.78           1.26
  THI       10.54            3.34           3.00

선형성 (혼합물 농도 3–24 µM 에서의 K·C)
  DQ    K=7.05e3   1/K=141.8 µM   K·C 0.021–0.169   준선형
  TBZ   K=3.39e4   1/K= 29.5 µM   K·C 0.102–0.813   포화 진입
  THI   K=3.10e4   1/K= 32.2 µM   K·C 0.093–0.744   포화 진입
```

**두 가지가 나온다.**

1. **응답인자가 하나로 굳지 않는다.** 정당한 세 추정이 3~8배로 흩어진다.
   보정하려는 편향보다 추정 산포가 크다. 저농도 창에 점이 4개뿐이고
   R²가 0.71–0.97로 들쭉날쭉한 것이 직접 원인이다.
2. **혼합물 농도가 선형구간을 벗어난다.** TBZ·THI는 최대 K·C가 0.74–0.81 로
   등온선의 휜 구간에 있다. 이 구간에서 "농도에 무관한 상수 응답인자" 는
   정의 자체가 성립하지 않는다 — 응답인자가 농도의 함수다.

### 수정안

- 상수 응답인자를 단일성분 검량에서 뽑아 **고정 입력으로 교체하는 안은 권하지
  않는다.** 위 2번 때문에 그 상수가 존재하지 않는다.
- 대신 이미 있는 **경쟁 Langmuir**(`calibration.py:116 concentration_from_coverage`)
  가 K를 통해 포화를 직접 다룬다. 농도 판독은 그쪽으로 간다 —
  인계 노트의 "확정 경로" 와 같은 결론이다.
- 상수 RF를 굳이 보고해야 한다면 **세 경로의 산포를 불확실도로 함께** 싣는다.
  단일 숫자로 쓰면 재현되지 않는다.
- 검량 점을 저농도(0.1–5 µM)에서 더 찍으면 (c)의 R²가 올라가 산포가 줄어든다.

---

## 4. 벤치마크 공정성 — 문제 있음

- `benchmark_loo(methods=("nnls","pls","rf","cnn","mlp"))` (`dl_model.py:660`)
  에 **응답인자 보정 NNLS 조건이 없다.** NNLS는 무보정 그대로, MLP는 라벨로
  응답편향을 암묵 학습한 상태로 맞붙는다. MLP 우위의 상당 부분이
  "응답인자를 아느냐" 차이일 수 있고, 현재 설계로는 분리가 안 된다.
- **수정안**: `_fit_predict` 에 `"nnls_rf"` 를 추가한다. train fold의 NNLS 관측과
  참값으로만 응답인자를 적합하고(`validate._response_factors` 재사용),
  test fold에 `correct_fractions` 로 적용한다. fold 안에서 적합해야
  3번의 in-sample 문제를 되풀이하지 않는다.
- 이름 충돌: 여기의 `"rf"` 는 **random forest**, `validate.py` 의 RF는
  **response factor** 다. 같은 표에 올리면 반드시 오독된다. 새 조건은
  `nnls_rfcorr` 처럼 구분되게 쓴다.
- "모든 모델을 같은 분할에서" 는 1번을 고치면 자동으로 해결된다
  (`benchmark_loo` 도 맵 단위이므로).

---

## 5. 검증 실험 — 대부분 미구현

- **음성 대조**: 구현 없음. `include_blank` (`dl_model.py:271`) 로 blank를 하나의
  클래스로 학습하는 길은 있으나 기본값 `False` 다.
  **구조적 한계도 있다** — 조성 헤드는 softmax라 출력 합이 항상 1이다.
  blank 클래스를 켜지 않으면 부재 성분이 정확히 0이 되는 것은 원리적으로 불가능하고,
  모델은 없는 물질에도 지분을 나눠줄 수밖에 없다. 음성 대조를 하려면
  `include_blank=True` 가 전제다. 이건 논문에 명시해야 할 성질이다.
- **외삽 검증**: µM 헤드에만 가드가 있다 (`ood_threshold` = 학습 특징거리의 99분위,
  `ranges_M`, `dl_model.py:526-527,797`). **조성 헤드에는 없다.**
- **보정 전/후 회수율**: 둘 다 출력된다 (`page_validate.py:1082` `e0 → e1`) ✓
  다만 3번대로 e1은 in-sample이다.
- **RSD**: **전무.** `validate.py:228` 이 `mean_recovery` (평균)만 만든다.
  SD도 RSD도 없다 → SANTE 기준(회수율 70–120%, RSD ≤ 20%) 비교가 지금은 불가능.
  **수정안**: `ValidateResult` 에 `recovery_sd` / `recovery_rsd` 를 추가하고
  `acc[n]` 리스트에서 `np.std(..., ddof=1)` 로 함께 계산한다. 값은 이미 모여 있어
  몇 줄이면 된다.

---

## 6. 라벨·수치 일관성 — 문제 있음

- **같은 "응답인자" 가 두 규약으로 정규화된다.**
  - `validate.py:149-150` — 최솟값을 1로 (`mn = min(rf.values())`)
  - `dl_quantify.py:215` — 기준 성분을 1로 (`r = np.exp(logr - logr[ref])`)

  같은 데이터에서 다른 숫자가 나온다. 어느 규약인지 밝히지 않으면 대조 불가.
  한쪽으로 통일하거나, 출력에 anchor를 항상 병기한다
  (`page_validate.py:632` 는 병기하고 있다 — 이쪽이 맞다).

- **"prediction-to-true ratio" 가 두 뜻으로 쓰인다.**

  | 뜻 | 정의 | 단위 | 제안 명칭 |
  |---|---|---|---|
  | 98% | 측정 µM / 참 µM | % | **회수율 (recovery)** |
  | 29.7× | 표면 관측비 / 용액 참비 | 무차원 배수 | **SERS 응답인자 (response factor)** |

  후자는 "농도당 신호(signal per unit concentration)" 의 상대값이다.
  두 숫자는 축이 달라 같은 이름으로 부르면 안 된다.

- **슬라이드 3 vs 4**: 1번에서 본 대로 코드가 만드는 "composition error" 가
  세 종류다. UI는 `kind` 로 라벨하지만 CSV·보고서로 나가면 그 라벨이 사라진다.
  **수정안**: `composition_metrics.csv` 와 보고서 본문에 프로토콜 문자열
  (`train-set` / `leave-one-map-out` / `leave-one-condition-out`)을 **한 칸으로**
  같이 적는다. 그러면 슬라이드로 옮겨도 따라간다.

---

## 우선순위

1. **1번 그룹 키 수정** — 논지가 걸려 있고, 고치면 4번의 "같은 분할" 도 따라온다
2. **5번 RSD 추가** — 몇 줄, SANTE 비교의 전제
3. **4번 `nnls_rfcorr` 조건 추가** — MLP 우위의 근거를 분리
4. **6번 명칭·프로토콜 라벨 정리** — 측정 없이 문서만으로 가능
5. **3번** — 검량 저농도 점을 더 찍기 전에는 상수 RF를 확정하지 않는다
