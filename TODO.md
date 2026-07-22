# UNMIXR — TODO (open items)

_업데이트: 2026-07-22. 5-탭 네이티브 앱(`python unmixr.py`) 기준._

## 현재 상태 (완료된 것)
5-탭 PyQt 앱으로 정착: **Samples · Model · Predict · Quantify · Real data**.
플랫 모듈 구조(`unmixr.py` 셸 + `ui_common.py` + `page_*.py` + 도메인 로직 모듈).
- **Samples** — 파일별 class/batch/role(train·test) 수동 지정, 픽셀 수 표시, `samples.csv` 매니페스트.
  배치 인식 그룹핑(`THI`, `THI_(2)`, `THI_2` → 한 클래스의 배치들).
- **Model** — 실데이터 학습 벤치. 알고리즘 6종(RF·ResNet1D·SVM·k-NN·LogReg·GBM),
  전처리(ALS baseline·미분·L2/SNV/none·trim), 스플릿 5종
  (spatial·random(leaky)·batch·**batch-CV(mean±SD)**·manual). 학습곡선·혼동행렬·PCA·
  per-class F1 + **판별 밴드(ANOVA F per wavenumber)**. 진행률 바.
- **Predict** — 미지 맵 로드 → 픽셀별 NNLS 조성. **RGB 합성맵**(클릭→픽셀 파이차트),
  희석 CSV 로드 시 **픽셀별 절대농도(µM)** + 포화(Σθ>0.85) 경고 + 검량 R² 품질 표시.
- **Quantify / Real data** — 경쟁흡착 정량 · 실데이터 파이프라인(도메인 무관, 농약은 한 예시).
- 도메인 중립화 완료(임의 물질), 아이콘/로고 정리, CTK 중복 도구 삭제·통합.

## 다음 세션 — 실데이터 검증 대기
사용자가 **3배치 × 400포인트 순물질 세트** 수집 중. 준비되면:
1. **Samples**에 파일 로드 → 물질별 class, batch 1/2/3 지정.
2. **Model** split=`batch-CV (mean±SD)` 로 학습 → 배치 간 일반화 정확도(평균±SD) 확인.
   - random(leaky)은 같은 스팟 재사용이라 과대평가; batch/batch-CV가 정직한 지표.
   - 클래스마다 배치 수가 동일하고 ≥2여야 batch-CV 동작(아니면 명확한 ValueError).
3. **판별 밴드(ANOVA F)** 로 물질별 marker 밴드 순위 확인.
   - 주의(유사반복): 맵 안 픽셀은 독립표본 아님 → F는 **밴드 순위**엔 유효하나
     p-value는 배치(replicate map)가 적으면 과대. 배치 수가 곧 독립단위.

## 열린 개선 항목 (선택)
- **검량선 품질** — 현재 저품질 검량으로도 프로그램은 완성. 신뢰할 µM엔
  **넓은 농도 범위 단일성분 희석 시리즈** + **저농도(비포화) 스팟** 데이터 필요.
- **혼합 검출** — THI 응답계수가 커서(과거 관찰 ~13×) 소수성분이 묻힘 →
  픽셀투표로 복구 중. 실측 혼합 세트로 재검증 필요.
- **chemical space(미학습 분자)** — 물질군 확장 시 향후 과제.

## worktree 마무리 (세션 종료 후 수동, 있으면)
```powershell
git worktree prune
git branch -d <머지된 브랜치>   # (선택)
```
