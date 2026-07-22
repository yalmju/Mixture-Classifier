# UNMIXR — TODO (open items)

_남긴 날: 2026-07-22. 다음 세션에서 이어서._

## 1. Model 페이지 중복 정리 ★ — [x] 해결 (재설계로)
결정: (A/B/C) 대신 **Model 페이지를 실데이터 학습 도구로 탈바꿈**. 이제 합성 데모가
아니라 **실제 pest Reference 맵(DQ/THI/TBZ/BLK)** 을 folder-picker로 로드해
**단일성분 분류기를 직접 학습**함 → Discriminator(RF, 혼합/전략)와 역할이 갈림.
- 새 모듈 `model_training.py` (UI 무관, numpy/sklearn; torch는 지연 import).
- 백엔드 2종 선택: **RandomForest**(OOB 학습곡선) · **ResNet1D**(에폭별 loss 곡선, torch).
- honest **spatial split**(맵 왼쪽 학습 / 오른쪽 평가) → confusion·per-class P/R/F1·PCA.
- 두 경로 실데이터 형식 합성맵으로 end-to-end 검증 완료.

## 2. Model "Load refs" 형식 버그 — [x] 무의미해짐
Model이 더 이상 `wavenumber, C1, C2…`(성분=열) CSV를 받지 않음. Map과 동일한
**폴더 로더**(`Reference/*_corrected.csv`, 파일당 1맵)를 사용 → 형식 불일치 자체가 사라짐.
torch 미설치 시 ResNet 선택하면 **명확한 에러 메시지**로 안내.

## 3. 참조 형식 불일치 — [x] 통일됨
Model·Discriminator 둘 다 이제 pest **폴더(다중 맵)** 형식으로 통일.

## 4. 자잘한 폴리시
- [x] 두 도구 appearance 드롭다운 기본 표기 `System` → `Light` 로.
- [x] Map tool 사이드바 맨 위 중복 타이틀(“SERS Discriminator”) 제거 → 헤더바(“SERS map”)
  + view-nav 필만 남김. (창 제목/독스트링은 UI가 아니라 유지)
- [x] Mixture tool 사이드바 옛 타이틀 흔적 확인 — 없음. 헤더바(“Mixture · detect
  components + ratio”) + 컨트롤바뿐, 정리 불필요.

## 5. worktree 마무리 (세션 종료 후 수동)
```powershell
Remove-Item -Recurse -Force ".claude\worktrees\mixture-pest-modular-gui-3a9dee"
git worktree prune
git branch -d claude/mixture-pest-modular-gui-3a9dee   # (선택) 머지된 브랜치 삭제
```
- 참고: `.claude/worktrees/code-file-structure-39f50f` 는 이 작업과 무관한 별도 worktree.

---

## 참고 — 실데이터 관찰 (이번 세션 결론)
- **단독 4클래스**: spatial split **100%** (누수 아님 — 견고, 다른 스팟에서도 분리됨).
- **혼합 검출**: THI 응답계수 **~13×** 로 소수성분(DQ/TBZ) 묻힘 →
  **픽셀투표**로 F1 **0.73 → 0.92**, 조합정확 20% → 80%.
- **Quantify(검량)**: 합성 등온선에서 K 정확 복원. **실검량은** 100 µM 넓은 비율/농도
  + **단일성분 희석 시리즈** 데이터가 생기면 바로 가능.
- **논문 방향(XAI × 경쟁흡착)**: 응답계수 13× · 경쟁 flip · 조합 confusion의 THI 뭉침 ·
  픽셀투표 복구 · marker 밴드(THI 1366 등) — 근거 이미 앱에 있음.
- **chemical space(미학습 분자)** 는 향후(농약 패밀리 확장 시).
