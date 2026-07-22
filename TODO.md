# UNMIXR — TODO (open items)

_남긴 날: 2026-07-22. 다음 세션에서 이어서._

## 1. Model 페이지 중복 정리 (결정 필요) ★
Map tool의 **Validation** 이 이미 다중참조 단일성분 분류 + **5-fold CV confusion (0.985)**
+ 노이즈 열화곡선을 **실데이터**로 함. UNMIXR **Discriminator** 페이지도 실데이터
confusion + 혼합검출 전략을 함. → Model 페이지(합성 데모)는 이와 겹침.

- [ ] **(A) Model 제거** → UNMIXR = Quantify / Discriminator 2탭. 가장 단순, 중복 제거. _(검토 중 추천)_
- [ ] **(B) Model 입력만 고침** — Map처럼 여러 참조 파일(파일당 1성분)로 로드. 유지하되 실데이터로.
- [ ] **(C) 합치기** — Model의 혼합(다중라벨) 지표를 Discriminator 페이지로 통합.

## 2. Model "Load refs" 형식 버그
- 현상: 맵 CSV(`X num,20…`)를 넣으면 `could not convert string to float` 실패.
- 원인: Model은 `wavenumber, C1, C2, …`(성분 = 열) 형식만 받음. Map은 파일당 1맵.
- 할일: (A 선택 시 무의미) · (B 선택 시) **다중 파일 로더 + 각 파일 평균스펙트럼 추출 +
  명확한 에러 메시지**.

## 3. 참조 형식 불일치
- Model = 1파일·다중 열 / Map = 다중 파일. 통일 여부 결정(2번과 연결).

## 4. 자잘한 폴리시
- [x] 두 도구 appearance 드롭다운 기본 표기 `System` → `Light` 로.
- [ ] Map tool 사이드바 맨 위 중복 타이틀(“SERS Discriminator”)이 헤더바와 겹침 → 정리.
- [ ] Mixture tool 사이드바에 남은 옛 타이틀 흔적 확인(상단바 이전 후).

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
