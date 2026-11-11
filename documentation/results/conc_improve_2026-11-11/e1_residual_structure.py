"""E1. 채택 식(전체 스펙트럼 ÷ 2137, PLS 8)의 held-out 오차가 어떤 모양인가 (64 조건, 조건 묶음 5-fold × 20).
  ① 재현 확인  ② 한 맵 안 세 성분이 같이 틀리는가(공통) / 따로 틀리는가  ③ 참 농도에 따른 눌림이 차지하는 몫
  ④ 공통 오차가 무엇과 같이 움직이는가  ⑤ 다른 성분 농도가 오차에 남기는 흔적
  실행:  .venv/bin/python3 documentation/results/conc_improve_2026-11-11/e1_residual_structure.py
"""
from common import *
say = Log(f"{HERE}/e1_residual_structure.txt")
X = base_features(); P = cv_predict(X, pls_fit_predict(8)); d = metrics(P); dc = metrics(cv_predict(X, const_fit_predict))
say(f"맵 {N} · 조건 {len(np.unique(cond))}")
say(line("대조: 상수", dc)); say(line("채택 식 (스펙트럼 ÷ 2137, PLS 8)", d))
E = (P - LT[None]).mean(0)                     # 맵별 평균 held-out 잔차 (dex), 부호 있음
Er = P - LT[None]
say(f"\n잔차(반복 평균) RMSE {np.sqrt((E**2).mean()):.3f} dex · 한 번 분할 RMSE {np.sqrt((Er**2).mean()):.3f} dex "
    f"(분할에 따라 바뀌는 부분 {np.sqrt(((Er - E[None])**2).mean()):.3f})")
say("\n② 공통 / 성분별")
c = E.mean(1); s = E - c[:, None]
tot = (E ** 2).sum(); say(f"  제곱오차 중 맵 공통(세 성분 평균) {100 * 3 * (c**2).sum() / tot:.1f} % · 성분별 {100 * (s**2).sum() / tot:.1f} %")
cc = np.corrcoef(E.T); say(f"  성분 사이 잔차 상관: DQ–TBZ {cc[0,1]:+.2f} · DQ–THI {cc[0,2]:+.2f} · TBZ–THI {cc[1,2]:+.2f}")
say(f"  공통 오차 표준편차 {c.std():.3f} dex (×{10**c.std():.2f}) · 성분별 표준편차 DQ {s[:,0].std():.3f} TBZ {s[:,1].std():.3f} THI {s[:,2].std():.3f}")
say("\n③ 참 농도에 따른 눌림")
for j in range(3):
    b = np.polyfit(LT[:, j], E[:, j], 1); fit = np.polyval(b, LT[:, j])
    say(f"  {SUB[j]}: 잔차 = {b[0]:+.2f}·log C + … · 제곱오차 중 눌림 몫 {100 * (fit**2).sum() / (E[:, j]**2).sum():.1f} % · "
        f"눌림 뺀 나머지 표준편차 {np.std(E[:, j] - fit):.3f} dex")
fitall = np.column_stack([np.polyval(np.polyfit(LT[:, j], E[:, j], 1), LT[:, j]) for j in range(3)])
say(f"  전체: 눌림 몫 {100 * (fitall**2).sum() / tot:.1f} % · 나머지 RMSE {np.sqrt(((E - fitall)**2).mean()):.3f} dex")
E2 = E - fitall; c2 = E2.mean(1)
say(f"  눌림을 뺀 뒤 공통 몫 {100 * 3 * (c2**2).sum() / (E2**2).sum():.1f} %")
say("\n⑤ 다른 성분 농도의 흔적 (잔차를 세 성분 log 농도로 회귀)")
A = np.c_[LT, np.ones(N)]
for j in range(3):
    w = np.linalg.lstsq(A, E[:, j], rcond=None)[0]; r2 = 1 - ((E[:, j] - A @ w) ** 2).sum() / ((E[:, j] - E[:, j].mean()) ** 2).sum()
    say(f"  {SUB[j]} 잔차 = {w[0]:+.2f}·logDQ {w[1]:+.2f}·logTBZ {w[2]:+.2f}·logTHI  (R² {r2:.2f})")
say("\n④ 공통 오차(눌림 뺀 뒤)가 무엇과 같이 움직이는가 — 상관계수 r (n=64, |r|>0.25 면 p<0.05)")
ink = np.array([band(PA[i][sel_of(i)], INK).mean() for i in range(N)])
inkcv = np.array([band(PA[i][sel_of(i)], INK).std() / band(PA[i][sel_of(i)], INK).mean() for i in range(N)])
tot_i = np.array([PA[i][sel_of(i)].sum(1).mean() for i in range(N)])
totcv = np.array([PA[i][sel_of(i)].sum(1).std() / PA[i][sel_of(i)].sum(1).mean() for i in range(N)])
bg = np.array([(PR[i] - PA[i]).mean() for i in range(N)])
nsel = np.array([len(sel_of(i)) for i in range(N)]); mass = M.mean(1)
mt = z["mtime"][ing]; order = np.argsort(np.argsort(mt))
feats = {"ln 잉크 2137 세기": np.log(ink), "잉크 세기 픽셀 CV": inkcv, "ln 전체 세기": np.log(tot_i), "전체 세기 픽셀 CV": totcv,
         "ln ALS 배경 높이": np.log(bg), "ln 배경/잉크": np.log(bg / ink), "선택 픽셀 수": nsel, "조성 단계 분석물 질량": mass,
         "파일 시각 순서": order, "ln 농도 합(참)": np.log10(T.sum(1))}
for k, v in feats.items():
    say(f"  {k:<22s} 공통 {np.corrcoef(v, c2)[0,1]:+.2f} | DQ {np.corrcoef(v, E2[:,0])[0,1]:+.2f} TBZ {np.corrcoef(v, E2[:,1])[0,1]:+.2f} THI {np.corrcoef(v, E2[:,2])[0,1]:+.2f}")
say("\n가장 크게 틀린 맵 8개 (잔차 dex: DQ TBZ THI)")
for i in np.argsort(-np.abs(E).max(1))[:8]:
    say(f"  {os.path.basename(str(paths[i])):<26s} {E[i,0]:+.2f} {E[i,1]:+.2f} {E[i,2]:+.2f}  잉크 {ink[i]:7.0f}  순서 {order[i]}")
np.savez(f"{HERE}/e1_residuals.npz", E=E, T=T, paths=paths)
