"""저장된 논문 모델 세 판의 농도 판독 비교 + "정답 = 뿌린 조성" 기준 점검 (2026-11-10, 11-11 갱신).

  (채택)      pls_composition_261110.dlm            전체 스펙트럼 ÷ 2137 PLS 회귀(8)   ← 2026-11-11 논문 모델로 확정
  _bandink    pls_composition_261110_bandink.dlm    세 밴드 ÷ 2137 + 몫 (11-10 식)
  _bandonly   pls_composition_261110_bandonly.dlm   세 밴드 + 몫 (잉크 밴드로 나누지 않음)

[1] 64 격자 조건: 학습 52맵은 모델에 저장된 조건 하나씩 빼기 기록, 시험 12맵은 저장 모델을 그대로 적용
    (스펙트럼은 학습과 같은 전처리 — dl_model._map_spectra: 포화 보정 → 맵당 100픽셀 → ALS 기준선 제거).
    농도는 성분별로만 센다(총 농도로 묶은 수치·두 장 판독은 11-11 사용자 결정으로 뺌).
[2] 조성: 참 조성이 같은 두 맵(몫은 같고 농도만 절반인 격자 조건)의 PLS 판독이 서로 얼마나 다른가 — 오차가 맵마다
    무작위인지, 조성에 따라 정해진 어긋남인지.  (results/fig4_paper_pls_2026-11-10/heldout_maps.csv)

  실행:  .venv/bin/python3 documentation/results/conc_head_2026-11-10/model_variants.py
"""
import os, sys, csv, warnings, numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(f"{HERE}/../../..")
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/documentation/scripts")
sys.argv = sys.argv[:1]
import dl_model, train_paper_pls as tp
from dataset import load_mixture_list, load_mixture_roles
from real_data import load_map
OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


roles = load_mixture_roles(tp.PURE)
test = [tp._here(it)[0] for it in load_mixture_list(tp.PURE, filename_truth=True) if roles.get(it[0], "train") == "test"]
test = [p for p in test if tp.conc_of(p) and all(v in tp.GRID for v in tp.conc_of(p))]
cubes = {}
for p in test:
    wn, cube, _a, _b = load_map(p); wn = np.asarray(wn, float)
    cubes[p] = (wn, np.asarray(cube, float).reshape(-1, len(wn)))


def like_training(m, wn, X, map_id):
    mask = (wn >= m["lo"]) & (wn <= m["hi"])
    return np.array(dl_model._map_spectra(X, mask, int(m["px_per_map"]), baseline_correct=not m.get("nnls_screen"),
                                          sampling=m["pixel_sampling"], sampling_seed=int(m["sampling_seed"]), map_id=map_id))


def reads(path):
    m = dl_model.load_model(path); u = m["uM"]; ev = u["loo_eval"]; R = {}
    for t, p, pa, ok in zip(ev["true_uM"], ev["pred_uM"], ev["paths"], ev["in_fit_range"]):
        if ok:
            R[tuple(int(v) for v in t)] = (np.asarray(p, float), "loo")
    for p in test:
        wn, X = cubes[p]
        um, subs, meta = dl_model.apply_uM_pixels(m, wn, like_training(m, wn, X, p), return_meta=True)
        R[tp.conc_of(p)] = (np.array([meta["map_uM"][s] for s in subs]), "test")
    return R


def line(name, P, T, extra=""):
    r = np.abs(np.log10(P) - np.log10(T))
    say(f"  {name:34s} | {100 * np.mean(r <= np.log10(2) + 1e-9):5.1f} | {100 * np.mean(r <= np.log10(1.5)):5.1f} | "
        f"{10 ** np.median(r):.2f}× | {np.sqrt(np.mean(r ** 2)):.3f}" + extra)


D = tp.OUT_DIR
M = {"세 밴드 + 몫": f"{D}/pls_composition_261110_bandonly.dlm", "11-10 식: 세 밴드 ÷ 2137 + 몫": f"{D}/pls_composition_261110_bandink.dlm",
     "채택: 전체 스펙트럼 ÷ 2137 PLS": f"{D}/pls_composition_261110.dlm"}
ALL = {k: reads(v) for k, v in M.items()}
conds = sorted(next(iter(ALL.values())))
T = np.array(conds, float); split = np.array([next(iter(ALL.values()))[c][1] for c in conds])
gm = 10 ** np.log10(T[split == "loo"]).mean()
say(f"[1] 64 격자 조건 한 장 판독 ({(split == 'loo').sum()} 조건 빼기 + 시험 {(split == 'test').sum()}맵)")
say("    (2배 이내 % | 1.5배 % | 중앙 배수 | RMSE dex | 기울기 | 시험 12맵 2배 % | DQ·TBZ·THI 2배 % | 참 3·6·12·24 µM 성분 2배 % | 24/24/24 µM)")
line("대조: 상수 %.1f µM" % gm, np.full_like(T, gm), T)
for k, R in ALL.items():
    P = np.array([R[c][0] for c in conds]); r = np.abs(np.log10(P) - np.log10(T)); ok = r <= np.log10(2)
    e = R[(24, 24, 24)][0]
    line(k, P, T, f" | {np.polyfit(np.log10(T).ravel(), np.log10(P).ravel(), 1)[0]:.2f} | {100 * ok[split == 'test'].mean():5.1f} | "
         + "·".join(f"{100 * ok[:, j].mean():.0f}" for j in range(3)) + " | "
         + "·".join(f"{100 * ok[T == v].mean():.0f}" for v in (3, 6, 12, 24)) + f" | {e[0]:.0f}/{e[1]:.0f}/{e[2]:.0f}")

pairs = [(c, tuple(v // 2 for v in c)) for c in conds if all(v % 2 == 0 and v // 2 in tp.GRID for v in c)]

say("\n[2] 조성 — 참 조성이 같은 두 맵(농도만 절반인 격자 조건, 27 짝)의 PLS 판독")
rows = [r for r in csv.DictReader(open(f"{REPO}/documentation/results/fig4_paper_pls_2026-11-10/heldout_maps.csv", encoding="utf-8")) if r["grid64"] == "1"]
S = {tuple(int(float(r[f"{s}_true_uM"])) for s in ("DQ", "TBZ", "THI")): np.array([float(r[f"{s}_PLS"]) for s in ("DQ", "TBZ", "THI")]) for r in rows}
for c in S:
    S[c] = S[c] / S[c].sum()
x = np.array([np.array(c, float) / sum(c) for c, _ in pairs]); A = np.array([S[c] for c, _ in pairs]); B = np.array([S[h] for _, h in pairs])
eA, eB = A - x, B - x
mx = lambda E: 100 * np.abs(E).max(1).mean()
say(f"  뿌린 조성 대비 오차(조건별 최대 몫 오차의 평균): 높은 쪽 {mx(eA):.1f} %p · 절반 농도 쪽 {mx(eB):.1f} %p")
say(f"  두 맵끼리의 차이: {mx(A - B):.1f} %p   (오차가 서로 무관하면 √(a²+b²) 꼴 ≈ {np.hypot(mx(eA), mx(eB)):.1f}, 같은 어긋남이면 0)")
say(f"  두 맵 오차의 상관 r = {np.corrcoef(eA.ravel(), eB.ravel())[0, 1]:+.2f}  (성분별 DQ {np.corrcoef(eA[:, 0], eB[:, 0])[0, 1]:+.2f} · "
    f"TBZ {np.corrcoef(eA[:, 1], eB[:, 1])[0, 1]:+.2f} · THI {np.corrcoef(eA[:, 2], eB[:, 2])[0, 1]:+.2f})")
say(f"  두 맵 평균의 오차: {mx((A + B) / 2 - x):.1f} %p")
say(f"  평균 오차(편향, %p): 높은 쪽 " + " / ".join(f"{v:+.1f}" for v in 100 * eA.mean(0)) + " · 절반 농도 쪽 " + " / ".join(f"{v:+.1f}" for v in 100 * eB.mean(0)))
open(f"{HERE}/model_variants.txt", "w", encoding="utf-8").write("\n".join(OUT) + "\n")
