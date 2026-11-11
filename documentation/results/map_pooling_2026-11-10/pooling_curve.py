"""맵핑의 효과 — 한 맵에서 픽셀을 몇 개 모아 읽느냐에 따라 조성 오차가 어떻게 변하는가 (2026-11-10).

논문 모델(PLS 8 LV)과 같은 학습 집합·전처리. held-out 64 조건(조건 하나씩 빼기 52 + 시험 12).
각 held-out 맵에서 무작위 픽셀(최대 400개 풀)을 n 개 뽑아 픽셀별 PLS 출력을 평균 → 3성분 몫.
n = 1 은 "한 점만 찍었을 때", n 이 커질수록 "맵으로 종합했을 때".

  실행:  .venv/bin/python3 -u documentation/results/map_pooling_2026-11-10/pooling_curve.py
"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(f"{HERE}/../../..")
sys.path.insert(0, f"{REPO}/documentation/scripts")
import train_paper_pls as tp
import dl_model
from sklearn.cross_decomposition import PLSRegression
from real_data import load_map

POOL, DRAWS, NS = 400, 300, (1, 3, 10, 30, 100, 300)
cap = {"order": [], "kw": {}}


class Captured(Exception):
    pass


_ms, _fp = dl_model._map_spectra, dl_model._fit_predict


def ms(cube, mask, *a, **k):
    cap["mask"] = mask
    if k.get("map_id") is not None:
        cap["order"].append(k["map_id"]); cap["kw"][k["map_id"]] = k.get("baseline_correct", True)
    return _ms(cube, mask, *a, **k)


def fp(method, Xtr, Ytr, Xte, **k):          # 첫 호출 = 시험 맵 채점 → 학습 행렬을 꺼내고 멈춘다
    cap.update(X=np.asarray(Xtr), Y=np.asarray(Ytr), paths=np.asarray(k["train_maps"], object), Xte=np.asarray(Xte))
    raise Captured


dl_model._map_spectra, dl_model._fit_predict = ms, fp
try:
    tp.main()
except Captured:
    pass
dl_model._map_spectra, dl_model._fit_predict = _ms, _fp

X, Y, P, MASK = cap["X"], cap["Y"], cap["paths"], cap["mask"]
train_set = set(P.tolist())
test_maps = [p for p in dict.fromkeys(cap["order"]) if p not in train_set]
conc = {p: tp.conc_of(p) for p in list(train_set) + test_maps}
cond = np.array([str(conc[p]) if conc[p] else p for p in P], object)
is_grid = lambda p: conc[p] is not None and all(v in tp.GRID for v in conc[p])
held = [(p, "loo") for p in dict.fromkeys(P.tolist()) if is_grid(p)] + [(p, "test") for p in test_maps if is_grid(p)]
print(f"학습 행렬 {X.shape} · held-out 격자 맵 {len(held)} (loo {sum(s == 'loo' for _, s in held)} · test {sum(s == 'test' for _, s in held)})", flush=True)

full = PLSRegression(n_components=tp.N_COMPONENTS).fit(X, Y)
rng = np.random.default_rng(0)
T, PX, REP = [], [], []
xte_rows = {p: i for i, p in enumerate(test_maps)}
for k, (p, split) in enumerate(held):
    c = np.array(conc[p], float); T.append(c / c.sum())
    if split == "loo":
        ck = str(conc[p]); tr = cond != ck
        mdl = PLSRegression(n_components=tp.N_COMPONENTS).fit(X[tr], Y[tr])
        rep = mdl.predict(X[P == p])
    else:
        mdl = full
        i = xte_rows[p]; rep = mdl.predict(cap["Xte"][100 * i:100 * (i + 1)])
    _w, cube, _m, _c = load_map(p)
    cube = np.asarray(cube, float)
    idx = rng.choice(len(cube), min(POOL, len(cube)), replace=False)
    specs = _ms(cube[idx], MASK, len(idx), baseline_correct=cap["kw"].get(p, True), sampling="legacy")
    F = np.array([dl_model._composition_features(y) for y in specs], np.float32)
    PX.append(mdl.predict(F)); REP.append(rep.mean(0))
    print(f"  {k + 1}/{len(held)} {os.path.basename(p)} · 픽셀 {len(cube)} → 풀 {len(F)}", flush=True)
T = np.array(T); REP = np.array(REP)


def share(raw):
    q = np.clip(np.asarray(raw, float)[..., :3], 0, None); s = q.sum(-1, keepdims=True)
    return np.where(s > 1e-9, q / np.clip(s, 1e-9, None), 1 / 3)


def err(q, t):
    return 50 * np.abs(q - t).sum(-1)


OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


srt = np.sort(T, 1); uniq = srt[:, 2] > srt[:, 1]
say(f"맵핑의 효과 — held-out {len(held)} 조건, 무작위 픽셀 n 개를 모아 읽은 조성 (각 맵 {DRAWS}회 뽑기, 풀 최대 {POOL}픽셀)")
say(f"  대조: 1/3씩 {err(np.full_like(T, 1 / 3), T).mean():.1f} %p · 모델 기록(대표 100픽셀) {err(share(REP), T).mean():.1f} %p")
say("  n픽셀 | 조성 오차 평균 %p | 맵 안 뽑기별 산포(표준편차, 맵 평균) %p | 오차 20 %p 넘는 판독 % | 몫 2배 이내(조건) % | 최다 성분 맞힘 %")
res = {}
for n in NS + ("all",):
    E, D, F2 = [], [], []
    for i, px in enumerate(PX):
        m = len(px) if n == "all" else min(n, len(px))
        if n == "all":
            q = share(px.mean(0))[None]
        else:
            q = share(np.array([px[rng.choice(len(px), m, replace=False)].mean(0) for _ in range(DRAWS)]))
        E.append(err(q, T[i])); D.append(q.argmax(1) == T[i].argmax())
        F2.append(1 / np.minimum(q / T[i], T[i] / np.clip(q, 1e-9, None)).mean(1) <= 2)
    Em = np.array([e.mean() for e in E]); Es = np.array([e.std() for e in E])
    big = 100 * np.mean([np.mean(e > 20) for e in E])
    dom = 100 * np.mean([d.mean() for d, u in zip(D, uniq) if u])
    f2 = 100 * np.mean([f.mean() for f in F2])
    res[str(n)] = (Em.mean(), Es.mean(), big, f2, dom)
    say(f"  {str(n):>5s} | {Em.mean():5.1f} | {Es.mean():5.1f} | {big:5.1f} | {f2:5.1f} | {dom:5.1f}")
one = np.array([err(share(px), T[i]) for i, px in enumerate(PX)], object)
within = np.mean([np.var(share(px), axis=0).sum() for px in PX])
say(f"  픽셀 하나의 판독이 맵 안에서 흩어지는 정도(몫 분산 합의 맵 평균) {within:.4f} → 표준편차 {100 * np.sqrt(within):.1f} %p")
open(f"{HERE}/pooling_curve.txt", "w", encoding="utf-8").write("\n".join(OUT) + "\n")
np.savez(f"{HERE}/pooling_curve.npz", T=T, rep=REP, names=np.array([os.path.basename(p) for p, _ in held]),
         split=np.array([s for _, s in held]), **{f"px{i}": px for i, px in enumerate(PX)},
         **{f"n{k}": np.array(v) for k, v in res.items()})
