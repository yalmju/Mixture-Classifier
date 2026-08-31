# -*- coding: utf-8 -*-
"""TBZ 폭주의 구조적 해법 테스트 — 맵당 총량 residual 하나 + 조성 분배.

진단: TB24 계열에서 TBZ 밴드가 자체 억제(~6x)돼 T_TBZ = Ccal/ratio 만 일관되게
낮다. 성분별 Δ 3개를 독립 학습하면 눌린 채널의 Δ 외삽이 지수 증폭으로 폭주한다
(배포는 6:24:6, 통합은 12:24:12 — 실행마다 다른 셀). 여기서는 자유도를 줄인다:

    T_hat = median_i(Ccal_i / ratio_i)  (맵당 스칼라)
    Δ_total = net(F) 1차원              →  C_i = ratio_i · T_hat · 10^Δ_total

조성이 정확하다는 관찰(recovery 정상)을 구조로 사용 — 한 성분만 날아갈 수 없다.
비교: 현행 neural (per-component Δ), 그리고 둘의 성분별 결합(hybrid: 총량 경로의
성분값과 현행 경로의 성분값 중 log 중간값? 아니고 단순 기하평균).

실행:  python -u run_total_residual_fix.py
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
MAIN = Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(MAIN / "documentation" / "scripts"))
import run_integrated_reevaluation as R  # noqa: E402


def train_total_residual(F, That, Ttot, seed=0, max_epochs=800):
    """R.train_residual 의 1차원 판 — 목표는 log10(true_total / T_hat)."""
    import torch
    import torch.nn as nn
    F = np.asarray(F, np.float32)
    target = (np.log10(np.clip(Ttot, 0.05, None)) - np.log10(np.clip(That, 0.05, None))).astype(np.float32)
    n = len(F); rng = np.random.default_rng(seed)
    perm = rng.permutation(n); nval = max(4, int(round(n * 0.2)))
    va, tr = perm[:nval], perm[nval:]
    mu = F[tr].mean(0); sd = F[tr].std(0) + 1e-6

    def make():
        torch.manual_seed(int(seed))
        return nn.Sequential(nn.Linear(F.shape[1], 64), nn.BatchNorm1d(64), nn.ReLU(),
                             nn.Dropout(0.25), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1))

    A = torch.tensor((F - mu) / sd); Y = torch.tensor(target).unsqueeze(1)
    net = make(); op = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=3e-3)
    hub = torch.nn.SmoothL1Loss(beta=0.25)
    best, best_ep, stale = float("inf"), 20, 0
    for ep in range(max_epochs):
        net.train(); op.zero_grad(); hub(net(A[tr]), Y[tr]).backward(); op.step()
        net.eval()
        with torch.no_grad():
            val = float(hub(net(A[va]), Y[va]))
        if val < best - 1e-4:
            best, best_ep, stale = val, ep + 1, 0
        else:
            stale += 1
        if stale >= 80 and ep >= 100:
            break
    mu = F.mean(0); sd = F.std(0) + 1e-6
    A = torch.tensor((F - mu) / sd)
    net = make(); op = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=3e-3)
    for _ in range(max(5, best_ep)):
        net.train(); op.zero_grad(); hub(net(A), Y).backward(); op.step()
    net.eval()
    return net, mu, sd, best_ep


def main():
    t0 = time.time()
    z = np.load(R.BUNDLE, allow_pickle=True)
    X, Y = z["X"].astype(np.float32), z["Y"].astype(np.float32)
    names, maps, C = z["names"].astype(object), z["mapkey"].astype(object), z["concentration"].astype(float)
    ab = np.asarray(json.loads(R.OLD_CONC.read_text(encoding="utf-8"))["excel_loglinear_ab"], float)
    import dl_model, torch

    abskey_px = np.array(["c:" + ",".join(f"{v:.8g}" for v in row) for row in C], object)
    cfolds = R.grouped_folds(abskey_px, 5, 701)
    recs = []
    for fi, held in enumerate(cfolds, 1):
        te = np.where(np.array([g in held for g in abskey_px], bool))[0]
        tr = np.where(np.array([g not in held for g in abskey_px], bool))[0]
        pred_te, net = dl_model._fit_torch_bag("mlp", X[tr], Y[tr], maps[tr], X[te],
                                               epochs=120, seed=1200 + fi, return_net=True)
        net.eval(); chunks = []
        with torch.no_grad():
            for st in range(0, len(tr), 256):
                chunks.append(torch.softmax(net(torch.tensor(X[tr][st:st + 256])), 1).numpy())
        ctx_tr = R.calibration_context(X[tr], np.vstack(chunks), names[tr], ab)
        ctx_te = R.calibration_context(X[te], pred_te, names[te], ab)
        truth = {str(n): C[np.where(names == n)[0][0]] for n in dict.fromkeys(names.tolist())}
        Ftr = np.stack([r["feature"] for r in ctx_tr]); Fte = np.stack([r["feature"] for r in ctx_te])
        Rtr = np.stack([r["ratio"] for r in ctx_tr]); Rte = np.stack([r["ratio"] for r in ctx_te])
        Ctr = np.stack([r["Ccal"] for r in ctx_tr]); Cte = np.stack([r["Ccal"] for r in ctx_te])
        Ttr = np.stack([truth[r["map"]] for r in ctx_tr]); Tte = np.stack([truth[r["map"]] for r in ctx_te])
        That_tr = np.median(Ctr / np.clip(Rtr, 1e-6, None), axis=1)
        That_te = np.median(Cte / np.clip(Rte, 1e-6, None), axis=1)
        # 총량 residual
        netT, mu, sd, ep = train_total_residual(Ftr, That_tr, Ttr.sum(1), seed=2200 + fi)
        with torch.no_grad():
            dT = netT(torch.tensor(((Fte - mu) / sd).astype(np.float32))).numpy().ravel()
        total_pred = That_te * 10.0 ** np.clip(dT, -2, 2)
        P_total = Rte * total_pred[:, None]
        # 현행 per-component neural (기준선)
        model, mu2, sd2, _, _ = R.train_residual(Ftr, Ctr, Ttr, seed=2200 + fi, max_epochs=800)
        P_neur, _ = R.predict_residual(model, Fte, mu2, sd2, Cte)
        for i, r in enumerate(ctx_te):
            recs.append({"map": r["map"], "true": Tte[i].tolist(),
                         "total": P_total[i].tolist(), "neural": np.asarray(P_neur)[i].tolist(),
                         "geo": (np.sqrt(np.clip(P_total[i], 1e-3, None) * np.clip(np.asarray(P_neur)[i], 1e-3, None))).tolist()})
        print(f"fold {fi}/5 done ({time.time()-t0:.0f}s, total-eph {ep})", flush=True)

    T = np.array([r["true"] for r in recs])
    kk = (T > 0).sum(1); low = np.all(np.isin(T, [3, 6, 12, 24]), axis=1)
    for v in ("neural", "total", "geo"):
        P = np.array([r[v] for r in recs])
        for g, sel in (("grid64", low), ("binary", kk == 2)):
            t, p = T[sel], P[sel]; ok = t > 0
            e = p[ok] - t[ok]; fold = p[ok] / t[ok]
            worst = np.array([np.max(np.maximum(p[i] / np.maximum(t[i], 1e-9),
                                                t[i] / np.maximum(p[i], 1e-9))[t[i] > 0]) for i in range(len(t))])
            print("%-7s %-7s RMSE %7.2f  MAE %6.2f  w2x %5.1f%%  >3x %2d  worstmax %5.1f"
                  % (v, g, np.sqrt(np.mean(e ** 2)), np.mean(np.abs(e)),
                     np.mean((fold >= 0.5) & (fold <= 2.0)) * 100, np.sum(worst > 3), worst.max()), flush=True)
    # 문제 셀 확인
    print("\nTB24 · DQ=THI 셀들 (TBZ 예측):")
    for mp in ("DQ6-TB24-TH6.csv", "DQ12-TB24-TH12.csv", "DQ3-TB24-TH3.csv", "DQ24-TB24-TH24.csv"):
        for r in recs:
            if r["map"] == mp:
                print(f"  {mp:<22} true TBZ 24 | neural {r['neural'][1]:6.1f}  total {r['total'][1]:6.1f}  geo {r['geo'][1]:6.1f}")
    out = Path(__file__).resolve().parents[1] / "results" / "total_residual_fix_20260831"
    out.mkdir(parents=True, exist_ok=True)
    R.dump_json(out / "records.json", {"records": recs})
    print(f"\ndone {(time.time()-t0)/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
