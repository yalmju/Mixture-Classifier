"""시뮬레이션 맵을 농도 식 맞춤에 더하면 눌림이 줄어드나 (2026-11-10).

시뮬레이터 = 사전학습에 쓰던 것 그대로(`dl_quantify.simulate_mixtures` 의 물리: 단독 희석계열로 맞춘
경쟁 Langmuir, DQ 만 Sips, 잉크 템플릿, 잡음 1.5 %). 3–24 µM 삼원 맵을 만들어 실측과 같은 입력
(세 밴드 세기 + PLS 몫)을 뽑고, 실측 64 격자 조건의 held-out 판독이 어떻게 바뀌는지 본다.

  실행:  .venv/bin/python3 documentation/results/conc_head_2026-11-10/sim_augment.py
"""
import os, sys, warnings, numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(f"{HERE}/../../..")
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/documentation/scripts")
import dl_model, train_paper_pls as tp
from dataset import load_mixture_list, load_mixture_roles, load_preprocess
from real_data import load_map
from io_utils import load_calibration_csv
from calibration import calibrate, fit_sips
from competitive import forward_spectrum, coverages

m = dl_model.load_model(tp.OUT.replace(".dlm", "_bandonly.dlm")); u = m["uM"]; S3 = list(u["subs"])  # 이전 식(세 밴드 + 몫) 모델로 고정 — 이 기록은 그 식에 대한 것
cfg = load_preprocess(tp.PURE)
subs, wn, mask, P, lo, hi = dl_model._refs(tp.PURE, cfg["baseline"], cfg["trim"])
wn_axis = np.asarray(wn, float)[mask]; bands = np.asarray(u["bands_cm"], float)
assert list(subs) == S3, (subs, S3)
base = lambda p: os.path.basename(str(p).replace(chr(92), "/"))

# ---- 실측 64 격자 조건의 입력 (학습 52: head_inputs, 시험 12: 저장 모델 적용) -------------------
z = np.load(f"{HERE}/head_inputs.npz", allow_pickle=True)
feat = {base(n): np.r_[np.log10(np.expm1(f[6:9]) + 1.0), f[3:5]] for n, f in zip(z["names"], z["F"])}
ev = u["loo_eval"]; Xr, Tr, cond, split = [], [], [], []
for p, t, ok in zip(ev["paths"], ev["true_uM"], ev["in_fit_range"]):
    if ok:
        Xr.append(feat[base(p)]); Tr.append(t); cond.append(str(tuple(t))); split.append("loo")
roles = load_mixture_roles(tp.PURE)
for it in load_mixture_list(tp.PURE, filename_truth=True):
    if roles.get(it[0], "train") != "test":
        continue
    p = tp._here(it)[0]; c = tp.conc_of(p)
    if c is None or not all(v in tp.GRID for v in c):
        continue
    w_, cube, _a, _b = load_map(p)
    meta = dl_model.apply_uM_pixels(m, w_, np.asarray(cube, float), return_meta=True)[2]
    Xr.append(np.asarray(u["mu"], float) + np.asarray(meta["feature_z"], float) * np.asarray(u["sd"], float))
    Tr.append(c); cond.append(str(tuple(c))); split.append("test")
Xr = np.array(Xr, float); Tr = np.array(Tr, float); cond = np.array(cond, object); split = np.array(split)

# ---- 시뮬레이터 (학습 코드의 사전학습 블록과 같은 설정) ---------------------------------------
ax_c, nc, dils = load_calibration_csv(tp.CALIB); mc = (ax_c >= lo) & (ax_c <= hi)
dil = [(dils[nc.index(s)][0], np.asarray(dils[nc.index(s)][1])[:, mc]) for s in S3]
# calibrate() 는 성분마다 희석점 수가 다르면 이 기계의 numpy 에서 배열로 못 묶는다 — 같은 계산을 성분별로 한다
from calibration import fit_isotherm
from competitive import fit_B
from types import SimpleNamespace
_K, _A, iso = np.zeros(3), np.zeros(3), np.ones(3)
for i, (cg, sp) in enumerate(dil):
    B_i = np.array([fit_B(y, P)[0][i] for y in sp])
    _A[i], _K[i] = fit_isotherm(np.asarray(cg, float), B_i)
    if S3[i].upper().startswith("DQ"):
        iso[i] = fit_sips(np.asarray(cg, float), B_i)[2]
cal = SimpleNamespace(K=_K, gA=_A)
NU = dl_model._nuisance_refs(tp.PURE, cfg["baseline"], mask)


def sim_map(c_uM, gain, rng, n_px=100, noise=0.015):
    c = np.asarray(c_uM, float) * 1e-6
    y0 = forward_spectrum(c, cal.K, cal.gA, P, gain=gain, m=iso)
    if NU is not None:
        b_sum = float((gain * cal.gA * coverages(c, cal.K, iso)).sum())
        for row in NU:
            y0 = y0 + 10 ** rng.uniform(np.log10(0.05), np.log10(1.5)) * b_sum * row
    px = y0[None] * rng.uniform(0.8, 1.25, (n_px, 1))                 # 픽셀 간 세기 요동
    px = np.clip(px + rng.normal(0, noise * y0.max(), px.shape), 0, None)
    pk = np.clip(np.asarray(dl_model.apply_model_pixels(m, wn_axis, px), float), 0, None)
    R = pk[:, :3] / (pk[:, :3].sum(1, keepdims=True) + 1e-12)
    return dl_model._band_linear_context(px, R, np.zeros(n_px, int), wn_axis, bands)[1][0]


def sim_set(n, sd_dex, seed):
    rng = np.random.default_rng(seed)
    C = 10 ** rng.uniform(np.log10(3), np.log10(24), (n, 3))
    return np.array([sim_map(c, 10 ** rng.normal(0, sd_dex), rng) for c in C]), C


OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


def slope(x, y):
    return np.polyfit(x, y, 1)[0]


say("시뮬레이션 맵을 농도 식에 더하면 — 실측 64 격자 조건 held-out 판독")
say(f"  시뮬레이터: K(1/M) " + " · ".join(f"{s} {k:.3g}" for s, k in zip(S3, cal.K))
    + " | 밝기 " + " · ".join(f"{a:.3g}" for a in cal.gA) + " | Sips 지수 " + " · ".join(f"{v:.2f}" for v in iso))

say("\n[1] 같은 64 조건에서 시뮬레이터가 내는 세기 대 실측")
rng0 = np.random.default_rng(1)
Xs0 = np.array([sim_map(c, 1.0, rng0) for c in Tr])
Ir, Is = 10 ** Xr[:, :3] - 1, 10 ** Xs0[:, :3] - 1
say(f"  세기 합 ~ 총 농도 지수: 실측 {slope(np.log10(Tr.sum(1)), np.log10(Ir.sum(1))):.2f} · 시뮬 {slope(np.log10(Tr.sum(1)), np.log10(Is.sum(1))):.2f}")
for k, s in enumerate(S3):
    say(f"  {s:3s} 밴드 ~ 자기 농도 지수: 실측 {slope(np.log10(Tr[:, k]), Xr[:, k]):+.2f} · 시뮬 {slope(np.log10(Tr[:, k]), Xs0[:, k]):+.2f}"
        f" | 세기 시뮬/실측 중앙 {np.median(Is[:, k] / Ir[:, k]):.2f}배")
sh = lambda X: np.c_[X[:, 3:5], 1 - X[:, 3:5].sum(1)]
tsh = Tr / Tr.sum(1, keepdims=True)
say(f"  PLS 가 읽은 조성 오차: 실측 맵 {50 * np.abs(sh(Xr) - tsh).sum(1).mean():.1f} %p · 시뮬 맵 {50 * np.abs(sh(Xs0) - tsh).sum(1).mean():.1f} %p")
P0 = dl_model._predict_band_linear(u["coef"], Xs0)
f0 = np.maximum(P0 / Tr, Tr / P0)
say(f"  지금 식(실측으로 맞춤)이 시뮬 맵을 읽으면: 2배 이내 {100 * np.mean(f0 <= 2):.1f} % · 중앙 {np.median(f0):.2f}×"
    f" · 판독/참 중앙 {np.median(P0 / Tr):.2f}")


def heldout(Xs=None, Ts=None, w_sim=1.0, sim_only=False):
    """실측 64 조건: 학습 52 는 조건 하나씩 빼기, 시험 12 는 학습 52(+시뮬) 식. 가중 최소제곱."""
    Pd = np.zeros_like(Tr); tr_all = split == "loo"
    def fit(idx):
        if sim_only:
            X, T, w = Xs, Ts, np.ones(len(Xs))
        elif Xs is None:
            X, T, w = Xr[idx], Tr[idx], np.ones(idx.sum())
        else:
            X = np.vstack([Xr[idx], Xs]); T = np.vstack([Tr[idx], Ts])
            w = np.r_[np.ones(idx.sum()), np.full(len(Xs), w_sim * idx.sum() / len(Xs))]
        A = np.c_[X, np.ones(len(X))] * np.sqrt(w)[:, None]
        return np.linalg.lstsq(A, np.log10(T) * np.sqrt(w)[:, None], rcond=None)[0]
    Wt = fit(tr_all)
    for c in dict.fromkeys(cond[tr_all].tolist()):
        te = tr_all & (cond == c)
        Pd[te] = 10 ** (np.c_[Xr[te], np.ones(te.sum())] @ fit(tr_all & (cond != c)))
    te = ~tr_all
    Pd[te] = 10 ** (np.c_[Xr[te], np.ones(te.sum())] @ Wt)
    return Pd


def row(name, Pd):
    f = np.maximum(Pd / Tr, Tr / Pd)
    sl = np.mean([slope(np.log10(Tr[:, k]), np.log10(Pd[:, k])) for k in range(3)])
    hi_ = (Tr.sum(1) >= 45)
    i24 = np.where((Tr == 24).all(1))[0][0]
    say(f"  {name:34s} | {100 * np.mean(f <= 2):5.1f} | {100 * np.mean(f <= 1.5):5.1f} | {np.median(f):.2f}× | {sl:.2f} | "
        f"{np.median(Pd[Tr == 3]):4.1f} → {np.median(Pd[Tr == 24]):4.1f} | {np.median(Pd[hi_].sum(1) / Tr[hi_].sum(1)):.2f} | "
        + "/".join(f"{v:.0f}" for v in Pd[i24]))


say("\n[2] 식을 무엇으로 맞추느냐에 따른 실측 held-out 판독 (64 조건 · 성분 192개)")
say("  맞춤 자료                           | 2배 이내 % | 1.5배 % | 중앙 | 판독 대 참 기울기 | 참 3 → 24 µM 판독 중앙 | 총 ≥45 µM 판독/참 | 24/24/24")
row("실측만 (지금)", heldout())
for label, sd in (("액적 산포 없음", 0.0), ("산포 ×1.38 (실측과 같게)", 0.14)):
    Xs, Ts = sim_set(400, sd, 0)
    row(f"시뮬만 400맵, {label}", heldout(Xs, Ts, sim_only=True))
    for w in (0.25, 1.0, 4.0):
        row(f"실측 + 시뮬(가중 {w:g}), {label}", heldout(Xs, Ts, w_sim=w))
f = np.maximum(8.4 / Tr, Tr / 8.4)
say(f"  대조: 상수 8.4 µM                    | {100 * np.mean(f <= 2):5.1f} | {100 * np.mean(f <= 1.5):5.1f} | {np.median(f):.2f}×")
open(f"{HERE}/sim_augment.txt", "w", encoding="utf-8").write("\n".join(OUT) + "\n")
