"""E3. 한 맵만으로 — 입력과 식을 바꿔 가며 같은 분할(조건 묶음 5-fold × 20)로 비교. 기준 = 채택 식.
  실행:  .venv/bin/python3 documentation/results/conc_improve_2026-11-11/e3_variants.py
"""
from common import *
from scipy.signal import savgol_filter
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
say = Log(f"{HERE}/e3_variants.txt")
X0 = base_features(); ref = metrics(cv_predict(X0, pls_fit_predict(8)))
say(line("대조: 상수", metrics(cv_predict(X0, const_fit_predict)))); say(line("기준: 스펙트럼 ÷ 2137, PLS 8", ref))
ALL = {}


def run(name, X, fp=None, P=None):
    P = cv_predict(X, fp or pls_fit_predict(8)) if P is None else P
    ALL[name] = P; say(line(name, metrics(P), ref))


def per_map(f):
    return np.array([f(i, sel_of(i)) for i in range(N)])


ink_mean = per_map(lambda i, s: band(PA[i][s], INK).mean())
say("\n── 입력을 바꿈 (식은 PLS 8)")
run("나누지 않음 (잉크 밴드는 스펙트럼 안에 그대로)", base_features(divide=False))
for w in (1, 5, 20):
    run(f"÷2137 + ln 잉크 세기 열(가중 {w})", np.c_[X0, w * np.log(ink_mean)])
run("ALS 안 한 스펙트럼 ÷ 2137", base_features(P=PR))
run("ALS 안 한 스펙트럼, 나누지 않음", base_features(P=PR, divide=False))
mk = np.abs(ax - INK) <= 15
run("잉크 = 2137 ± 15 면적", per_map(lambda i, s: np.log1p(PA[i][s].mean(0)) - np.log1p(np.clip(PA[i][s].mean(0)[mk], 0, None).sum())))
run("잉크 = 평균 스펙트럼의 2137 높이", per_map(lambda i, s: np.log1p(PA[i][s].mean(0)) - np.log1p(band(PA[i][s].mean(0), INK))))
run("픽셀마다 나눈 뒤 평균", per_map(lambda i, s: (np.log1p(PA[i][s]) - np.log1p(band(PA[i][s], INK))[:, None]).mean(0)))
run("픽셀 ln 평균 ÷ ln 잉크 평균 (기하평균)", per_map(lambda i, s: np.log1p(PA[i][s]).mean(0) - np.log1p(band(PA[i][s], INK)).mean()))
allpx = lambda i: np.where(M[i] >= 0)[0]
run("모든 픽셀(분석물 질량 문턱 없음)", np.array([np.log1p(PA[i][allpx(i)].mean(0)) - np.log1p(band(PA[i][allpx(i)], INK).mean()) for i in range(N)]))
run("픽셀 중앙값 스펙트럼", per_map(lambda i, s: np.log1p(np.median(PA[i][s], 0)) - np.log1p(np.median(band(PA[i][s], INK)))))


def trimmed(i, s, lo=0.1, hi=0.9):
    t = PA[i][s].sum(1); q = np.quantile(t, [lo, hi]); k = s[(t >= q[0]) & (t <= q[1])]
    return np.log1p(PA[i][k].mean(0)) - np.log1p(band(PA[i][k], INK).mean())


run("밝기 10–90 % 픽셀만", per_map(trimmed))
run("밝기 위 절반 픽셀만", per_map(lambda i, s: trimmed(i, s, 0.5, 1.0)))
run("제곱근: sqrt(평균 / 잉크)", per_map(lambda i, s: np.sqrt(PA[i][s].mean(0) / band(PA[i][s], INK).mean())))
run("선형: 평균 / 잉크", per_map(lambda i, s: PA[i][s].mean(0) / band(PA[i][s], INK).mean()))
run("범위 500–1800 cm⁻¹ 만", X0[:, ax <= 1800])
run("잉크 밴드 구간(2050–2250) 뺌", X0[:, (ax < 2050) | (ax > 2250)])
run("1차 미분 (SG 15점)", savgol_filter(X0, 15, 2, deriv=1, axis=1))
run("SG 평활 (11점)", savgol_filter(X0, 11, 2, axis=1))
run("4 채널씩 묶음", X0[:, :X0.shape[1] // 4 * 4].reshape(N, -1, 4).mean(2))
run("픽셀 분위 25·50·75 % 스펙트럼 이어 붙임", per_map(lambda i, s: np.concatenate(
    [np.log1p(np.quantile(PA[i][s], q, 0)) - np.log1p(band(PA[i][s], INK).mean()) for q in (0.25, 0.5, 0.75)])))


def slope_spec(i, s):
    k = band(PA[i][s], INK); kc = k - k.mean(); b = (kc[:, None] * (PA[i][s] - PA[i][s].mean(0))).sum(0) / (kc ** 2).sum()
    return np.log1p(np.clip(b, 0, None) * k.mean()) - np.log1p(k.mean())


run("맵 안 픽셀 회귀 기울기 스펙트럼(잉크 대비)", per_map(slope_spec))
tot = per_map(lambda i, s: PA[i][s].mean(0).sum())
run("전체 면적으로 나눔 + ln(면적/잉크) 열", np.c_[per_map(lambda i, s: np.log1p(PA[i][s].mean(0)) - np.log(tot[i] / 1000)), 5 * np.log(tot / ink_mean)])

say("\n── 식을 바꿈 (입력은 채택 식)")
for k in (10, 12, 16):
    run(f"PLS {k}", X0, pls_fit_predict(k))


def pls1(k):
    def f(Xa, Ya, Xb):
        return np.column_stack([pls_fit_predict(k)(Xa, Ya[:, [j]], Xb)[:, 0] for j in range(Ya.shape[1])])
    return f


for k in (6, 8, 10):
    run(f"성분별 따로 PLS {k} (PLS1)", X0, pls1(k))


def ridge(Xa, Ya, Xb):
    mu = Xa.mean(0); return RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xa - mu, Ya).predict(Xb - mu)


run("Ridge (α 는 학습 쪽 안에서 고름)", X0, ridge)


def pcr(k):
    def f(Xa, Ya, Xb):
        p = PCA(k).fit(Xa); A = np.c_[p.transform(Xa), np.ones(len(Xa))]
        return np.c_[p.transform(Xb), np.ones(len(Xb))] @ np.linalg.lstsq(A, Ya, rcond=None)[0]
    return f


for k in (8, 12, 16):
    run(f"주성분 회귀 {k}", X0, pcr(k))


def on_scores(k, make):
    def f(Xa, Ya, Xb):
        p = PLSRegression(n_components=k, scale=False).fit(Xa, Ya); Sa, Sb = p.transform(Xa), p.transform(Xb)
        sd = Sa.std(0) + 1e-12; Sa, Sb = Sa / sd, Sb / sd
        return np.column_stack([make().fit(Sa, Ya[:, j]).predict(Sb) for j in range(Ya.shape[1])])
    return f


run("PLS 8 점수 → 커널 릿지(RBF)", X0, on_scores(8, lambda: KernelRidge(kernel="rbf", alpha=0.1, gamma=0.02)))
run("PLS 8 점수 → SVR(RBF)", X0, on_scores(8, lambda: SVR(C=3.0, epsilon=0.03, gamma=0.02)))


def quad(k):
    def f(Xa, Ya, Xb):
        p = PLSRegression(n_components=k, scale=False).fit(Xa, Ya); Sa, Sb = p.transform(Xa), p.transform(Xb)
        sd = Sa.std(0) + 1e-12; Sa, Sb = Sa / sd, Sb / sd
        A = np.c_[Sa, Sa ** 2, np.ones(len(Sa))]; B = np.c_[Sb, Sb ** 2, np.ones(len(Sb))]
        W = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ Ya); return B @ W
    return f


run("PLS 8 점수 + 제곱항", X0, quad(8))


def bagged(k, nb=25):
    def f(Xa, Ya, Xb):
        rng = np.random.default_rng(len(Xa)); out = 0
        for _ in range(nb):
            i = rng.integers(0, len(Xa), len(Xa)); out = out + pls_fit_predict(k)(Xa[i], Ya[i], Xb)
        return out / nb
    return f


run("PLS 8 배깅(25)", X0, bagged(8))
run("PLS 12 배깅(25)", X0, bagged(12))


def target_tf(k, fwd, inv):
    def f(Xa, Ya, Xb):
        return np.log10(np.clip(inv(pls_fit_predict(k)(Xa, fwd(10 ** Ya), Xb)), 0.5, 500))
    return f


run("목표를 sqrt(µM) 로", X0, target_tf(8, np.sqrt, np.square))
run("목표를 µM 그대로", X0, target_tf(8, lambda v: v, lambda v: v))

# 격자 밖이지만 성분이 모두 3–33 µM 인 맵을 학습에만 더함
extra = np.where(~ing & (T_all.min(1) >= 3) & (T_all.max(1) <= 33))[0]
say(f"\n── 학습에만 더하는 격자 밖 맵(성분 모두 3–33 µM): {[tuple(int(v) for v in T_all[i]) for i in extra]}")
if len(extra):
    Xe = []
    for i in extra:
        s = np.where(M_all[i] >= 0.5)[0]; s = s if len(s) else np.where(M_all[i] >= 0)[0]
        Xe.append(np.log1p(PA_all[i][s].mean(0)) - np.log1p(band(PA_all[i][s], INK).mean()))
    Xe = np.array(Xe); Ye = np.log10(T_all[extra])
    run("격자 밖 3–33 µM 맵을 학습에 더함", X0, lambda Xa, Ya, Xb: pls_fit_predict(8)(np.r_[Xa, Xe], np.r_[Ya, Ye], Xb))

say("\n── 픽셀 단위로 맞추고 맵에서 종합 (픽셀 스펙트럼 ÷ 맵 잉크)")


def pixel_level(k, own_ink=False, agg=np.mean, reps=20):
    F = []
    for i in range(N):
        s = sel_of(i); kk = band(PA[i][s], INK)
        F.append(np.log1p(np.clip(PA[i][s], 0, None)) - (np.log1p(kk)[:, None] if own_ink else np.log1p(kk.mean())))
    out = np.zeros((reps, N, 3))
    for r in range(reps):
        fo = folds(r)
        for q in range(5):
            tr = np.where(fo != q)[0]; te = np.where(fo == q)[0]
            Xa = np.vstack([F[i] for i in tr]); Ya = np.vstack([np.repeat(LT[[i]], len(F[i]), 0) for i in tr])
            p = PLSRegression(n_components=k, scale=False).fit(Xa, Ya)
            for i in te:
                out[r, i] = agg(np.asarray(p.predict(F[i])).reshape(len(F[i]), -1), 0)
    return out


run("픽셀 PLS 8 → 맵 평균", None, P=pixel_level(8))
run("픽셀 PLS 12 → 맵 평균", None, P=pixel_level(12))
run("픽셀 PLS 12 → 맵 중앙값", None, P=pixel_level(12, agg=np.median))
run("픽셀 PLS 12 (픽셀 자기 잉크로 나눔) → 맵 평균", None, P=pixel_level(12, own_ink=True))

say("\n── 두 식의 판독을 평균")
zc = np.load(f"{REPO}/documentation/results/conc_head_2026-11-10/specpls_cv_cache.npz")
key = {tuple(t): i for i, t in enumerate(zc["T"])}; XA = zc["XA"][[key[tuple(t)] for t in T]]
import dl_model
PA_band = cv_predict(XA, lambda Xa, Ya, Xb: np.log10(dl_model._predict_band_linear(dl_model._fit_band_linear(Xa, 10 ** Ya), Xb)))
run("밴드 식(세 밴드 ÷ 2137 + 몫) 단독", None, P=PA_band)
run("채택 식 + 밴드 식 평균", None, P=(ALL["PLS 10"] * 0 + cv_predict(X0, pls_fit_predict(8)) + PA_band) / 2)
np.savez_compressed(f"{HERE}/e3_predictions.npz", names=np.array(list(ALL)), P=np.array([ALL[k] for k in ALL]))
