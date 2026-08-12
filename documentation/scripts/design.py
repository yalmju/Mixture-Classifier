"""How many maps to pin the exclusion exponent q in  R_i/R_THI ~ total^q ?

The exponent is estimated from a gain-free ratio, so its precision is
    SE(q) = sigma / ( SD(log10 total) * sqrt(n) )
sigma comes from the 6 maps already measured; SD(log10 total) is a DESIGN choice.
"""
import numpy as np, math

# ---- measured scatter from the 6 maps --------------------------------------
TOT = np.array([36, 36, 36, 18, 72, 72], float)
R_DQ = np.array([0.160, 0.055, 0.203, 0.590, 0.059, 0.119])
R_TBZ = np.array([1.023, 0.789, 0.470, 0.206, 0.841, 0.630])
x = np.log10(TOT)

print("=== current estimate, from the 6 maps we have ===")
sig = {}
for lbl, y in (("DQ", np.log10(R_DQ)), ("TBZ", np.log10(R_TBZ))):
    sl, ic = np.polyfit(x, y, 1)
    resid = y - (sl * x + ic)
    s = resid.std(ddof=2)
    se = s / (x.std() * math.sqrt(len(x)))
    sig[lbl] = s
    print(f"  R_{lbl}/R_THI ~ total^({sl:+.2f} +- {se:.2f})   residual sigma = {s:.3f} dex ({10**s:.2f}x)")
print(f"  design so far: totals {sorted(set(TOT.astype(int)))} -> SD(log10 total) = {x.std():.3f} dex  (narrow)")

# ---- what precision do we need? --------------------------------------------
# the exclusion correction spans q * (range of log10 total). Over the 3-48 uM
# grid the totals run 9..144 uM = 1.2 dex, so an error SE in q costs 1.2*SE dex
# of composition error. Target: keep it under 0.1 dex (1.26x).
SPAN = math.log10(144 / 9)
TARGET = 0.10 / SPAN
print(f"\n=== target ===\n  grid totals 9..144 uM = {SPAN:.2f} dex span")
print(f"  to keep the exclusion correction accurate to 0.10 dex (1.26x) -> SE(q) <= {TARGET:.3f}")

# ---- n vs design spread -----------------------------------------------------
print("\n=== maps needed  n = (sigma / (SD_x * SE_target))^2 ===")
designs = [
    ("현재와 같은 좁은 범위 (총 18-72)", 0.207),
    ("총 18-144 (2 levels, half/half)", 0.30),
    ("총 9-144 균등 (5 levels)", 0.40),
    ("총 9-144, 양끝에 몰아서", 0.60),
]
print(f"{'design':>36} {'SD_x':>6} {'n (DQ)':>8} {'n (TBZ)':>9}")
for lbl, sd in designs:
    ns = {k: (sig[k] / (sd * TARGET)) ** 2 for k in sig}
    print(f"{lbl:>36} {sd:6.2f} {math.ceil(ns['DQ']):8d} {math.ceil(ns['TBZ']):9d}")

# ---- simulation check -------------------------------------------------------
print("\n=== simulation check (does the formula hold?) ===")
rng = np.random.default_rng(0)
LEVELS = np.array([3, 6, 12, 24, 48], float)


def sim(n, q_true=-1.25, sigma=None, trials=600):
    sigma = sigma or sig["DQ"]
    out = []
    for _ in range(trials):
        idx = rng.integers(0, 5, size=(n, 3))
        C = LEVELS[idx]
        t = np.log10(C.sum(axis=1))
        if t.std() < 1e-6:
            continue
        y = q_true * t + rng.normal(0, sigma, n)
        out.append(np.polyfit(t, y, 1)[0])
    return np.std(out)


for n in (12, 20, 25, 30, 40, 60, 125):
    se = sim(n)
    err = se * SPAN
    print(f"  n={n:4d} 무작위 조합 -> SE(q)={se:.3f}   조성 보정 오차 {10**err:.2f}x")

print("\n=== 반복은 여기서 도움이 안 된다 (비는 게인이 상쇄되므로) ===")
print("  게인 산포 1.48x -> 절대 uM에만 영향. 배제항 추정에는 서로 다른 조성이 필요.")
