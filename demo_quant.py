"""
demo_quant.py
=============
Concentration-ratio recovery under competitive adsorption -- and an HONEST
map of when it works and when the physics forbids it.

Two regimes:
  PART 1  unsaturated (dilute) surface  -> ratio recovery is accurate.
          This proves the gain/competition cancellation works.
  PART 2  one compound SATURATES the surface (your hard case) -> the
          dominant compound's concentration information is physically lost
          (its SERS signal plateaus), so its ratio is only a lower bound.
          The buried minors' MUTUAL ratio is still recoverable while their
          peaks clear the noise; below that they are flagged <LOQ.

The saturation indicator is total coverage  S/(1+S) = sum(theta).
Rule of thumb: keep sum(theta) < ~0.5 (dilute the sample) to quantify.
"""
import numpy as np
from synthetic import make_components, measure_pure
from sers_mixture import preprocess
from competitive import (
    forward_spectrum, calibrate_response, recover_ratios, coverages,
)

axis = np.linspace(400, 1800, 500)
n, names = 3, ["A", "B", "C"]
A_true = np.array([1.0, 1.2, 0.9])       # intrinsic SERS brightness


def build(K_true, seed=1):
    rng = np.random.default_rng(seed)
    profiles, _ = make_components(n, axis, rng)
    P = preprocess(np.array([measure_pure(profiles[i], axis, rng,
                    noise=0.0, baseline=0.0) for i in range(n)]))
    return P, rng


def synth(C, K, P, rng, noise=0.03, gain=7.3):
    y = forward_spectrum(C, K, A_true, P, gain=gain)
    y = y + rng.normal(0, noise * y.max(), y.shape)
    return preprocess(np.clip(y, 0, None)[None, :])[0]


def show(tag, Ctrue, K, P, rng, R):
    Y = synth(Ctrue, K, P, rng)
    res = recover_ratios(Y, P, R, names=names, quant_floor=0.02)
    tr = Ctrue / Ctrue.sum()
    th = coverages(Ctrue, K)
    print(f"  {tag}")
    print(f"    surface coverage sum(theta) = {th.sum():.3f}  "
          f"({'SATURATED' if th.sum() > 0.6 else 'ok'})")
    print(f"    true  ratio : {dict(zip(names, np.round(tr,3)))}")
    print(f"    recover     : {res['pretty']}")


# ======================================================================
# PART 1 -- comparable affinities, dilute surface: recovery is accurate
# ======================================================================
print("=" * 64)
print("PART 1  unsaturated surface  ->  ratios recover accurately")
print("=" * 64)
K1 = np.array([3.0, 1.0, 2.0])
P, rng = build(K1)
C_cal = np.array([0.05, 0.05, 0.05])
R = calibrate_response(synth(C_cal, K1, P, rng, gain=4.1), C_cal, P)
show("A:B:C = 5:3:2", np.array([0.10, 0.06, 0.04]), K1, P, rng, R)
show("A:B:C = 6:1:3", np.array([0.12, 0.02, 0.06]), K1, P, rng, R)

# ======================================================================
# PART 2 -- two failure modes when A dominates
#   (a) SATURATION  : A's own concentration becomes unrecoverable
#   (b) NOISE FLOOR : a minor whose K*C is tiny vs A sinks under the noise
# ======================================================================
print("=" * 64)
print("PART 2  when A dominates: two distinct walls")
print("=" * 64)

# --- (a) saturation: dominance is only in AFFINITY, minors same concentration
K2 = np.array([50.0, 1.0, 1.0])
P, rng = build(K2)
C_cal = np.array([0.02, 0.02, 0.02])
R = calibrate_response(synth(C_cal, K2, P, rng, gain=4.1), C_cal, P)
print("(a) SATURATION -- dilute the sample to escape it (A:B:C = 1:1:1):")
for f, tag in [(2.0, "concentrated"), (0.02, "diluted 100x")]:
    C = np.array([1.0, 1.0, 1.0]) * f * 0.02
    Y = synth(C, K2, P, rng)
    res = recover_ratios(Y, P, R, names=names)
    th = coverages(C, K2).sum()
    r = res['ratio']
    print(f"    {tag:13s} sum(theta)={th:5.3f} -> "
          f"A/B/C = {r[0]*100:4.1f}/{r[1]*100:4.1f}/{r[2]*100:4.1f} %  "
          f"(true 33/33/33)")

# --- (b) noise floor: how faint a minor can be and still be quantified
print("(b) NOISE FLOOR -- averaging M spectra lifts a buried minor:")
print("    true A:B = 1 : x  (equal affinity), C fixed dilute, noise=3%")
K3 = np.array([1.0, 1.0, 1.0])
P, rng = build(K3, seed=2)
C_cal = np.array([0.05, 0.05, 0.05])
R = calibrate_response(synth(C_cal, K3, P, rng, gain=4.1), C_cal, P)
Cfix = np.array([0.20, 0.01, 0.005])          # B is 1/20 of A, C is 1/40
true = Cfix / Cfix.sum()
for M in [1, 10, 100]:
    # average M noisy replicate spectra -> noise ~ /sqrt(M)
    reps = np.array([synth(Cfix, K3, P, rng, noise=0.03) for _ in range(M)])
    Yavg = reps.mean(0)
    res = recover_ratios(Yavg, P, R, names=names, noise_frac=0.03/np.sqrt(M))
    r = res['ratio']
    print(f"    M={M:>3} | A/B/C = {r[0]*100:4.1f}/{r[1]*100:4.1f}/{r[2]*100:4.1f}"
          f"  (true {true[0]*100:.0f}/{true[1]*100:.0f}/{true[2]*100:.0f})"
          f"  B<LOQ={bool(res['below_LOQ'][1])}")

print("=" * 64)
print("Takeaways:")
print(" - Method is exact in the dilute, above-noise regime (Part 1).")
print(" - SATURATION (sum theta -> 1): dilute the sample to quantify the")
print("   dominant; otherwise its ratio is only a lower bound.")
print(" - NOISE FLOOR: a minor with tiny K*C needs SNR (averaging, better")
print("   substrate) or a clean marker band; if K_A*C_A >> K_minor*C_minor")
print("   by ~1000x, it is physically below detection -> report <LOQ, and")
print("   fix it experimentally (standard addition, mask/deplete A).")
print("=" * 64)
