"""
demo_compare.py
===============
Explain competitive adsorption from measured mixtures:
titrate analyte A (B, C held fixed), simulate WITH competition, then show
 - A's band SATURATES (Langmuir fits, linear/additive fails),
 - the fixed partners B,C get DISPLACED as A rises,
 - the recovered affinity K_A matches the truth.
"""
import numpy as np
from synthetic import make_components
from sers_mixture import preprocess
from competitive import forward_spectrum, coverages
from competitive_compare import additive_residual, fit_titration, displacement

rng = np.random.default_rng(5)
axis = np.linspace(400, 1800, 500)
n = 3
prof, _ = make_components(n, axis, rng)
P = preprocess(prof)
K_true = np.array([0.08, 0.01, 0.01])     # A binds 8x stronger
A_bright = np.array([1.0, 1.1, 0.9])

C_A = np.array([2, 5, 10, 20, 50, 100, 200, 500.0])
C_fixed = np.array([20.0, 20.0])          # B, C held constant

BA, BB, BC, resids = [], [], [], []
for ca in C_A:
    C = np.array([ca, C_fixed[0], C_fixed[1]])
    y = forward_spectrum(C, K_true, A_bright, P, gain=6.0)
    y = preprocess(np.clip(y + rng.normal(0, 0.02*y.max(), y.shape), 0, None)[None])[0]
    B, _, res = additive_residual(y, P)
    BA.append(B[0]); BB.append(B[1]); BC.append(B[2]); resids.append(res)

BA, BB, BC = map(np.array, (BA, BB, BC))

print("=" * 60)
print("Competitive adsorption from a titration of A (B,C fixed)")
print("=" * 60)
fit = fit_titration(C_A, BA)
print(f"A band vs [A]:  linear R²={fit['r2_linear']:.3f}   "
      f"Langmuir R²={fit['r2_langmuir']:.3f}")
print(f"  -> additive(linear) {'FAILS' if fit['r2_linear']<0.9 else 'ok'}, "
      f"competition(Langmuir) {'fits' if fit['r2_langmuir']>0.95 else 'partial'}")
print(f"  recovered K_A = {fit['K']:.3f}   (true {K_true[0]:.3f})")
print("-" * 60)
sB, _ = displacement(C_A, BB)
sC, _ = displacement(C_A, BC)
print(f"fixed partner B band slope vs [A]: {sB:+.2e}  "
      f"({'DISPLACED' if sB<0 else 'no'})")
print(f"fixed partner C band slope vs [A]: {sC:+.2e}  "
      f"({'DISPLACED' if sC<0 else 'no'})")
print("-" * 60)
print(f"mean shape residual (NNLS of pures): {np.mean(resids):.3f}")
print("  (small -> intensity competition dominates; large -> also chemical/")
print("   orientation non-additivity to model)")
print("=" * 60)
print("Read-out: A saturates + B,C drop as A rises = textbook competitive")
print("adsorption, quantified straight from your measured mixtures.")
print("=" * 60)
