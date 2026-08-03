# -*- coding: utf-8 -*-
"""Physics-embedded model: fit competitive-Langmuir parameters (K, gA per compound) to the
REAL mixtures (known µM) by differentiable optimisation, instead of using single-compound
calibration. Then invert B -> composition on unseen ternary. Compared with PLS.

Uses absolute µM = filename_code x 10.  B is the NNLS projection of the (un-normalised,
baseline-removed) mixture mean spectrum, so all magnitudes are self-consistent.
"""
import os, sys, glob
sys.path.insert(0, os.getcwd())
import numpy as np

root = "S:/Google Drive"
acf = next(os.path.join(e.path, "ACF_PEST_DB") for e in os.scandir(root)
           if e.is_dir() and os.path.isdir(os.path.join(e.path, "ACF_PEST_DB")))
pure = os.path.join(acf, "Pure")
d_bin = os.path.join(acf, "Ratio", "Binary"); d_tern = os.path.join(acf, "Ratio", "Tertiary")

from unmix import _templates, _baseline_removed, _l2
from real_data import load_map
from dataset import is_blank
from sers_mixture import als_baseline
from competitive import fit_B
from validate import parse_mixture_label
from dl_quantify import _ratio
from sklearn.cross_decomposition import PLSRegression
import torch

LO, HI = 300.0, 1800.0
names, wn, means = _templates(pure, True, None); m = (wn >= LO) & (wn <= HI)
nb = [names[i] for i in range(len(names)) if not is_blank(names[i])]
THI = nb.index("THI"); nonTHI = [i for i in range(3) if i != THI]
nonbg = [i for i in range(len(names)) if not is_blank(names[i])]
P = _l2(_baseline_removed(means[nonbg][:, m], True))

def mix_B_and_spec(p):
    _w, c, _mn, _c = load_map(p); c = np.asarray(c, float)[:, m]
    w = c.sum(1); w = w / (w.sum() + 1e-12); mean = w @ c
    y = np.clip(mean - als_baseline(mean), 0, None)
    B, _ = fit_B(y, P)                                   # UN-normalised → keeps magnitude
    return B, y / (np.linalg.norm(y) + 1e-12)

def dataset(folder):
    B, X, C = [], [], []
    for p in sorted(glob.glob(os.path.join(folder, "*_corrected.csv"))):
        t = parse_mixture_label(os.path.basename(p), nb)
        if t and len(t) >= 2:
            b, x = mix_B_and_spec(p); B.append(b); X.append(x)
            C.append([t.get(n, 0) * 1e-5 for n in nb])   # code x10µM = code x1e-5 M
    return np.array(B), np.array(X), np.array(C, float)

Bb, Xb, Cb = dataset(d_bin); Bt, Xt, Ct = dataset(d_tern)
Yb = np.array([_ratio(c) for c in Cb]); Yt = np.array([_ratio(c) for c in Ct])
print(f"binary {len(Bb)} · ternary {len(Bt)}\n")

# ---- differentiable competitive-Langmuir fit: B_i = gA_i · K_i C_i /(1+Σ K_j C_j) ----
Ct_b = torch.tensor(Cb, dtype=torch.float64)
Bt_b = torch.tensor(Bb, dtype=torch.float64)
logK = torch.zeros(3, dtype=torch.float64, requires_grad=True)   # K = 1e4 * exp(logK)
loggA = torch.zeros(3, dtype=torch.float64, requires_grad=True)  # gA = scale * exp(loggA)
K0, gA0 = 1e4, float(Bb.max())
opt = torch.optim.Adam([logK, loggA], lr=0.05)
for it in range(3000):
    K = K0 * torch.exp(logK); gA = gA0 * torch.exp(loggA)
    KC = K * Ct_b                                        # (n,3)
    theta = KC / (1.0 + KC.sum(1, keepdim=True))
    Bpred = gA * theta
    loss = ((Bpred - Bt_b) ** 2).mean() / (gA0 ** 2)
    opt.zero_grad(); loss.backward(); opt.step()
Kf = (K0 * torch.exp(logK)).detach().numpy(); gAf = (gA0 * torch.exp(loggA)).detach().numpy()
print("fitted-on-mixtures  K :", np.array2string(Kf, precision=1))
print("fitted-on-mixtures  gA:", np.array2string(gAf, precision=0))

def invert(B):
    """B -> composition ratio via the fitted competitive Langmuir."""
    out = []
    for b in np.atleast_2d(B):
        th = np.clip(b / gAf, 0, 0.999)
        tot = min(th.sum(), 0.999)
        C = th / (Kf * (1 - tot) + 1e-30)
        out.append(_ratio(C))
    return np.array(out)

# ---- compare on unseen ternary ----
pls = PLSRegression(n_components=3).fit(Xb, Yb)
def norm_rows(A): return np.array([_ratio(np.clip(a, 0, None)) for a in A])
P_phys = invert(Bt); P_pls = norm_rows(pls.predict(Xt))

def summ(name, Pm):
    e = np.mean([0.5 * np.abs(Pm[i] - Yt[i]).sum() for i in range(len(Yt))])
    en = np.mean([np.abs(Pm[i][nonTHI] - Yt[i][nonTHI]).sum() for i in range(len(Yt))])
    rec = np.mean([Pm[i][nonTHI].sum() for i in range(len(Yt))])
    det = np.mean([np.mean([(Pm[i][j] > 0.05) for j in nonTHI if Yt[i][j] > 0.05] or [1.0]) for i in range(len(Yt))])
    print(f"  {name:<22} comp_err {e:.1%}  |  buried nonTHI err {en:.2f}  recovered {rec:.2f}  detect {det:.0%}")

print("\nunseen TERNARY:")
summ("physics-embedded (fit)", P_phys)
summ("PLS", P_pls)
print(f"  (true non-THI mass mean {np.mean([Yt[i][nonTHI].sum() for i in range(len(Yt))]):.2f})")
print("\nper mixture true | physics | PLS:")
for i in range(len(Yt)):
    f = lambda v: "/".join(f"{x:.2f}" for x in v)
    print(f"  {f(Yt[i])}   phys {f(P_phys[i])}   pls {f(P_pls[i])}")
