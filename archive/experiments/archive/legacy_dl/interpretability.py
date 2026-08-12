# -*- coding: utf-8 -*-
"""Interpretability of the full-spectrum MLP composition model:
  (1) Integrated Gradients — per-wavenumber attribution for each compound (SHAP-style,
      pure torch, no shap dependency);
  (2) band permutation importance — accuracy drop when a spectral window is shuffled;
  (3) ligand ablation — zero a compound's marker bands → its predicted fraction should
      collapse (causal reliance on the right features).
Trains the model once on all mixtures. Saves docs/dl_interpretability.png.
"""
import os, sys, glob
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = "S:/Google Drive"
acf = next(os.path.join(e.path, "ACF_PEST_DB") for e in os.scandir(root)
           if e.is_dir() and os.path.isdir(os.path.join(e.path, "ACF_PEST_DB")))
pure = os.path.join(acf, "Pure")
folders = [os.path.join(acf, "Ratio", "Binary"), os.path.join(acf, "Ratio", "Tertiary")]
from unmix import _templates, _baseline_removed, _l2, compute_vip_bands
from real_data import load_map
from dataset import is_blank
from sers_mixture import als_baseline
from validate import parse_mixture_label
from dl_quantify import simulate_mixtures, train_composition, _spec_net, _ratio
from calibration import calibrate
from io_utils import load_calibration_csv
import torch

LO, HI = 300.0, 1800.0
names, wn, means = _templates(pure, True, None); m = (wn >= LO) & (wn <= HI)
wnv = wn[m]
nb = [names[i] for i in range(len(names)) if not is_blank(names[i])]
nonbg = [i for i in range(len(names)) if not is_blank(names[i])]
P = _l2(_baseline_removed(means[nonbg][:, m], True))
ax_c, names_c, dils = load_calibration_csv(os.path.join(acf, "Ratio", "results", "calibration_spectra.csv"))
mc = (ax_c >= LO) & (ax_c <= HI)
calib = calibrate([(dils[names_c.index(n)][0], np.asarray(dils[names_c.index(n)][1])[:, mc]) for n in nb], P, nb)
vip = compute_vip_bands(pure, baseline=True, trim=(LO, HI))[1]        # {name:[wavenumbers]}

def mix_spec(p):
    _w, c, _mn, _c = load_map(p); c = np.asarray(c, float)[:, m]
    w = c.sum(1); w = w/(w.sum()+1e-12); mean = w@c
    y = np.clip(mean-als_baseline(mean), 0, None); return y/(np.linalg.norm(y)+1e-12)
X, Y = [], []
for folder in folders:
    for p in sorted(glob.glob(os.path.join(folder, "*_corrected.csv"))):
        t = parse_mixture_label(os.path.basename(p), nb)
        if t and len(t) >= 2:
            X.append(mix_spec(p)); Y.append(_ratio([t.get(n,0) for n in nb]))
X = np.array(X, np.float32); Y = np.array(Y, np.float32); N, nfeat = X.shape

rng = np.random.default_rng(0)
Xs, Cs = simulate_mixtures(P, calib.K, calib.gA, 8000, rng, noise=0.015, baseline=0.03, gain_lo=0.8, gain_hi=1.25)
PRE = (np.array([r/(np.linalg.norm(r)+1e-12) for r in Xs], np.float32), np.array([_ratio(c) for c in Cs], np.float32))
model = train_composition(X, Y, 3, pretrain=PRE, seed=0)              # fit once on all
net = _spec_net(model["n_feat"], model["n_comp"]); net.load_state_dict(model["state"]); net.eval()

def predict(Xa):
    with torch.no_grad():
        return torch.softmax(net(torch.tensor(np.atleast_2d(Xa).astype(np.float32))), 1).numpy()

# ---------- (1) Integrated Gradients ----------
def ig(x, c, steps=48):
    xt = torch.tensor(x, dtype=torch.float32); base = torch.zeros_like(xt)
    tot = torch.zeros_like(xt)
    for k in range(1, steps + 1):
        s = (base + (k/steps)*(xt-base)).unsqueeze(0).requires_grad_(True)
        out = torch.softmax(net(s), 1)[0, c]
        g, = torch.autograd.grad(out, s)
        tot += g[0]
    return ((xt-base) * (tot/steps)).detach().numpy()

attr = {}
for c, name in enumerate(nb):
    present = np.where(Y[:, c] > 0.05)[0]
    A = np.mean([np.abs(ig(X[i], c)) for i in present], axis=0)
    attr[name] = A / (A.max() + 1e-12)

# ---------- (2) band permutation importance ----------
def comp_err(Pm): return np.mean([0.5*np.abs(Pm[i]-Y[i]).sum() for i in range(N)])
E0 = comp_err(predict(X))
edges = np.arange(LO, HI+1, 60.0)
perm = []
rngp = np.random.default_rng(1)
for a, b in zip(edges[:-1], edges[1:]):
    reg = (wnv >= a) & (wnv < b)
    if reg.sum() == 0: perm.append((a+b)/2, ); continue
    ds = []
    for _ in range(5):
        Xp = X.copy(); Xp[:, reg] = X[rngp.permutation(N)][:, reg]
        ds.append(comp_err(predict(Xp)) - E0)
    perm.append(((a+b)/2, float(np.mean(ds))))
perm = np.array(perm)

# ---------- (3) ligand ablation ----------
print("Ligand ablation — zero a compound's VIP bands, measure its predicted fraction:")
abl = {}
base_pred = predict(X)
for c, name in enumerate(nb):
    bands = vip.get(name, [])
    reg = np.zeros(nfeat, bool)
    for wv in bands:
        reg |= (np.abs(wnv - wv) <= 10)
    Xa = X.copy(); Xa[:, reg] = 0.0
    pa = predict(Xa)
    present = np.where(Y[:, c] > 0.05)[0]
    before = base_pred[present, c].mean(); after = pa[present, c].mean()
    abl[name] = (before, after)
    print(f"  {name}: bands {[f'{w:.0f}' for w in bands]}  pred {before:.2f} → {after:.2f} "
          f"({100*(after-before)/(before+1e-9):+.0f}%)")

# ---------- figures (three separate PNGs, no titles — figure-set ready) ----------
COL = {"DQ": "#1a73e8", "TBZ": "#4a9e2a", "THI": "#d6336c"}
D = os.path.join(os.getcwd(), "docs")
# (1) Integrated-Gradients attribution — DQ/TBZ/THI stacked
figA, axes = plt.subplots(len(nb), 1, figsize=(7.2, 5.4), sharex=True)
for ax, name in zip(axes, nb):
    ax.plot(wnv, attr[name], color=COL[name], lw=0.9)
    ax.fill_between(wnv, attr[name], color=COL[name], alpha=0.25)
    for wv in vip.get(name, []):
        ax.axvline(wv, color="#555", ls=":", lw=1.0)
    ax.set_ylabel(f"{name}\nIG attr", color=COL[name], fontweight="bold", fontsize=9)
    ax.set_xlim(LO, HI); ax.set_ylim(0, 1.15)
    for s in ("top","right"): ax.spines[s].set_visible(False)
axes[-1].set_xlabel("wavenumber (cm-1)")
figA.tight_layout(); figA.savefig(os.path.join(D, "interp_ig.png"), dpi=120, facecolor="white", bbox_inches="tight")
# (2) Band permutation importance
figB, axp = plt.subplots(figsize=(7.2, 4.2))
axp.plot(perm[:,0], np.clip(perm[:,1],0,None)*100, color="#25303b", lw=1.2)
axp.fill_between(perm[:,0], np.clip(perm[:,1],0,None)*100, color="#8b95a1", alpha=0.3)
for name in nb:
    for wv in vip.get(name, []):
        axp.axvline(wv, color=COL[name], ls=":", lw=1.0)
axp.set_xlabel("wavenumber (cm-1)"); axp.set_ylabel("Δ composition error (%) when shuffled")
axp.set_xlim(LO, HI)
for s in ("top","right"): axp.spines[s].set_visible(False)
figB.tight_layout(); figB.savefig(os.path.join(D, "interp_permutation.png"), dpi=120, facecolor="white", bbox_inches="tight")
# (3) Ligand ablation
figC, axa = plt.subplots(figsize=(5.4, 4.2))
x = np.arange(len(nb)); w = 0.38
before = [abl[n][0] for n in nb]; after = [abl[n][1] for n in nb]
axa.bar(x-w/2, before, w, label="full spectrum", color=[COL[n] for n in nb], alpha=0.9)
axa.bar(x+w/2, after, w, label="VIP bands ablated", color=[COL[n] for n in nb], alpha=0.35, hatch="//")
axa.set_xticks(x); axa.set_xticklabels(nb); axa.set_ylabel("mean predicted fraction")
axa.legend(fontsize=8, framealpha=0)
for s in ("top","right"): axa.spines[s].set_visible(False)
figC.tight_layout(); figC.savefig(os.path.join(D, "interp_ablation.png"), dpi=120, facecolor="white", bbox_inches="tight")
# data dumps so every panel is redrawable
import csv as _csv
with open(os.path.join(os.getcwd(),"docs","interpretability_ig.csv"),"w",newline="",encoding="utf-8") as _f:
    _w=_csv.writer(_f); _w.writerow(["wavenumber_cm-1"]+[f"IG_{n}" for n in nb])
    for j in range(len(wnv)): _w.writerow([f"{wnv[j]:.1f}"]+[f"{attr[n][j]:.4f}" for n in nb])
with open(os.path.join(os.getcwd(),"docs","interpretability_permutation.csv"),"w",newline="",encoding="utf-8") as _f:
    _w=_csv.writer(_f); _w.writerow(["wavenumber_cm-1","delta_comp_error_frac"])
    for r in perm: _w.writerow([f"{r[0]:.1f}",f"{r[1]:.5f}"])
with open(os.path.join(os.getcwd(),"docs","interpretability_ablation.csv"),"w",newline="",encoding="utf-8") as _f:
    _w=_csv.writer(_f); _w.writerow(["substance","full_spectrum_frac","VIP_ablated_frac"])
    for n in nb: _w.writerow([n,f"{abl[n][0]:.4f}",f"{abl[n][1]:.4f}"])
print("saved docs/dl_interpretability.png + 3 CSVs (ig/permutation/ablation)")
