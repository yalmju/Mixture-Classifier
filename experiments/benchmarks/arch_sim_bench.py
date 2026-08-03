# -*- coding: utf-8 -*-
"""Architecture (MLP vs 1D-CNN) x simulator (base vs domain-randomised) benchmark,
full 35-map leave-one-out. Metrics: overall composition error, buried non-THI error,
buried detection. Uses the minor-component-weighted L1 loss (identity-agnostic)."""
import os, sys, glob
sys.path.insert(0, os.getcwd())
import numpy as np
root = "S:/Google Drive"
acf = next(os.path.join(e.path, "ACF_PEST_DB") for e in os.scandir(root)
           if e.is_dir() and os.path.isdir(os.path.join(e.path, "ACF_PEST_DB")))
pure = os.path.join(acf, "Pure")
folders = [os.path.join(acf, "Ratio", "Binary"), os.path.join(acf, "Ratio", "Tertiary")]
from unmix import _templates, _baseline_removed, _l2
from real_data import load_map
from dataset import is_blank
from sers_mixture import als_baseline
from validate import parse_mixture_label
from dl_quantify import simulate_mixtures, _ratio
from calibration import calibrate
from io_utils import load_calibration_csv
import torch, torch.nn as nn

LO, HI = 300.0, 1800.0
names, wn, means = _templates(pure, True, None); m = (wn >= LO) & (wn <= HI)
nb = [names[i] for i in range(len(names)) if not is_blank(names[i])]
THI = nb.index("THI"); nonTHI = [i for i in range(3) if i != THI]
nonbg = [i for i in range(len(names)) if not is_blank(names[i])]
P = _l2(_baseline_removed(means[nonbg][:, m], True))
ax_c, names_c, dils = load_calibration_csv(os.path.join(acf, "Ratio", "results", "calibration_spectra.csv"))
mc = (ax_c >= LO) & (ax_c <= HI)
calib = calibrate([(dils[names_c.index(n)][0], np.asarray(dils[names_c.index(n)][1])[:, mc]) for n in nb], P, nb)

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
X = np.array(X, np.float32); Y = np.array(Y, np.float32); N = len(X); nfeat = X.shape[1]
print(f"{N} mixtures · n_feat {nfeat}")

def make_sim(rich):
    rng = np.random.default_rng(0)
    if rich:   # domain randomisation: wider gain, stronger/ varied noise + baseline
        Xs, Cs = simulate_mixtures(P, calib.K, calib.gA, 8000, rng, noise=0.035,
                                   baseline=0.06, gain_lo=0.5, gain_hi=1.8)
    else:
        Xs, Cs = simulate_mixtures(P, calib.K, calib.gA, 8000, rng, noise=0.015,
                                   baseline=0.03, gain_lo=0.8, gain_hi=1.25)
    Xs = np.array([r/(np.linalg.norm(r)+1e-12) for r in Xs], np.float32)
    return Xs, np.array([_ratio(c) for c in Cs], np.float32)

def mlp(): return nn.Sequential(nn.Linear(nfeat,256), nn.BatchNorm1d(256), nn.ReLU(),
                                nn.Dropout(0.15), nn.Linear(256,64), nn.ReLU(), nn.Linear(64,3))
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Sequential(nn.Conv1d(1,16,7,padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
                               nn.Conv1d(16,32,5,padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.AdaptiveAvgPool1d(8))
        self.f = nn.Sequential(nn.Flatten(), nn.Linear(32*8,64), nn.ReLU(), nn.Dropout(0.15), nn.Linear(64,3))
    def forward(self, x): return self.f(self.c(x.unsqueeze(1)))

sm = nn.LogSoftmax(dim=1)
def wl(net, xb, yb):                         # minor-component-weighted L1
    p = sm(net(xb)).exp(); w = 1.0 + 2.0*(1.0-yb); return (w*(p-yb).abs()).sum(1).mean()

def train_pred(build, pre, tr, i, seed):
    torch.manual_seed(seed); net = build()
    Xp, Yp = pre
    op = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xp_t = torch.tensor(Xp); Yp_t = torch.tensor(Yp)
    for _ in range(20):                       # pretrain (minibatch)
        for b in range(0, len(Xp), 256):
            op.zero_grad(); wl(net, Xp_t[b:b+256], Yp_t[b:b+256]).backward(); op.step()
    op = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-3)
    Xt = torch.tensor(X[tr]); Yt = torch.tensor(Y[tr])
    for _ in range(220):
        net.train(); op.zero_grad(); wl(net, Xt, Yt).backward(); op.step()
    net.eval()
    with torch.no_grad(): return torch.softmax(net(torch.tensor(X[i][None,:])),1).numpy()[0]

def report(tag, Pm):
    ce = np.mean([0.5*np.abs(Pm[i]-Y[i]).sum() for i in range(N)])
    ti = np.where(Y[:,THI]>0)[0]
    be = np.mean([np.abs(Pm[i][nonTHI]-Y[i][nonTHI]).sum() for i in ti])
    det = np.mean([np.mean([(Pm[i][j]>0.05) for j in nonTHI if Y[i][j]>0.05] or [1.0]) for i in ti])
    print(f"  {tag:<22} comp {ce:.1%}  buried_err {be:.2f}  detect {det:.0%}")

print("\n35-map LOO (minor-weighted loss):")
for arch, build in [("MLP", mlp), ("1D-CNN", CNN)]:
    for rich in (False, True):
        pre = make_sim(rich)
        Pm = np.array([train_pred(build, pre, [j for j in range(N) if j!=i], i, i) for i in range(N)])
        report(f"{arch} · {'rich' if rich else 'base'} sim", Pm)
