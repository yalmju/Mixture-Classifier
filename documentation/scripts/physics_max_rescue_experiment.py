"""Paired baseline vs surface-constrained MLP on the hardest held-out mixture.

Uses Ratio/260814_mixture_final as the single source of truth.  The target
DQ12-TB6-TH3 condition is excluded from both fits.  Both models start from the
same physics-pretrained weights; only lambda_surface differs (0 vs 0.03).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from calibration import calibrate, fit_sips
from dl_model import (_composition_features, _map_spectra, _nuisance_refs,
                      _refs, nnls_hit_spectra)
from dl_quantify import _ratio, _spec_net, simulate_mixtures, surface_composition
from io_utils import load_calibration_csv
from physics_surface_loss import (SurfacePhysics, prepare_surface_physics,
                                  surface_consistency_loss)
from real_data import load_map
from unmix import _baseline_removed, _l2

SUB = ["DQ", "TBZ", "THI"]
COL = {"DQ": "#1687e0", "TBZ": "#23a36d", "THI": "#ef5275"}
TARGET = (12, 6, 3)


def parse_folder(folder):
    rows = []
    for name in sorted(os.listdir(folder)):
        m = re.match(r"DQ(\d+)-TB(\d+)-THI?(\d+)\.csv$", name, re.I)
        if m:
            c = tuple(map(int, m.groups()))
            rows.append((os.path.join(folder, name), c))
    low = [(p, c) for p, c in rows if all(v in (3, 6, 12, 24) for v in c)]
    high = [(p, c) for p, c in rows if not all(v in (3, 6, 12, 24) for v in c)]
    return low, high


def calibration_path(db):
    for parts in [("STD", "260727_Spec"), ("STD", "260727_VIP"),
                  ("Pest", "Standard", "260729")]:
        p = os.path.join(db, *parts, "calibration_spectra.csv")
        if os.path.exists(p):
            return p
    raise FileNotFoundError("calibration_spectra.csv")


def gather_rows(pure, items, P, px_per_map=20):
    X, Y, C, S, paths = [], [], [], [], []
    for n, (path, conc_uM) in enumerate(items):
        print(f"loading {n+1}/{len(items)} {os.path.basename(path)}", flush=True)
        _w, hit_cube, _meta = nnls_hit_spectra(
            pure, path, baseline=True, trim=None, min_frac=.15)
        specs = _map_spectra(hit_cube, np.ones(hit_cube.shape[1], bool),
                             px_per_map, baseline_correct=False)
        true_C = np.asarray(conc_uM, float) * 1e-6
        true_y = _ratio(true_C)
        for raw in specs:
            X.append(_composition_features(raw, "log1p_raw"))
            Y.append(true_y)
            C.append(true_C)
            S.append(surface_composition(
                _composition_features(np.asarray(raw)[None, :], "legacy_l2"), P)[0])
            paths.append(path)
    return (np.asarray(X, np.float32), np.asarray(Y, np.float32),
            np.asarray(C, float), np.asarray(S, np.float32), paths)


def physics_pretrain(pure, calib_csv, P, mask):
    ax, names, dils = load_calibration_csv(calib_csv)
    mc = (ax >= 300) & (ax <= 1800)
    dil = [(dils[names.index(s)][0], np.asarray(dils[names.index(s)][1])[:, mc]) for s in SUB]
    cal = calibrate(dil, P, SUB)
    m = np.ones(3)
    m[0] = fit_sips(cal.C_series[0], cal.B_series[0])[2]
    rng = np.random.default_rng(0)
    Xp, Cp = simulate_mixtures(P, cal.K, cal.gA, 5000, rng, noise=.015,
                               baseline=.03, gain_lo=.8, gain_hi=1.25,
                               iso_m=m, nuisance=None)
    return _composition_features(Xp, "log1p_raw").astype(np.float32), np.asarray([_ratio(c) for c in Cp], np.float32), cal, m


def fit_models(X, Y, C, S, pre, cal, m, epochs, weights=(0.0, .03)):
    import torch
    torch.manual_seed(0)
    base = _spec_net(X.shape[1], 3)
    sm = torch.nn.LogSoftmax(dim=1)
    Xp, Yp = map(torch.tensor, pre)
    op = torch.optim.Adam(base.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(25):
        base.train(); op.zero_grad(); pred = sm(base(Xp)).exp()
        w = 1 + 2 * (1 - Yp); loss = (w * (pred - Yp).abs()).sum(1).mean()
        loss.backward(); op.step()
    initial = {k: v.detach().clone() for k, v in base.state_dict().items()}
    physics = prepare_surface_physics(C, S, SurfacePhysics(cal.K, cal.gA, m, weight=.03))
    Xt, Yt = torch.tensor(X), torch.tensor(Y); ridx = torch.arange(len(Xt))
    out = {}
    for lam in weights:
        net = _spec_net(X.shape[1], 3); net.load_state_dict(initial)
        opt = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-3)
        hist = []
        for ep in range(epochs):
            net.train(); opt.zero_grad(); pred = sm(net(Xt)).exp()
            w = 1 + 2 * (1 - Yt); ld = (w * (pred - Yt).abs()).sum(1).mean()
            lp = surface_consistency_loss(pred, ridx, physics)
            loss = ld + float(lam) * lp; loss.backward(); opt.step()
            hist.append((float(loss.detach()), float(ld.detach()), float(lp.detach())))
            if ep % 20 == 0 or ep == epochs - 1:
                print(f"lambda={lam:g} epoch={ep+1}/{epochs} data={hist[-1][1]:.4f} physics={hist[-1][2]:.4f}", flush=True)
        net.eval(); out[lam] = (net, hist)
    return out


def target_features(pure, path):
    _w, cube, _meta = nnls_hit_spectra(pure, path, baseline=True, trim=None, min_frac=.15)
    raws = _map_spectra(cube, np.ones(cube.shape[1], bool), 0, baseline_correct=False)
    return np.asarray([_composition_features(r, "log1p_raw") for r in raws], np.float32)


def predict(net, X):
    import torch
    net.eval()
    with torch.no_grad():
        p = torch.softmax(net(torch.tensor(X)), 1).numpy().mean(0)
    return p / p.sum()


def nnls_and_spectrum(pure, path):
    names, wn, mask, P, lo, hi = _refs(pure, True, None)
    w, cube, _, _ = load_map(path); w = np.asarray(w); mm = (w >= lo) & (w <= hi)
    Y = np.asarray(cube, float)[:, mm]; wt = Y.sum(1); raw = (wt/(wt.sum()+1e-12)) @ Y
    y = _l2(_baseline_removed(raw[None, :], True))[0]
    B, _ = nnls(P.T, y); frac = B/(B.sum()+1e-12)
    return w[mm], raw, y, P, B, frac


def draw(outdir, wn, y, P, B, true, nnls_f, base, phys, errors):
    fig = plt.figure(figsize=(5.4, 3.0), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.9, 1], hspace=.12, wspace=.28)
    a0=fig.add_subplot(gs[0,0]); a1=fig.add_subplot(gs[1,0],sharex=a0); ar=fig.add_subplot(gs[:,1])
    dom=int(np.argmax(B)); dname=SUB[dom]; dcomp=B[dom]*P[dom]
    res=np.clip(y-dcomp,0,None)
    a0.plot(wn,y,color="#34383e",lw=.9); a0.fill_between(wn,0,dcomp,color=COL[dname],alpha=.28)
    a0.text(.01,.86,"Observed spectrum",transform=a0.transAxes,weight="bold")
    a0.text(.99,.86,f"{dname}-dominated NNLS",transform=a0.transAxes,ha="right",color=COL[dname],weight="bold")
    a1.plot(wn,res,color="#747b86",lw=.85); a1.text(.01,.86,f"After {dname} projection",transform=a1.transAxes,weight="bold")
    for s, vals in {"DQ":[1174,1572],"TBZ":[1010,1270],"THI":[550,1368]}.items():
        for v in vals:
            i=int(np.argmin(abs(wn-v))); a1.scatter(wn[i],res[i],s=12,color=COL[s],zorder=4)
    a1.set_xlabel("Raman shift (cm$^{-1}$)"); fig.text(.015,.52,"Normalized intensity (a.u.)",rotation=90,va="center")
    for a in (a0,a1): a.spines[["top","right"]].set_visible(False); a.set_yticks([])
    a0.tick_params(labelbottom=False)
    ar.axis("off"); ar.text(.5,.96,"Held-out rescue",ha="center",weight="bold",fontsize=9)
    rows=[("True",true,None),("NNLS",nnls_f,errors["NNLS"]),("MLP",base,errors["MLP"]),("Physics-MLP",phys,errors["Physics-MLP"])]
    y0=.80
    for label,p,e in rows:
        ar.text(.02,y0,label,transform=ar.transAxes,weight="bold" if "Physics" in label else None)
        ar.text(.98,y0,"/".join(f"{100*x:.0f}" for x in p),transform=ar.transAxes,ha="right")
        if e is not None: ar.text(.98,y0-.08,f"error {e:.1f}%",transform=ar.transAxes,ha="right",color="#667085")
        y0-=.20
    ar.text(.5,.02,"$L=L_{data}+0.03L_{surface}$",transform=ar.transAxes,ha="center",weight="bold",fontsize=8)
    fig.suptitle("Physics-constrained recovery of the hardest held-out mixture",x=.06,ha="left",weight="bold",fontsize=10)
    p=os.path.join(outdir,"panel_h_physics_max_rescue.png"); fig.savefig(p,dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig); return p,res


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--db",required=True); ap.add_argument("--out",required=True); ap.add_argument("--epochs",type=int,default=120); a=ap.parse_args()
    os.makedirs(a.out,exist_ok=True); folder=os.path.join(a.db,"Ratio","260814_mixture_final"); pure=os.path.join(a.db,"Pure")
    low,high=parse_folder(folder); target=next(p for p,c in low if c==TARGET)
    keys=sorted(c for _,c in low); order=np.random.default_rng(0).permutation(len(keys)); folds=[[keys[i] for i in order[f::4]] for f in range(4)]; fi=next(i for i,f in enumerate(folds) if TARGET in f)
    held=set(folds[fi]); train=high+[(p,c) for p,c in low if c not in held]
    print(f"fold={fi+1} train={len(train)} held_low={len(held)} target={target}",flush=True)
    _subs,_wn,mask,P,_lo,_hi=_refs(pure,True,None)
    X,Y,C,S,_=gather_rows(pure,train,P,20)
    preX,preY,cal,m=physics_pretrain(pure,calibration_path(a.db),P,mask)
    models=fit_models(X,Y,C,S,(preX,preY),cal,m,a.epochs)
    Xt=target_features(pure,target); pred0=predict(models[0.0][0],Xt); predp=predict(models[.03][0],Xt)
    wn,raw,y,P2,B,nf=nnls_and_spectrum(pure,target); true=_ratio(np.asarray(TARGET,float)); err=lambda p:float(.5*abs(p-true).sum()*100)
    errors={"NNLS":err(nf),"MLP":err(pred0),"Physics-MLP":err(predp)}
    png,res=draw(a.out,wn,y,P2,B,true,nf,pred0,predp,errors)
    with open(os.path.join(a.out,"panel_h_physics_max_rescue.csv"),"w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["method","DQ_pct","TBZ_pct","THI_pct","error_pct"]); w.writerow(["True",*(true*100),0]); w.writerow(["NNLS",*(nf*100),errors["NNLS"]]); w.writerow(["MLP_lambda0",*(pred0*100),errors["MLP"]]); w.writerow(["Physics_MLP_lambda0.03",*(predp*100),errors["Physics-MLP"]])
    with open(os.path.join(a.out,"panel_h_physics_max_rescue_spectrum.csv"),"w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["raman_shift_cm-1","measured_raw","preprocessed","dominant_component_fit","residual"]); dom=int(np.argmax(B)); w.writerows(zip(wn,raw,y,B[dom]*P2[dom],res))
    print(png,flush=True); print("true",true,"nnls",nf,"mlp",pred0,"physics",predp,"errors",errors,flush=True)


if __name__=="__main__": main()
