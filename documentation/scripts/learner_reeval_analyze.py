"""learner_reeval_run.py 결과 집계 — 실행:  ../../.venv/bin/python3 learner_reeval_analyze.py > ../results/learner_reeval_2026-11-10/summary.txt"""
import numpy as np, sys, collections
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "learner_reeval_2026-11-10")
z=np.load(os.path.join(OUT, "eval_learners.npz"),allow_pickle=True)
YM=z["YM"]; CM=z["CM"]
grid=CM.max(1)<=24.5; k=(CM>0).sum(1)
SUB={"grid64":grid,"hi-bin":(~grid)&(k==2),"hi-tern":(~grid)&(k==3),"all89":np.ones(len(YM),bool)}
NREP={"cond":20,"ratio":10,"grid":10}
BASE=["nnls","pls8","plsT","ridgeT","rf","mlp_s0","mlp_s1","mlp_s2","mlp350"]
def preds(s,r):
    d={n:z[f"{s}|{r}|{n}"] for n in BASE}
    d["mlp_ens3"]=(d["mlp_s0"]+d["mlp_s1"]+d["mlp_s2"])/3
    d["pls+mlp"]=(d["pls8"]+d["mlp_s0"])/2
    d["pls+mlpE"]=(d["pls8"]+d["mlp_ens3"])/2
    d["pls+rf"]=(d["pls8"]+d["rf"])/2
    return d
def dev(p): return 50*np.abs(p-YM).sum(1)
rng=np.random.default_rng(0)
for s,nr in NREP.items():
    P=[preds(s,r) for r in range(nr)]
    M=list(P[0].keys())
    D={n:np.array([dev(P[r][n]) for r in range(nr)]) for n in M}     # (rep, map)
    D["mlp(1seed)"]=(D["mlp_s0"]+D["mlp_s1"]+D["mlp_s2"])/3
    print(f"\n##### scheme={s}  repeats={nr}")
    subs=["grid64"] if s=="grid" else list(SUB)
    show=["nnls","pls8","plsT","ridgeT","rf","mlp(1seed)","mlp350","mlp_ens3","pls+mlp","pls+mlpE","pls+rf"]
    print(f"{'method':11s}"+"".join(f"{q:>22s}" for q in subs))
    for n in show:
        row=f"{n:11s}"
        for q in subs:
            m=SUB[q]; per_rep=np.nanmean(D[n][:,m],1)
            row+=f"   {per_rep.mean():5.2f} ±{per_rep.std(ddof=1):4.2f} med{np.nanmedian(D[n][:,m]):5.1f}"
        print(row)
    print("-- paired vs pls8 (per-map mean over repeats; bootstrap over maps) / fraction of repeats where method < pls8")
    for n in ["plsT","ridgeT","rf","mlp(1seed)","mlp350","mlp_ens3","pls+mlp","pls+mlpE","nnls"]:
        row=f"{n:11s}"
        for q in subs:
            m=SUB[q]; d=np.nanmean(D[n][:,m],0)-np.nanmean(D["pls8"][:,m],0)
            bs=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(4000)]
            if n=="mlp(1seed)":
                w=np.mean([np.nanmean(D[f"mlp_s{j}"][:,m],1)<np.nanmean(D["pls8"][:,m],1) for j in range(3)])
            else:
                w=np.mean(np.nanmean(D[n][:,m],1)<np.nanmean(D["pls8"][:,m],1))
            row+=f"   {d.mean():+5.2f} [{np.percentile(bs,2.5):+5.2f},{np.percentile(bs,97.5):+5.2f}] win{100*w:3.0f}%"
        print(row)
    # seed SD of MLP at fixed folds vs fold SD
    m=SUB["grid64"]
    seedsd=np.std([np.nanmean(D[f"mlp_s{j}"][:,m],1) for j in range(3)],axis=0,ddof=1).mean()
    print(f"-- grid64: MLP seed-SD (same folds) {seedsd:.2f} %p ; fold-SD MLP {np.nanmean(D['mlp_s0'][:,m],1).std(ddof=1):.2f} ; fold-SD PLS {np.nanmean(D['pls8'][:,m],1).std(ddof=1):.2f}")
    print(f"   single-split range grid64: PLS {np.nanmean(D['pls8'][:,m],1).min():.1f}–{np.nanmean(D['pls8'][:,m],1).max():.1f} ; MLP(s0) {np.nanmean(D['mlp_s0'][:,m],1).min():.1f}–{np.nanmean(D['mlp_s0'][:,m],1).max():.1f} ; diff(MLP−PLS) {(np.nanmean(D['mlp_s0'][:,m],1)-np.nanmean(D['pls8'][:,m],1)).min():+.1f}…{(np.nanmean(D['mlp_s0'][:,m],1)-np.nanmean(D['pls8'][:,m],1)).max():+.1f}")
    # declared-total on grid64
    print("-- grid64 declared total: within-2x (%) / accuracy mean min(p/t,t/p) / worst-decile dev")
    for n in ["nnls","pls8","plsT","rf","mlp_s0","mlp350","mlp_ens3","pls+mlpE"]:
        w=[];a=[];t=[]
        for r in range(nr):
            p=P[r][n][m]; y=YM[m]; q=np.clip(p,1e-6,None)/y
            w.append(np.mean((q>=0.5)&(q<=2))*100); a.append(np.mean(np.minimum(q,1/q)))
            t.append(np.percentile(dev(P[r][n])[m],90))
        print(f"   {n:9s} {np.mean(w):5.1f} ±{np.std(w,ddof=1):3.1f}   {np.mean(a):.3f} ±{np.std(a,ddof=1):.3f}   p90 {np.mean(t):5.1f}")
    pk=np.vstack([z[f"{s}|{r}|picks"] for r in range(nr)])
    print("-- tuned picks: PLS nc", dict(collections.Counter(pk[:,0].astype(int).tolist())), " ridge alpha", dict(collections.Counter(pk[:,1].tolist())))
