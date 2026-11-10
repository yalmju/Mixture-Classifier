"""Four-class PLS versus the stored four-class MLP, binary/ternary recovery only."""
from __future__ import annotations
import csv, json, os, re, sys, time
from pathlib import Path
import numpy as np
from sklearn.cross_decomposition import PLSRegression

ROOT=Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
sys.path.insert(0,str(ROOT))
from dl_model import (_refs, nnls_hit_spectra, _map_spectra, _composition_features,
                      load_model, save_model)
from real_data import load_map
from dataset import discover_references, is_blank, base_and_batch

PURE=Path(r"S:\Google Drive\내 드라이브\ACF_PEST_DB\Pure")
MIX=Path(r"S:\Google Drive\내 드라이브\ACF_PEST_DB\Ratio\260814_mixture_final")
OLD=Path(r"S:\Google Drive\내 드라이브\ACF_PEST_DB\260819_Model\composition26081902.dlm")
OUT=ROOT/"documentation/results/four_class_binary_ternary_20260825"
MODEL_OUT=ROOT/"outputs/models_20260825"
SUBS=["DQ","TBZ","THI"]
TRIM=(500.0,2500.0); PX=400
BANDS=np.array([1570.,1270.,1367.])

def parse_c(path):
    m=re.search(r"DQ([0-9.]+)-TB([0-9.]+)-THI?([0-9.]+)",Path(path).name,re.I)
    return np.array([float(m.group(i)) for i in range(1,4)]) if m else None

def norm(p):
    p=np.clip(np.asarray(p,float),0,None); return p/(p.sum(1,keepdims=True)+1e-12)

def group_folds(keys,n=5,seed=825):
    u=list(dict.fromkeys(keys)); rng=np.random.default_rng(seed); u=[u[i] for i in rng.permutation(len(u))]
    bins=[set() for _ in range(n)]
    for i,k in enumerate(u): bins[i%n].add(k)
    return bins

def invcal(I,ab):
    return np.clip(10**((I-ab[:,1])/ab[:,0]),.1,100.)

def build_cache(path):
    subs,wn,mask,P,lo,hi=_refs(str(PURE),False,TRIM)
    assert subs==SUBS,(subs,SUBS)
    X=[];Y=[];groups=[];maps=[];is_mix=[]; conc=[]; band_by_map={}
    files=sorted(MIX.glob("*.csv"),key=lambda p:p.name.lower())
    for k,p in enumerate(files,1):
        c=parse_c(p); y=c/(c.sum()+1e-12); g="r:"+",".join(f"{v:.6f}" for v in y)
        _w,cube,_meta=nnls_hit_spectra(str(PURE),str(p),baseline=False,trim=TRIM,min_frac=.15)
        specs=_map_spectra(cube,np.ones(cube.shape[1],bool),PX,spread=True,baseline_correct=False,
                           sampling="legacy",sampling_seed=0,map_id=str(p))
        a=np.asarray(specs,float); axis=np.asarray(_w,float)
        b=[]
        for centre in BANDS:
            mm=np.abs(axis-centre)<=10; b.append(float(np.mean(np.max(a[:,mm],axis=1))))
        band_by_map[p.name]=b
        for ya in specs:
            X.append(_composition_features(ya)); Y.append([*y,0.]); groups.append(g)
            maps.append(p.name); is_mix.append(True); conc.append(c)
        if k%10==0: print(f"loaded mixture {k}/{len(files)}",flush=True)
    # The old file's LOO paths show seven BLK/INK maps and no pure-reference maps.
    for c,p in discover_references(str(PURE)):
        label=base_and_batch(c)[0]
        if not is_blank(label): continue
        _w,cube,_m,_c=load_map(p)
        specs=_map_spectra(cube,mask,PX,spread=True,baseline_correct=True,
                           sampling="legacy",sampling_seed=0,map_id=p)
        for ya in specs:
            X.append(_composition_features(ya)); Y.append([0.,0.,0.,1.]); groups.append(str(p))
            maps.append(os.path.basename(p)); is_mix.append(False); conc.append([0.,0.,0.])
        print(f"loaded blank {os.path.basename(p)} ({len(specs)} px)",flush=True)
    np.savez_compressed(path,X=np.asarray(X,np.float32),Y=np.asarray(Y,np.float32),
                        groups=np.asarray(groups,object),maps=np.asarray(maps,object),
                        is_mix=np.asarray(is_mix,bool),conc=np.asarray(conc,float),
                        P=np.asarray(P,float),wn=np.asarray(wn,float),
                        band_names=np.asarray(list(band_by_map),object),
                        band_values=np.asarray(list(band_by_map.values()),float))
    return np.load(path,allow_pickle=True)

def pool_by_map(pred,Y,maps,mask):
    rows={}
    for name in dict.fromkeys(maps[mask].tolist()):
        ii=mask&(maps==name); rows[str(name)]=(Y[ii][0],pred[ii].mean(0))
    return rows

def recovery(T,P,sel):
    T=T[sel];P=P[sel]; out={}
    for j,s in enumerate(SUBS):
        z=T[:,j]>0; q=P[z,j]/T[z,j]*100
        out[s]={"n":int(z.sum()),"mean_pct":float(q.mean()),"median_pct":float(np.median(q)),
                "q1_pct":float(np.quantile(q,.25)),"q3_pct":float(np.quantile(q,.75)),
                "within80_120_pct":float(np.mean((q>=80)&(q<=120))*100)}
    allq=P[T>0]/T[T>0]*100
    out["ALL"]={"n":int(len(allq)),"mean_pct":float(allq.mean()),"median_pct":float(np.median(allq)),
                "q1_pct":float(np.quantile(allq,.25)),"q3_pct":float(np.quantile(allq,.75)),
                "within80_120_pct":float(np.mean((allq>=80)&(allq<=120))*100)}
    return out

def main():
    t0=time.time();OUT.mkdir(parents=True,exist_ok=True);MODEL_OUT.mkdir(parents=True,exist_ok=True)
    cache=OUT/"four_class_102maps_cache.npz"
    z=np.load(cache,allow_pickle=True) if cache.exists() else build_cache(cache)
    X=z["X"].astype(np.float32);Y=z["Y"].astype(np.float32);G=z["groups"].astype(object)
    M=z["maps"].astype(object); IM=z["is_mix"].astype(bool); C=z["conc"].astype(float)
    mix_groups=[]
    for name in dict.fromkeys(M[IM].tolist()):
        ii=np.where(IM&(M==name))[0][0]; mix_groups.append(str(G[ii]))
    folds=group_folds(mix_groups,5); pred=np.full_like(Y,np.nan,float); fold_by_map={}
    ref=np.where(~IM)[0]
    for fi,held in enumerate(folds,1):
        te=np.where(IM&np.array([str(g) in held for g in G]))[0]
        tr=np.concatenate([np.where(IM&~np.array([str(g) in held for g in G]))[0],ref])
        print(f"PLS fold {fi}/5 train={len(tr)} test={len(te)}",flush=True)
        sk=PLSRegression(n_components=8,max_iter=500,tol=1e-6).fit(X[tr].astype(float),Y[tr].astype(float))
        pred[te]=norm(sk.predict(X[te].astype(float)))
        for name in dict.fromkeys(M[te].tolist()): fold_by_map[str(name)]=fi
    pls_rows=pool_by_map(pred,Y,M,IM)

    old=load_model(OLD); oe=old["loo_eval"]; mlp_rows={}
    for t,p,path in zip(oe["true"],oe["pred"],oe["paths"]):
        name=Path(str(path)).name
        if name in pls_rows: mlp_rows[name]=(np.asarray(t,float),np.asarray(p,float))
    names=sorted(set(pls_rows)&set(mlp_rows)); assert len(names)==102,len(names)
    T=np.stack([pls_rows[n][0][:3] for n in names]); T=norm(T)
    Ppls=norm(np.stack([pls_rows[n][1][:3] for n in names]))
    Pmlp=norm(np.stack([mlp_rows[n][1][:3] for n in names]))
    Cmap=np.stack([parse_c(n) for n in names]); kval=(Cmap>0).sum(1)
    nonzero=np.where(Cmap>0,Cmap,np.inf); imbalance=Cmap.max(1)/nonzero.min(1)
    keep=imbalance<100
    subsets={"binary":keep&(kval==2),"ternary":keep&(kval==3)}

    result={"protocol":{"mixture_maps":102,"classes":[*SUBS,"BLK"],
              "mlp":"stored 260819 condition-LOO medians","pls":"condition-grouped 5-fold PLS8",
              "exclusion":"max/min nonzero concentration >=100","subsets":["binary","ternary"]},
            "composition":{},"concentration":{}}
    for method,Pp in {"MLP_old4class":Pmlp,"PLS8_4class":Ppls}.items():
        result["composition"][method]={s:recovery(T,Pp,m) for s,m in subsets.items()}

    # Same explicit pie-partitioned single-calibration transform for both models.
    cd=json.loads((ROOT/"documentation/results/integrated_final_20260824/full/concentration_results.json").read_text(encoding="utf-8"))
    ab=np.asarray(cd["protocol"]["calibration_ab"],float)
    bands={str(n):v for n,v in zip(z["band_names"].astype(object),z["band_values"].astype(float))}
    I=np.stack([bands[n] for n in names])
    conc_pred={}
    for method,Pp in {"MLP_old4class":Pmlp,"PLS8_4class":Ppls}.items():
        conc_pred[(method,"raw_signal_partition")]=invcal(I*Pp,ab)
        conc_pred[(method,"above_intercept_partition")]=invcal(ab[:,1]+Pp*(I-ab[:,1]),ab)
        result["concentration"][method]={}
        for variant in ["raw_signal_partition","above_intercept_partition"]:
            Cp=conc_pred[(method,variant)]
            result["concentration"][method][variant]={s:recovery(Cmap,Cp,m) for s,m in subsets.items()}

    # Full-fit PLS DLM for the app, with the same four classes and preprocessing.
    sk=PLSRegression(n_components=8,max_iter=500,tol=1e-6).fit(X.astype(float),Y.astype(float))
    model={"subs":[*SUBS,"BLK"],"lo":500.,"hi":2500.,"n_feat":1290,"P":z["P"].astype(float),
           "feature_mode":"log1p_raw","calib_csv_text":None,"calib_csv_name":None,"uM":None,"has_uM":False,
           "n_train":int(len(X)),"n_maps":int(len(set(M.tolist()))),"px_per_map":400,
           "pixel_sampling":"legacy","sampling_seed":0,"training_level":"pixels","nnls_screen":True,
           "screen_min_frac":.15,"equal_volume_mix":False,"concentration_basis":"filename final uM",
           "screen_stats":None,"data_dir":str(PURE),"baseline":False,"trim":TRIM,
           "train_eval":{"true":[],"pred":[],"loss":[]},
           "loo_eval":{"true":[T[i].tolist()+[0.] for i in range(len(T))],
                       "pred":[np.r_[Ppls[i]*(1-pls_rows[names[i]][1][3]),pls_rows[names[i]][1][3]].tolist() for i in range(len(T))],
                       "paths":names,"level":"condition-grouped-5-fold"},
           "test_eval":None,"blank":"BLK","method":"pls","sk":sk,"n_components":8,
           "protocol_version":"four_class_binary_ternary_20260825"}
    save_model(model,MODEL_OUT/"UNMIXR_PLS8_4class_102mixtures_20260825.dlm")

    (OUT/"four_class_binary_ternary_results.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    with (OUT/"recovery_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        fields=["endpoint","method","variant","subset","component","n","mean_recovery_pct","median_recovery_pct","q1_pct","q3_pct","within80_120_pct"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for method,sets in result["composition"].items():
            for subset,comps in sets.items():
                for component,v in comps.items(): w.writerow({"endpoint":"composition","method":method,"variant":"pie","subset":subset,"component":component,"n":v["n"],"mean_recovery_pct":v["mean_pct"],"median_recovery_pct":v["median_pct"],"q1_pct":v["q1_pct"],"q3_pct":v["q3_pct"],"within80_120_pct":v["within80_120_pct"]})
        for method,vars_ in result["concentration"].items():
            for variant,sets in vars_.items():
                for subset,comps in sets.items():
                    for component,v in comps.items(): w.writerow({"endpoint":"concentration","method":method,"variant":variant,"subset":subset,"component":component,"n":v["n"],"mean_recovery_pct":v["mean_pct"],"median_recovery_pct":v["median_pct"],"q1_pct":v["q1_pct"],"q3_pct":v["q3_pct"],"within80_120_pct":v["within80_120_pct"]})
    print(json.dumps(result,indent=2,ensure_ascii=False));print(f"elapsed_min={(time.time()-t0)/60:.1f}",flush=True)

if __name__=="__main__": main()
