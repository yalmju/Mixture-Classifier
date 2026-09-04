"""Pool repeat maps to 92 unique conditions and finalize binary/ternary recovery."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.cross_decomposition import PLSRegression

ROOT=Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
sys.path.insert(0,str(ROOT))
from dl_model import load_model
from documentation.scripts.run_4class_binary_ternary import (
    OUT, OLD, SUBS, group_folds, norm, pool_by_map, recovery, parse_c, invcal)

def main():
    z=np.load(OUT/"four_class_102maps_cache.npz",allow_pickle=True)
    X=z["X"].astype(np.float32);Y=z["Y"].astype(np.float32);G=z["groups"].astype(object)
    M=z["maps"].astype(object);IM=z["is_mix"].astype(bool)
    mix_groups=[]
    for name in dict.fromkeys(M[IM].tolist()):
        ii=np.where(IM&(M==name))[0][0];mix_groups.append(str(G[ii]))
    folds=group_folds(mix_groups,5);pred=np.full_like(Y,np.nan,float);ref=np.where(~IM)[0]
    for fi,held in enumerate(folds,1):
        isheld=np.array([str(g) in held for g in G])
        te=np.where(IM&isheld)[0];tr=np.concatenate([np.where(IM&~isheld)[0],ref])
        print(f"PLS fold {fi}/5",flush=True)
        sk=PLSRegression(n_components=8,max_iter=500,tol=1e-6).fit(X[tr].astype(float),Y[tr].astype(float))
        pred[te]=norm(sk.predict(X[te].astype(float)))
    pls_rows=pool_by_map(pred,Y,M,IM)
    old=load_model(OLD);mlp_rows={}
    for t,p,path in zip(old["loo_eval"]["true"],old["loo_eval"]["pred"],old["loo_eval"]["paths"]):
        name=Path(str(path)).name
        if name in pls_rows:mlp_rows[name]=(np.asarray(t,float),np.asarray(p,float))
    mapnames=sorted(set(pls_rows)&set(mlp_rows)); bands={str(n):v for n,v in zip(z["band_names"],z["band_values"])}

    # Average repeated maps with the same absolute concentration label.
    groups={}
    for name in mapnames: groups.setdefault(tuple(parse_c(name).tolist()),[]).append(name)
    keys=sorted(groups); C=np.asarray(keys,float);T=[];Pmlp=[];Ppls=[];I=[];members=[]
    for key in keys:
        ns=groups[key];members.append(ns)
        T.append(pls_rows[ns[0]][0][:3])
        Pmlp.append(np.mean([norm(np.asarray(mlp_rows[n][1][:3])[None,:])[0] for n in ns],0))
        Ppls.append(np.mean([norm(np.asarray(pls_rows[n][1][:3])[None,:])[0] for n in ns],0))
        I.append(np.mean([bands[n] for n in ns],0))
    T=norm(T);Pmlp=norm(Pmlp);Ppls=norm(Ppls);I=np.asarray(I)
    kval=(C>0).sum(1);imb=C.max(1)/np.where(C>0,C,np.inf).min(1);keep=imb<100
    subsets={"binary":keep&(kval==2),"ternary":keep&(kval==3)}
    cd=json.loads((ROOT/"documentation/results/integrated_final_20260824/full/concentration_results.json").read_text(encoding="utf-8"))
    ab=np.asarray(cd["protocol"]["calibration_ab"],float)
    result={"protocol":{"unique_conditions_total":int(len(C)),"included_after_100x_exclusion":int(keep.sum()),
             "binary_n":int(subsets["binary"].sum()),"ternary_n":int(subsets["ternary"].sum()),
             "repeat_pooling":"mean pie and band signal across identical absolute concentration labels"},
            "composition":{},"concentration":{}}
    for method,P in {"MLP_old4class":Pmlp,"PLS8_4class":Ppls}.items():
        result["composition"][method]={s:recovery(T,P,m) for s,m in subsets.items()}
        result["concentration"][method]={}
        for variant,Cp in {"raw_signal_partition":invcal(I*P,ab),
                           "above_intercept_partition":invcal(ab[:,1]+P*(I-ab[:,1]),ab)}.items():
            result["concentration"][method][variant]={s:recovery(C,Cp,m) for s,m in subsets.items()}
    (OUT/"four_class_92condition_results.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    with (OUT/"recovery_92conditions.csv").open("w",newline="",encoding="utf-8-sig") as f:
        fields=["endpoint","method","variant","subset","component","n","mean_recovery_pct","median_recovery_pct","q1_pct","q3_pct","within80_120_pct"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for method,sets in result["composition"].items():
            for subset,comps in sets.items():
                for comp,v in comps.items():w.writerow({"endpoint":"composition","method":method,"variant":"pie","subset":subset,"component":comp,"n":v["n"],"mean_recovery_pct":v["mean_pct"],"median_recovery_pct":v["median_pct"],"q1_pct":v["q1_pct"],"q3_pct":v["q3_pct"],"within80_120_pct":v["within80_120_pct"]})
        for method,variants in result["concentration"].items():
            for variant,sets in variants.items():
                for subset,comps in sets.items():
                    for comp,v in comps.items():w.writerow({"endpoint":"concentration","method":method,"variant":variant,"subset":subset,"component":comp,"n":v["n"],"mean_recovery_pct":v["mean_pct"],"median_recovery_pct":v["median_pct"],"q1_pct":v["q1_pct"],"q3_pct":v["q3_pct"],"within80_120_pct":v["within80_120_pct"]})
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=="__main__":main()
