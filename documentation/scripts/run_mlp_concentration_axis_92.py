"""MLP composition + single-component calibration concentration axis, 92 conditions."""
import csv,json,pickle,re,sys
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from scipy.stats import pearsonr,spearmanr

ROOT=Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
sys.path.insert(0,str(ROOT))
from io_utils import load_calibration_csv
CACHE=ROOT/"documentation/results/four_class_binary_ternary_20260825/four_class_102maps_cache.npz"
DLM=Path(r"S:\Google Drive\내 드라이브\ACF_PEST_DB\260819_Model\composition26081902.dlm")
CAL=Path(r"S:\Google Drive\내 드라이브\ACF_PEST_DB\260821_Calib_144-9\calibration_spectra.csv")
OUT=ROOT/"documentation/results/mlp_concentration_axis_92_20260825"
SUBS=["DQ","TBZ","THI"]
# weighted 1-72 uM fits from the user-provided Calibration_260821 mean-SD sheet
AB=np.array([[4625.655127843041,2252.5161949229278],
             [3814.2832708893084,436.0644069596556],
             [4571.069160200576,919.411596318233]],float)

def norm(x):
 x=np.clip(np.asarray(x,float),0,None); return x/(x.sum(axis=-1,keepdims=True)+1e-12)
def parse(name):
 m=re.search(r"DQ([0-9.]+)-TB([0-9.]+)-THI?([0-9.]+)",name,re.I)
 return np.array([float(m.group(i)) for i in range(1,4)]) if m else None
def rkey(c): return ":".join(f"{v:.8f}" for v in norm(c[None,:])[0])
def folds(keys,n=5,seed=825):
 u=list(dict.fromkeys(keys)); rng=np.random.default_rng(seed); u=[u[i] for i in rng.permutation(len(u))]
 out=[set() for _ in range(n)]
 for i,k in enumerate(u): out[i%n].add(k)
 return out

def calibration_transfer(P):
 axis,names,dils=load_calibration_csv(str(CAL)); mask=(axis>=500)&(axis<=2500)
 assert mask.sum()==P.shape[1]
 Q=[]; diag=[]
 for j,s in enumerate(SUBS):
  C,Y=dils[names.index(s)]; bb=[];cc=[]
  for c,y in zip(np.asarray(C)*1e6,np.asarray(Y)[:,mask]):
   if 9<=c<=72:
    b,_=nnls(P.T,np.clip(y,0,None)); bb.append(b[j]);cc.append(c)
  bb=np.asarray(bb);cc=np.asarray(cc); target=AB[j,0]*np.log10(cc)+AB[j,1]
  A=np.c_[bb,np.ones(len(bb))]; q=np.linalg.lstsq(A,target,rcond=None)[0]; pred=A@q
  r2=1-np.sum((target-pred)**2)/(np.sum((target-target.mean())**2)+1e-12)
  Q.append(q);diag.append({"component":s,"n":len(bb),"slope":q[0],"intercept":q[1],"r2":r2})
 return np.asarray(Q),diag

def map_B(X,maps,is_mix,P):
 raw=np.expm1(X.astype(float)); out={}
 for name in dict.fromkeys(maps[is_mix].tolist()):
  ii=np.where(is_mix&(maps==name))[0]; B=[]
  for y in raw[ii]: B.append(nnls(P.T,np.clip(y,0,None))[0])
  out[str(name)]=np.median(np.asarray(B),axis=0)
 return out

def metrics(T,P):
 present=T>0;t=T[present];p=P[present];q=p/t; loge=np.log2(np.clip(q,1e-9,None))
 d={"n_components":len(t),"mean_recovery_pct":100*q.mean(),"median_recovery_pct":100*np.median(q),
    "q1_recovery_pct":100*np.quantile(q,.25),"q3_recovery_pct":100*np.quantile(q,.75),
    "within80_120_pct":100*np.mean((q>=.8)&(q<=1.2)),"mae_uM":np.mean(np.abs(p-t)),
    "rmse_log2":np.sqrt(np.mean(loge**2)),"median_fold_error":2**np.median(np.abs(loge)),
    "euclidean_mean_uM":np.mean(np.linalg.norm(P-T,axis=1)),
    "euclidean_median_uM":np.median(np.linalg.norm(P-T,axis=1))}
 d["pearson_r"]=pearsonr(t,p).statistic if np.std(t)>0 and np.std(p)>0 else None
 d["spearman_rho"]=spearmanr(t,p).statistic if np.std(t)>0 and np.std(p)>0 else None
 mt,mp=t.mean(),p.mean();d["ccc"]=2*np.mean((t-mt)*(p-mp))/(t.var()+p.var()+(mt-mp)**2+1e-12)
 return {k:(float(v) if v is not None else None) for k,v in d.items()}

def main():
 OUT.mkdir(parents=True,exist_ok=True);z=np.load(CACHE,allow_pickle=True)
 X=z["X"];maps=z["maps"].astype(object);mix=z["is_mix"].astype(bool);P=z["P"].astype(float)
 old=pickle.load(open(DLM,"rb"));oof={}
 for p,path in zip(old["loo_eval"]["pred"],old["loo_eval"]["paths"]):
  name=Path(str(path)).name
  if parse(name) is not None:oof[name]=norm(np.asarray(p[:3])[None,:])[0]
 B=map_B(X,maps,mix,P);Q,diag=calibration_transfer(P)
 rec=[]
 for name in sorted(set(oof)&set(B)):
  c=parse(name);nz=c[c>0];k=(c>0).sum()
  if nz.max()/nz.min()>=100 or k not in (2,3):continue
  I=B[name]*Q[:,0]+Q[:,1];app=np.clip(10**((I-AB[:,1])/AB[:,0]),.1,500)
  rec.append({"name":name,"true":c,"pie":oof[name],"app":app,"subset":"binary" if k==2 else "ternary","key":rkey(c)})
 # pool exact duplicate absolute condition
 pools={}
 for r in rec:pools.setdefault(tuple(r["true"]),[]).append(r)
 rows=[]
 for key,rr in pools.items():
  rows.append({"condition":f"DQ{key[0]:g}-TB{key[1]:g}-TH{key[2]:g}","true":np.array(key,float),
   "pie":norm(np.mean([x["pie"] for x in rr],0)[None,:])[0],"app":np.mean([x["app"] for x in rr],0),
   "subset":rr[0]["subset"],"key":rr[0]["key"],"n_maps":len(rr)})
 rows.sort(key=lambda r:tuple(r["true"]));assert len(rows)==92,len(rows)
 T=np.stack([r["true"] for r in rows]);Pie=np.stack([r["pie"] for r in rows]);App=np.stack([r["app"] for r in rows]);present=T>0
 cand=np.divide(App,Pie,out=np.full_like(App,np.nan),where=(Pie>=.03)&present)
 total=np.exp(np.nanmedian(np.log(np.where(cand>0,cand,np.nan)),axis=1));Raw=Pie*total[:,None]
 keys=[r["key"] for r in rows];Off=np.zeros_like(T);foldid=np.zeros(len(rows),int);offsets=[]
 for fi,held in enumerate(folds(keys),1):
  te=np.array([k in held for k in keys]);tr=~te;mult=[];true_total=T.sum(1)
  for j in range(3):
   ok=tr&present[:,j]&np.isfinite(cand[:,j])&(cand[:,j]>0)
   mult.append(np.exp(np.median(np.log(true_total[ok])-np.log(cand[ok,j]))))
  adj=cand[te]*np.asarray(mult)[None,:];tt=np.exp(np.nanmedian(np.log(np.where(adj>0,adj,np.nan)),axis=1))
  Off[te]=Pie[te]*tt[:,None];foldid[te]=fi;offsets.append({"fold":fi,**{SUBS[j]:float(mult[j]) for j in range(3)}})
 methods={"single_calibration_direct":App,"MLP_ratio_constrained":Raw,"MLP_ratio_constrained_response_offset_CV":Off}
 masks={"binary":np.array([r["subset"]=="binary" for r in rows]),"ternary":np.array([r["subset"]=="ternary" for r in rows]),"all":np.ones(len(rows),bool)}
 summary={m:{s:metrics(T[z],p[z]) for s,z in masks.items()} for m,p in methods.items()}
 fields=["condition","subset","n_maps","fold"]
 prefixes=["true","mlp_ratio_pct","single_apparent_uM","candidate_total_uM","ratio_constrained_uM","response_offset_cv_uM"]
 for x in prefixes:fields += [f"{x}_{s}" for s in SUBS]
 fields += ["true_total_uM","ratio_constrained_total_uM","response_offset_cv_total_uM","euclidean_raw_uM","euclidean_offset_cv_uM"]
 with (OUT/"mlp_concentration_axis_92_predictions.csv").open("w",newline="",encoding="utf-8-sig") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for i,r in enumerate(rows):
   q={"condition":r["condition"],"subset":r["subset"],"n_maps":r["n_maps"],"fold":int(foldid[i])}
   vals={"true":T[i],"mlp_ratio_pct":Pie[i]*100,"single_apparent_uM":App[i],"candidate_total_uM":cand[i],"ratio_constrained_uM":Raw[i],"response_offset_cv_uM":Off[i]}
   for pre,a in vals.items():
    for j,s in enumerate(SUBS):q[f"{pre}_{s}"]=float(a[j]) if np.isfinite(a[j]) else ""
   q.update({"true_total_uM":T[i].sum(),"ratio_constrained_total_uM":Raw[i].sum(),"response_offset_cv_total_uM":Off[i].sum(),"euclidean_raw_uM":np.linalg.norm(Raw[i]-T[i]),"euclidean_offset_cv_uM":np.linalg.norm(Off[i]-T[i])});w.writerow(q)
 sr=[]
 for m,sets in summary.items():
  for s,v in sets.items():sr.append({"method":m,"subset":s,**v})
 with (OUT/"mlp_concentration_axis_92_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
  w=csv.DictWriter(f,fieldnames=list(sr[0]));w.writeheader();w.writerows(sr)
 result={"protocol":{"n_conditions":92,"binary_n":int(masks["binary"].sum()),"ternary_n":int(masks["ternary"].sum()),"exclusion":"nonzero max/min >=100","composition":"stored four-class MLP condition-LOO","concentration":"single apparent concentration; total=geometric median(apparent_i/MLP ratio_i)","offset":"train-ratio-fold component scalar only; no neural concentration corrector"},"calibration_ab":AB.tolist(),"spectral_transfer":diag,"response_multipliers_by_fold":offsets,"summary":summary}
 (OUT/"mlp_concentration_axis_92_results.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf8")
 print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
