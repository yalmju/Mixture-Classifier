"""Leakage-controlled map-level total-concentration head benchmark."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

ROOT=Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
sys.path.insert(0,str(ROOT/"documentation/scripts"))
from run_mlp_concentration_axis_92 import norm,parse,rkey,folds,metrics
CACHE=ROOT/"documentation/results/four_class_binary_ternary_20260825/four_class_102maps_cache.npz"
PREV=ROOT/"documentation/results/mlp_concentration_axis_92_20260825/mlp_concentration_axis_92_predictions.csv"
OUT=ROOT/"documentation/results/mlp_concentration_axis_92_20260825"

def map_features(X,maps,mix,P):
 out={};edges=np.linspace(0,X.shape[1],65,dtype=int)
 for name in dict.fromkeys(maps[mix].tolist()):
  a=X[mix&(maps==name)].astype(float);raw=np.expm1(a)
  bins=np.stack([a[:,edges[k]:edges[k+1]].mean(1) for k in range(64)],1)
  f=list(np.quantile(bins,[.25,.5,.75],axis=0).ravel())
  stats=np.c_[np.log1p(raw.sum(1)),np.log1p(np.linalg.norm(raw,axis=1)),np.log1p(raw.max(1))]
  f += list(np.quantile(stats,[.1,.25,.5,.75,.9],axis=0).ravel())
  B=[]
  for y in raw:B.append(nnls(P.T,np.clip(y,0,None))[0])
  f += list(np.quantile(np.asarray(B),[.1,.25,.5,.75,.9],axis=0).ravel())
  out[str(name)]=np.asarray(f,float)
 return out

def main():
 z=np.load(CACHE,allow_pickle=True);maps=z["maps"].astype(object);mix=z["is_mix"].astype(bool)
 feats=map_features(z["X"],maps,mix,z["P"].astype(float))
 # condition-level truth/MLP ratios and pooling of duplicate maps
 rows=list(csv.DictReader(open(PREV,encoding="utf-8-sig")));T=[];Pie=[];subs=[];keys=[];F=[];conds=[]
 for r in rows:
  c=np.array([float(r[f"true_{s}"]) for s in ["DQ","TBZ","THI"]]);p=np.array([float(r[f"mlp_ratio_pct_{s}"])/100 for s in ["DQ","TBZ","THI"]])
  same=[v for n,v in feats.items() if np.allclose(parse(n),c)]
  T.append(c);Pie.append(p);subs.append(r["subset"]);keys.append(rkey(c));F.append(np.mean(same,0));conds.append(r["condition"])
 T=np.asarray(T);Pie=np.asarray(Pie);F=np.asarray(F);y=np.log2(T.sum(1));keys=np.asarray(keys,object)
 pred={n:np.zeros(len(T)) for n in ["Ridge","PLS5","RandomForest","ExtraTrees","HistGradientBoosting","MLP_32_16"]};foldid=np.zeros(len(T),int)
 for fi,held in enumerate(folds(keys.tolist()),1):
  te=np.array([k in held for k in keys]);tr=~te;foldid[te]=fi
  models={
   "Ridge":make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-3,4,16))),
   "PLS5":make_pipeline(StandardScaler(),PLSRegression(n_components=5,scale=False,max_iter=1000)),
   "RandomForest":RandomForestRegressor(n_estimators=500,min_samples_leaf=2,max_features=.7,random_state=100+fi,n_jobs=-1),
   "ExtraTrees":ExtraTreesRegressor(n_estimators=500,min_samples_leaf=2,max_features=.7,random_state=200+fi,n_jobs=-1),
   "HistGradientBoosting":HistGradientBoostingRegressor(max_iter=250,max_leaf_nodes=7,l2_regularization=2,random_state=300+fi),
   "MLP_32_16":make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(32,16),activation="tanh",alpha=1.0,max_iter=2000,early_stopping=True,validation_fraction=.2,random_state=400+fi))}
  for n,m in models.items():
   m.fit(F[tr],y[tr]);q=np.asarray(m.predict(F[te])).reshape(-1);pred[n][te]=2**q
 summary={};masks={"binary":np.asarray(subs)=="binary","ternary":np.asarray(subs)=="ternary","all":np.ones(len(T),bool)}
 details=[]
 for n,total in pred.items():
  C=Pie*total[:,None];summary[n]={s:metrics(T[m],C[m]) for s,m in masks.items()}
  for i in range(len(T)):details.append({"condition":conds[i],"subset":subs[i],"fold":int(foldid[i]),"model":n,"true_total_uM":T[i].sum(),"pred_total_uM":total[i],"true_DQ":T[i,0],"pred_DQ":C[i,0],"true_TBZ":T[i,1],"pred_TBZ":C[i,1],"true_THI":T[i,2],"pred_THI":C[i,2]})
 with (OUT/"label_learned_total_head_92_predictions.csv").open("w",newline="",encoding="utf-8-sig") as f:
  w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
 sr=[]
 for n,ss in summary.items():
  for s,v in ss.items():sr.append({"model":n,"subset":s,**v})
 with (OUT/"label_learned_total_head_92_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
  w=csv.DictWriter(f,fieldnames=list(sr[0]));w.writeheader();w.writerows(sr)
 (OUT/"label_learned_total_head_92_results.json").write_text(json.dumps({"protocol":{"split":"5-fold held-out normalized-ratio groups","target":"log2 total concentration","composition":"stored four-class MLP OOF ratio","features":"map-level 64-bin log-spectrum quantiles + absolute signal quantiles + NNLS amplitude quantiles"},"summary":summary},indent=2,ensure_ascii=False),encoding="utf8")
 print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
