"""Small-data DL search for 64-condition concentration recovery.

All reported predictions are held out by normalized-ratio groups.  The target is
the log2 residual over the single-component calibration estimate.  Selection is
primary within-2x accuracy, then within-1.5x and MAE.
"""
import csv,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
sys.path.insert(0,str(ROOT/"documentation/scripts"))
from run_mlp_concentration_axis_92 import folds,parse,rkey
CACHE=ROOT/"documentation/results/four_class_binary_ternary_20260825/four_class_102maps_cache.npz"
BASE=ROOT/"documentation/results/mlp_concentration_axis_92_20260825/mlp_concentration_axis_92_predictions.csv"
OUT=ROOT/"documentation/results/lowgrid64_dl_autoresearch_20260825"
VALS={3.,6.,12.,24.}; SUBS=["DQ","TBZ","THI"]

def condition_features():
 d=pd.read_csv(BASE); ok=d[[f"true_{s}" for s in SUBS]].apply(lambda c:c.isin(VALS)).all(axis=1);d=d[ok].copy()
 z=np.load(CACHE,allow_pickle=True);X=z["X"].astype(float);M=z["maps"].astype(object);IM=z["is_mix"].astype(bool)
 feat={}
 for name in dict.fromkeys(M[IM].tolist()):
  c=parse(name)
  if c is None or not all(float(v) in VALS for v in c):continue
  a=X[IM&(M==name)];raw=np.expm1(a);edges=np.linspace(0,a.shape[1],33,dtype=int)
  med=np.median(a,axis=0);binned=np.array([med[edges[k]:edges[k+1]].mean() for k in range(32)])
  stats=np.c_[np.log1p(raw.sum(1)),np.log1p(np.linalg.norm(raw,axis=1)),np.log1p(raw.max(1))]
  sf=np.r_[binned,np.quantile(stats,[.1,.25,.5,.75,.9],axis=0).ravel()]
  feat.setdefault(tuple(c),[]).append(sf)
 C=d[[f"true_{s}" for s in SUBS]].to_numpy(float);Pie=d[[f"mlp_ratio_pct_{s}" for s in SUBS]].to_numpy(float)/100
 App=d[[f"single_apparent_uM_{s}" for s in SUBS]].to_numpy(float)
 Spec=np.stack([np.mean(feat[tuple(c)],axis=0) for c in C]); keys=[rkey(c) for c in C]
 # Component-row features: shared condition context + component identity/indexed physics.
 rows=[];target=[];map_index=[]
 for i in range(len(C)):
  context=np.r_[np.log2(np.clip(App[i],.1,None)),Pie[i],np.log2(App[i].sum()+.1),Spec[i]]
  for j in range(3):
   rows.append(np.r_[context,np.eye(3)[j],np.log2(App[i,j]),Pie[i,j]])
   target.append(np.log2(C[i,j])-np.log2(App[i,j]));map_index.append(i)
 return d,C,Pie,App,np.asarray(rows,np.float32),np.asarray(target,np.float32),np.asarray(map_index),keys

class Net(nn.Module):
 def __init__(self,nin,hidden,act,drop):
  super().__init__();A={"tanh":nn.Tanh,"relu":nn.ReLU,"gelu":nn.GELU}[act];layers=[];last=nin
  for h in hidden:
   layers += [nn.Linear(last,h),A()]
   if drop:layers += [nn.Dropout(drop)]
   last=h
  layers += [nn.Linear(last,1)];self.net=nn.Sequential(*layers)
 def forward(self,x):return self.net(x).squeeze(1)

def score(true,pred):
 q=pred/true;loge=np.log2(np.clip(q,1e-9,None))
 return {"within_2x_pct":100*np.mean((q>=.5)&(q<=2)),"within_1_5x_pct":100*np.mean((q>=2/3)&(q<=1.5)),
  "within_1_25x_pct":100*np.mean((q>=.8)&(q<=1.25)),"mae_uM":np.mean(abs(pred-true)),
  "rmse_log2":np.sqrt(np.mean(loge**2)),"median_recovery_pct":100*np.median(q),"pearson_r":np.corrcoef(true,pred)[0,1]}

def run_config(cfg,data):
 d,C,Pie,App,X,y,mi,keys=data;pred=np.zeros_like(y);foldid=np.zeros(len(C),int)
 for fi,held in enumerate(folds(keys),1):
  tem=np.array([k in held for k in keys]);trm=~tem;tr=np.isin(mi,np.where(trm)[0]);te=np.isin(mi,np.where(tem)[0]);foldid[tem]=fi
  mu=X[tr].mean(0);sd=X[tr].std(0)+1e-5;xa=torch.tensor((X[tr]-mu)/sd);ya=torch.tensor(y[tr]);xb=torch.tensor((X[te]-mu)/sd)
  ens=[]
  for seed in range(cfg["ensemble"]):
   torch.manual_seed(1000+fi*31+seed);net=Net(X.shape[1],cfg["hidden"],cfg["act"],cfg["drop"])
   opt=torch.optim.AdamW(net.parameters(),lr=cfg["lr"],weight_decay=cfg["wd"])
   best=None;bestloss=np.inf
   for ep in range(cfg["epochs"]):
    net.train();z=net(xa);loss=F.smooth_l1_loss(z,ya,beta=cfg["beta"])+cfg["prior"]*(z*z).mean()
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
    if ep>=cfg["epochs"]-30 and loss.item()<bestloss:bestloss=loss.item();best={k:v.detach().clone() for k,v in net.state_dict().items()}
   net.load_state_dict(best);net.eval()
   with torch.no_grad():ens.append(net(xb).numpy())
  pred[te]=np.median(np.stack(ens),axis=0)
 P=(App.reshape(-1)*2**pred).reshape(-1,3);return P,foldid,score(C.ravel(),P.ravel())

def main():
 torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True);data=condition_features();d,C,Pie,App,*_=data
 configs=[]
 for hidden,act,drop,beta,prior,wd in [
  ((16,8),"tanh",.1,.25,.02,.02),((32,16),"tanh",.1,.25,.02,.02),
  ((32,16,8),"tanh",.1,.25,.01,.01),((64,32),"tanh",.15,.25,.02,.02),
  ((32,16),"gelu",.1,.25,.01,.01),((64,32,16),"gelu",.15,.25,.02,.02),
  ((32,16),"relu",.1,.5,.01,.01),((64,32),"relu",.2,.5,.02,.02),
  ((16,8),"tanh",0,.5,0,.01),((32,16),"tanh",0,1.0,.005,.02),
  ((64,32),"gelu",.1,1.0,.005,.05),((16,),"tanh",.1,.25,.05,.05)]:
  configs.append({"hidden":hidden,"act":act,"drop":drop,"beta":beta,"prior":prior,"wd":wd,"lr":3e-3,"epochs":400,"ensemble":5})
 log=[];allpred={};foldid=None
 for i,cfg in enumerate(configs,1):
  P,fi,s=run_config(cfg,data);name=f"DL{i:02d}";allpred[name]=P;foldid=fi;log.append({"name":name,**cfg,**s});print(name,json.dumps(s),flush=True)
 log.sort(key=lambda x:(-x["within_2x_pct"],-x["within_1_5x_pct"],x["mae_uM"]));best=log[0]["name"]
 # prior best from the original 64-condition experiment
 prior={"name":"prior_neural_residual","within_2x_pct":84.8958333333,"within_1_5x_pct":67.7083333333,"within_1_25x_pct":43.75,"mae_uM":3.1660507324,"rmse_log2":.6727185132,"median_recovery_pct":99.8526852}
 with (OUT/"experiment_log.csv").open("w",newline="",encoding="utf-8-sig") as f:
  fields=["name","hidden","act","drop","beta","prior","wd","lr","epochs","ensemble","within_2x_pct","within_1_5x_pct","within_1_25x_pct","mae_uM","rmse_log2","median_recovery_pct","pearson_r"]
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(log)
 rows=[]
 for name,P in allpred.items():
  for i in range(len(C)):
   q={"condition":d.iloc[i]["condition"],"fold":int(foldid[i]),"model":name}
   for j,s in enumerate(SUBS):q[f"true_{s}"]=C[i,j];q[f"pred_{s}"]=P[i,j]
   rows.append(q)
 with (OUT/"heldout_predictions.csv").open("w",newline="",encoding="utf-8-sig") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 result={"protocol":{"conditions":64,"split":"normalized-ratio-grouped 5-fold","target":"log2 residual over single calibration","selection":"within2x, then within1.5x, then MAE","note":"configuration selection on the same fixed CV is exploratory"},"prior_baseline":prior,"best":log[0],"experiments":log}
 (OUT/"results.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,default=lambda x:list(x) if isinstance(x,tuple) else float(x)),encoding="utf8")
 print('BEST',best,json.dumps(log[0],default=lambda x:list(x) if isinstance(x,tuple) else float(x)),flush=True)
if __name__=="__main__":main()
