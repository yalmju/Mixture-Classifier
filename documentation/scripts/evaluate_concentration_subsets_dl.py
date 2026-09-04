"""Fixed DL05 evaluation for in-range ternary and binary concentration subsets."""
import csv,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.model_selection import KFold
from torch import nn
from torch.nn import functional as F
ROOT=Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
sys.path.insert(0,str(ROOT/"documentation/scripts"))
from run_mlp_concentration_axis_92 import folds,parse,rkey
CACHE=ROOT/"documentation/results/four_class_binary_ternary_20260825/four_class_102maps_cache.npz"
BASE=ROOT/"documentation/results/mlp_concentration_axis_92_20260825/mlp_concentration_axis_92_predictions.csv"
OUT=ROOT/"documentation/results/concentration_subsets_fixed_dl_20260825";SUBS=["DQ","TBZ","THI"]

class Net(nn.Module):
 def __init__(self,n):super().__init__();self.f=nn.Sequential(nn.Linear(n,32),nn.GELU(),nn.Dropout(.1),nn.Linear(32,16),nn.GELU(),nn.Dropout(.1),nn.Linear(16,1))
 def forward(self,x):return self.f(x).squeeze(1)
def score(t,p):
 q=p/t;l=np.log2(np.clip(q,1e-9,None));return {"n":len(t),"median_recovery_pct":100*np.median(q),"mean_recovery_pct":100*np.mean(q),"q1_pct":100*np.quantile(q,.25),"q3_pct":100*np.quantile(q,.75),"within_2x_pct":100*np.mean((q>=.5)&(q<=2)),"within_1_5x_pct":100*np.mean((q>=2/3)&(q<=1.5)),"within_1_25x_pct":100*np.mean((q>=.8)&(q<=1.25)),"mae_uM":np.mean(abs(p-t)),"rmse_log2":np.sqrt(np.mean(l*l)),"pearson_r":np.corrcoef(t,p)[0,1] if np.std(t)>0 and np.std(p)>0 else None}
def features(df):
 z=np.load(CACHE,allow_pickle=True);X=z["X"].astype(float);M=z["maps"].astype(object);IM=z["is_mix"].astype(bool);by={}
 for name in dict.fromkeys(M[IM].tolist()):
  c=parse(name)
  if c is None:continue
  a=X[IM&(M==name)];raw=np.expm1(a);edges=np.linspace(0,a.shape[1],33,dtype=int);med=np.median(a,0)
  b=np.array([med[edges[k]:edges[k+1]].mean() for k in range(32)]);st=np.c_[np.log1p(raw.sum(1)),np.log1p(np.linalg.norm(raw,axis=1)),np.log1p(raw.max(1))]
  by.setdefault(tuple(c),[]).append(np.r_[b,np.quantile(st,[.1,.25,.5,.75,.9],axis=0).ravel()])
 C=df[[f"true_{s}" for s in SUBS]].to_numpy(float);Pie=df[[f"mlp_ratio_pct_{s}" for s in SUBS]].to_numpy(float)/100;App=df[[f"single_apparent_uM_{s}" for s in SUBS]].to_numpy(float);Spec=np.stack([np.mean(by[tuple(c)],0) for c in C])
 R=[];Y=[];MI=[];JJ=[]
 for i in range(len(C)):
  ctx=np.r_[np.log2(np.clip(App[i],.1,None)),Pie[i],np.log2(App[i].sum()+.1),Spec[i]]
  for j in range(3):
   if C[i,j]<=0:continue
   R.append(np.r_[ctx,np.eye(3)[j],np.log2(App[i,j]),Pie[i,j]]);Y.append(np.log2(C[i,j])-np.log2(App[i,j]));MI.append(i);JJ.append(j)
 return C,Pie,App,np.asarray(R,np.float32),np.asarray(Y,np.float32),np.asarray(MI),np.asarray(JJ)
def split_masks(df,kind):
 n=len(df)
 if kind=="condition_5fold":
  ans=[]
  for _,te in KFold(5,shuffle=True,random_state=825).split(np.arange(n)):m=np.zeros(n,bool);m[te]=1;ans.append(m)
  return ans
 keys=[rkey(c) for c in df[[f"true_{s}" for s in SUBS]].to_numpy(float)]
 return [np.array([k in h for k in keys]) for h in folds(keys)]
def run(df,kind):
 C,Pie,App,X,y,mi,jj=features(df);out=np.zeros_like(y);fm=np.zeros(len(C),int)
 for fi,tem in enumerate(split_masks(df,kind),1):
  trm=~tem;tr=np.isin(mi,np.where(trm)[0]);te=np.isin(mi,np.where(tem)[0]);fm[tem]=fi;mu=X[tr].mean(0);sd=X[tr].std(0)+1e-5;xa=torch.tensor((X[tr]-mu)/sd);ya=torch.tensor(y[tr]);xb=torch.tensor((X[te]-mu)/sd);ens=[]
  for seed in range(3):
   torch.manual_seed(2000+fi*31+seed);net=Net(X.shape[1]);opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.01);best=None;bl=1e9
   for ep in range(400):
    net.train();z=net(xa);loss=F.smooth_l1_loss(z,ya,beta=.25)+.01*(z*z).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
    if ep>=370 and loss.item()<bl:bl=loss.item();best={k:v.detach().clone() for k,v in net.state_dict().items()}
   net.load_state_dict(best);net.eval()
   with torch.no_grad():ens.append(net(xb).numpy())
  out[te]=np.median(np.stack(ens),0)
 pred=App[mi,jj]*2**out;true=C[mi,jj];detail=[]
 for k,(m,j) in enumerate(zip(mi,jj)):detail.append({"condition":df.iloc[m].condition,"fold":int(fm[m]),"component":SUBS[j],"true_uM":true[k],"single_calibration_uM":App[m,j],"dl_uM":pred[k]})
 return score(true,App[mi,jj]),score(true,pred),detail
def main():
 torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True);d=pd.read_csv(BASE);tern=d[(d.subset=="ternary")&(d[[f"true_{s}" for s in SUBS]]<=100).all(axis=1)];binary=d[d.subset=="binary"]
 result={};details=[]
 for subset,df in [("ternary_inrange_67",tern),("binary_21",binary)]:
  result[subset]={"n_conditions":len(df)}
  for split in ["condition_5fold","ratio_grouped_5fold"]:
   cal,dl,det=run(df.reset_index(drop=True),split);result[subset][split]={"single_calibration":cal,"label_supervised_DL05":dl}
   for r in det:r.update({"subset":subset,"split":split});details.append(r)
 with (OUT/"heldout_predictions.csv").open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
 rows=[]
 for s,v in result.items():
  for split,z in v.items():
   if not isinstance(z,dict):continue
   for method,q in z.items():rows.append({"subset":s,"split":split,"method":method,**q})
 with (OUT/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (OUT/"results.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf8");print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
