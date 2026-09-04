"""Leakage-controlled autoresearch loop for the 92-condition composition MLP."""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
CACHE=ROOT/"documentation/results/four_class_binary_ternary_20260825/four_class_102maps_cache.npz"
OUT=ROOT/"documentation/results/mlp_hparam_92_20260825"
SUBS=("DQ","TBZ","THI")

def ratio_key(c):
 c=np.asarray(c,float); s=c.sum(); return tuple(np.round(c/s,7))

def grouped_folds(map_rows,n=5,seed=825):
 keys=[]; kinds={}
 for r in map_rows:
  k=ratio_key(r["conc"]); keys.append(k); kinds[k]=int((r["conc"]>0).sum())
 bins=[set() for _ in range(n)]; rng=np.random.default_rng(seed)
 for kind in (2,3):
  u=list(dict.fromkeys(k for k in keys if kinds[k]==kind)); u=[u[i] for i in rng.permutation(len(u))]
  for i,k in enumerate(u): bins[i%n].add(k)
 return bins

def load_data():
 z=np.load(CACHE,allow_pickle=True); X=z["X"].astype(np.float32);Y=z["Y"].astype(np.float32)
 M=z["maps"].astype(object);IM=z["is_mix"].astype(bool);C=z["conc"].astype(float)
 signal=np.log1p(np.expm1(np.clip(X,0,30)).sum(1)).astype(np.float32)
 rows=[]
 for name in dict.fromkeys(M.tolist()):
  idx=np.where(M==name)[0]; mix=bool(IM[idx[0]]); c=C[idx[0]]
  if mix:
   nz=c[c>0]
   if c.max()/nz.min()>=100: continue
  rows.append({"name":str(name),"idx":idx,"mix":mix,"conc":c.copy(),"target":Y[idx[0]].copy()})
 mixrows=[r for r in rows if r["mix"]]; blanks=[r for r in rows if not r["mix"]]
 assert len(mixrows)==93 and len({tuple(r['conc']) for r in mixrows})==92
 return X,Y,signal,mixrows,blanks

def choose_pixels(idx,signal,k,mode,seed):
 idx=np.asarray(idx); k=min(int(k),len(idx)); rng=np.random.default_rng(seed)
 if k==len(idx): return idx
 if mode=="random": return np.sort(rng.choice(idx,k,replace=False))
 order=idx[np.argsort(signal[idx])]; bins=np.array_split(order,8); out=[]
 base=k//8; rem=k%8
 for b,g in enumerate(bins):
  take=min(len(g),base+(b<rem));
  if take: out.extend(rng.choice(g,take,replace=False).tolist())
 if len(out)<k:
  left=np.setdiff1d(idx,np.asarray(out),assume_unique=False);out.extend(rng.choice(left,k-len(out),replace=False).tolist())
 return np.asarray(out,dtype=int)

class Net(nn.Module):
 def __init__(self,nin,nout,hidden,act,drop):
  super().__init__(); layers=[]; n=nin
  A=nn.GELU if act=="gelu" else nn.ReLU
  for i,h in enumerate(hidden):
   layers += [nn.Linear(n,h),nn.BatchNorm1d(h),A()]
   if drop>0: layers.append(nn.Dropout(drop if i==0 else drop/2))
   n=h
  layers.append(nn.Linear(n,nout));self.f=nn.Sequential(*layers)
 def forward(self,x):return self.f(x)

def pool_one(prob,sig,mode):
 if mode=="mean": return prob.mean(0)
 if mode=="median": return prob.median(0).values
 if mode=="trimmed10":
  n=len(prob); order=torch.argsort(sig); k=int(n*.1); keep=order[k:n-k] if n-2*k>0 else order; return prob[keep].mean(0)
 order=torch.argsort(sig); chunks=torch.tensor_split(order,min(8,len(order))); return torch.stack([prob[g].mean(0) for g in chunks if len(g)]).mean(0)

def map_loss(pred,target,kind,alpha):
 if kind=="ce": return -(target*torch.log(torch.clamp(pred,1e-7,1))).sum(-1)
 w=1+alpha*(1-target)
 if kind=="huber": return (w*F.smooth_l1_loss(pred,target,reduction="none",beta=.05)).sum(-1)
 return (w*(pred-target).abs()).sum(-1)

def train_fold(X,Y,signal,train_rows,test_rows,cfg,seed):
 torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(min(12,torch.get_num_threads()))
 net=Net(X.shape[1],4,tuple(cfg["hidden"]),cfg["act"],cfg["drop"])
 opt=torch.optim.AdamW(net.parameters(),lr=cfg["lr"],weight_decay=cfg["wd"])
 rng=np.random.default_rng(seed); train_groups=[]
 for i,r in enumerate(train_rows):
  px=choose_pixels(r["idx"],signal,cfg["pixels"],cfg["sample"],seed+101*i)
  train_groups.append((px,r["target"],2 if r["mix"] and (r["conc"]>0).sum()==2 else 3))
 for ep in range(cfg["epochs"]):
  order=rng.permutation(len(train_groups));net.train()
  for st in range(0,len(order),cfg["map_batch"]):
   losses=[]
   for gi in order[st:st+cfg["map_batch"]]:
    px,t,k=train_groups[gi]; xb=torch.tensor(X[px]); prob=torch.softmax(net(xb),1); target=torch.tensor(t)
    pooled=pool_one(prob,torch.tensor(signal[px]),cfg["train_pool"])
    weight=cfg["binary_weight"] if k==2 else 1.0
    loss=weight*map_loss(pooled[None],target[None],cfg["loss"],cfg["alpha"])[0]
    if cfg["pixel_aux"]>0: loss=loss+cfg["pixel_aux"]*map_loss(prob,target[None].expand_as(prob),cfg["loss"],cfg["alpha"]).mean()
    losses.append(loss)
   loss=torch.stack(losses).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
 preds=[];net.eval()
 with torch.no_grad():
  for r in test_rows:
   pp=[]
   for st in range(0,len(r["idx"]),256): pp.append(torch.softmax(net(torch.tensor(X[r["idx"][st:st+256]])),1))
   prob=torch.cat(pp); pooled=pool_one(prob,torch.tensor(signal[r["idx"]]),cfg["test_pool"]).numpy();preds.append((r,pooled))
 return preds

def aggregate(preds):
 by={}
 for r,p in preds: by.setdefault(tuple(r["conc"]),[]).append(p)
 T=[];P=[];kind=[]
 for c,ps in by.items():
  c=np.asarray(c,float); t=c/c.sum();q=np.mean(ps,0)[:3];q=np.clip(q,0,None);q=q/(q.sum()+1e-12)
  T.append(t);P.append(q);kind.append(int((c>0).sum()))
 return np.asarray(T),np.asarray(P),np.asarray(kind)

def metrics(T,P,kind):
 def one(mask):
  t=T[mask];p=P[mask];pos=t>0;rec=p[pos]/t[pos]
  return {"n":int(mask.sum()),"mae_pp":float(100*np.mean(abs(p-t))),"median_recovery_pct":float(100*np.median(rec)),"within80_120_pct":float(100*np.mean((rec>=.8)&(rec<=1.2))),"within1_5x_pct":float(100*np.mean((rec>=2/3)&(rec<=1.5))),"within2x_pct":float(100*np.mean((rec>=.5)&(rec<=2))),"known_total_mae_fraction":float(np.mean(abs(p[pos]-t[pos])))}
 return {"all":one(np.ones(len(T),bool)),"binary":one(kind==2),"ternary":one(kind==3)}

def configs():
 base={"hidden":[256,64],"act":"relu","drop":.15,"lr":3e-4,"wd":1e-3,"epochs":100,"pixels":64,"map_batch":8,"sample":"random","train_pool":"mean","test_pool":"mean","loss":"l1","alpha":2.,"pixel_aux":.05,"binary_weight":1.}
 changes=[
  ("B00_oldlike",{}),
  ("V01_maponly",{"pixel_aux":0.}),
  ("V02_bin8",{"pixel_aux":0.,"sample":"bin8","train_pool":"bin8","test_pool":"bin8"}),
  ("V03_trimtest",{"pixel_aux":0.,"sample":"bin8","train_pool":"bin8","test_pool":"trimmed10"}),
  ("V04_h128_32",{"hidden":[128,32],"pixel_aux":0.,"sample":"bin8","train_pool":"bin8","test_pool":"bin8"}),
  ("V05_h64_32",{"hidden":[64,32],"pixel_aux":0.,"sample":"bin8","train_pool":"bin8","test_pool":"bin8"}),
  ("V06_gelu128",{"hidden":[128,32],"act":"gelu","drop":.1,"pixel_aux":0.,"sample":"bin8","train_pool":"bin8","test_pool":"bin8"}),
  ("V07_gelu256",{"act":"gelu","drop":.1,"pixel_aux":0.,"sample":"bin8","train_pool":"bin8","test_pool":"bin8"}),
  ("V08_huber",{"hidden":[128,32],"act":"gelu","drop":.1,"pixel_aux":0.,"sample":"bin8","train_pool":"bin8","test_pool":"bin8","loss":"huber"}),
  ("V09_ce",{"hidden":[128,32],"act":"gelu","drop":.1,"pixel_aux":0.,"sample":"bin8","train_pool":"bin8","test_pool":"bin8","loss":"ce","alpha":0.}),
  ("V10_unweighted",{"hidden":[128,32],"act":"gelu","drop":.1,"pixel_aux":0.,"sample":"bin8","train_pool":"bin8","test_pool":"bin8","alpha":0.}),
  ("V11_binary15",{"hidden":[128,32],"act":"gelu","drop":.1,"pixel_aux":0.,"sample":"bin8","train_pool":"bin8","test_pool":"bin8","binary_weight":1.5}),
 ]
 out=[]
 for name,ch in changes:q=base.copy();q.update(ch);q["name"]=name;out.append(q)
 return out

def run_cfg(cfg,X,Y,signal,mixrows,blanks,folds,seed):
 allpred=[]
 for fi,held in enumerate(folds,1):
  te=[r for r in mixrows if ratio_key(r["conc"]) in held];tr=[r for r in mixrows if ratio_key(r["conc"]) not in held]+blanks
  allpred.extend(train_fold(X,Y,signal,tr,te,cfg,seed+fi*101))
 T,P,k=aggregate(allpred);return metrics(T,P,k),T,P,k

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=["screen","confirm"],default="screen");args=ap.parse_args()
 OUT.mkdir(parents=True,exist_ok=True);X,Y,signal,mixrows,blanks=load_data();folds=grouped_folds(mixrows)
 if args.stage=="screen": todo=configs();seeds=[1700]
 else:
  screen=json.loads((OUT/"screen_results.json").read_text(encoding="utf8")); top=[x["name"] for x in sorted(screen["experiments"],key=lambda x:(x["metrics"]["all"]["mae_pp"],x["metrics"]["binary"]["mae_pp"]))[:3]]
  base={q["name"]:q for q in configs()};todo=[]
  for n in top:
   for tag,lr,wd,ep in [("a",3e-4,1e-3,220),("b",1e-3,1e-3,180),("c",3e-4,1e-2,220)]:
    q=base[n].copy();q.update({"name":f"{n}_{tag}","lr":lr,"wd":wd,"epochs":ep});todo.append(q)
  seeds=[2700,3700,4700]
 experiments=[];t0=time.time()
 for ci,cfg in enumerate(todo,1):
  vals=[];preds=[]
  print(f"START {ci}/{len(todo)} {cfg['name']}",flush=True);ct=time.time()
  for seed in seeds:
   met,T,P,k=run_cfg(cfg,X,Y,signal,mixrows,blanks,folds,seed);vals.append(met);preds.append(P)
  keys=vals[0].keys();avg={s:{m:float(np.mean([v[s][m] for v in vals])) for m in vals[0][s]} for s in keys}
  experiments.append({"name":cfg["name"],"config":cfg,"seeds":seeds,"metrics":avg,"seed_metrics":vals})
  print(f"DONE {cfg['name']} all={avg['all']['mae_pp']:.3f} binary={avg['binary']['mae_pp']:.3f} ternary={avg['ternary']['mae_pp']:.3f} sec={time.time()-ct:.1f}",flush=True)
 payload={"protocol":{"conditions":92,"maps":93,"classes":[*SUBS,"BLK"],"split":"ratio-grouped 5-fold","features":"1290-channel log1p full spectrum","screening":"real NNLS-hit pixels only; no synthetic data","primary":"all-condition composition MAE (%p)","tie_break":"binary MAE then within80-120","stage":args.stage},"experiments":experiments,"elapsed_s":time.time()-t0}
 path=OUT/("screen_results.json" if args.stage=="screen" else "confirm_results.json");path.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf8")
 rows=[]
 for e in experiments:
  for s,m in e["metrics"].items():rows.append({"name":e["name"],"subset":s,**m})
 with (OUT/("screen_summary.csv" if args.stage=="screen" else "confirm_summary.csv")).open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(sorted([{"name":e["name"],**e["metrics"]["all"],"binary_mae":e["metrics"]["binary"]["mae_pp"]} for e in experiments],key=lambda x:x["mae_pp"]),indent=2),flush=True)
if __name__=="__main__":main()
