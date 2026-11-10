"""Patch the tuning prototype so BatchNorm sees a true multi-map batch."""
import sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
sys.path.insert(0,str(ROOT/"documentation/scripts"))
import tune_mlp_composition_92 as tm

def fixed_train_fold(X,Y,signal,train_rows,test_rows,cfg,seed):
 torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(min(12,torch.get_num_threads()))
 net=tm.Net(X.shape[1],4,tuple(cfg["hidden"]),cfg["act"],cfg["drop"])
 opt=torch.optim.AdamW(net.parameters(),lr=cfg["lr"],weight_decay=cfg["wd"])
 rng=np.random.default_rng(seed);groups=[]
 for i,r in enumerate(train_rows):
  px=tm.choose_pixels(r["idx"],signal,cfg["pixels"],cfg["sample"],seed+101*i)
  groups.append((px,r["target"],2 if r["mix"] and (r["conc"]>0).sum()==2 else 3))
 for _ in range(cfg["epochs"]):
  order=rng.permutation(len(groups));net.train()
  for st in range(0,len(order),cfg["map_batch"]):
   batch=[groups[i] for i in order[st:st+cfg["map_batch"]]];allpx=np.concatenate([g[0] for g in batch])
   allprob=torch.softmax(net(torch.tensor(X[allpx])),1);losses=[];pos=0
   for px,t,k in batch:
    prob=allprob[pos:pos+len(px)];pos+=len(px);target=torch.tensor(t)
    pooled=tm.pool_one(prob,torch.tensor(signal[px]),cfg["train_pool"]);weight=cfg["binary_weight"] if k==2 else 1.
    loss=weight*tm.map_loss(pooled[None],target[None],cfg["loss"],cfg["alpha"])[0]
    if cfg["pixel_aux"]>0:loss=loss+cfg["pixel_aux"]*tm.map_loss(prob,target[None].expand_as(prob),cfg["loss"],cfg["alpha"]).mean()
    losses.append(loss)
   loss=torch.stack(losses).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
 out=[];net.eval()
 with torch.no_grad():
  for r in test_rows:
   pp=[]
   for st in range(0,len(r["idx"]),256):pp.append(torch.softmax(net(torch.tensor(X[r["idx"][st:st+256]])),1))
   prob=torch.cat(pp);pooled=tm.pool_one(prob,torch.tensor(signal[r["idx"]]),cfg["test_pool"]).numpy();out.append((r,pooled))
 return out

tm.train_fold=fixed_train_fold
if __name__=="__main__":
 import tune_mlp_condition_92
 tune_mlp_condition_92.main()
