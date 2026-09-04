"""Dynamic pixel-resampling MLP search on condition-grouped 92-condition CV."""
import csv,json,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
sys.path.insert(0,str(ROOT/"documentation/scripts"))
import tune_mlp_composition_92 as tm

def train_dynamic(X,Y,signal,train_rows,test_rows,cfg,seed):
 torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(min(12,torch.get_num_threads()))
 net=tm.Net(X.shape[1],4,tuple(cfg['hidden']),cfg['act'],cfg['drop']);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['wd']);rng=np.random.default_rng(seed)
 for ep in range(cfg['epochs']):
  order=rng.permutation(len(train_rows));net.train()
  for st in range(0,len(order),cfg['map_batch']):
   batch=[train_rows[i] for i in order[st:st+cfg['map_batch']]];pxs=[tm.choose_pixels(r['idx'],signal,cfg['pixels'],cfg['sample'],int(rng.integers(2**31))) for r in batch]
   allpx=np.concatenate(pxs);allprob=torch.softmax(net(torch.tensor(X[allpx])),1);losses=[];pos=0
   for r,px in zip(batch,pxs):
    prob=allprob[pos:pos+len(px)];pos+=len(px);target=torch.tensor(r['target']);pooled=tm.pool_one(prob,torch.tensor(signal[px]),cfg['train_pool'])
    kind=2 if r['mix'] and (r['conc']>0).sum()==2 else 3;weight=cfg['binary_weight'] if kind==2 else 1.
    loss=weight*tm.map_loss(pooled[None],target[None],cfg['loss'],cfg['alpha'])[0]
    if cfg['pixel_aux']>0:loss=loss+cfg['pixel_aux']*tm.map_loss(prob,target[None].expand_as(prob),cfg['loss'],cfg['alpha']).mean()
    losses.append(loss)
   loss=torch.stack(losses).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
 out=[];net.eval()
 with torch.no_grad():
  for r in test_rows:
   pp=[]
   for st in range(0,len(r['idx']),256):pp.append(torch.softmax(net(torch.tensor(X[r['idx'][st:st+256]])),1))
   prob=torch.cat(pp);out.append((r,tm.pool_one(prob,torch.tensor(signal[r['idx']]),cfg['test_pool']).numpy()))
 return out

def cfgs():
 b={'hidden':[256,64],'act':'relu','drop':.15,'lr':3e-4,'wd':1e-3,'epochs':300,'pixels':32,'map_batch':4,'sample':'random','train_pool':'mean','test_pool':'mean','loss':'l1','alpha':2.,'pixel_aux':.05,'binary_weight':1.}
 ch=[('D00_old_dynamic',{}),('D01_maponly300',{'pixel_aux':0.}),('D02_maponly_lr1e3',{'pixel_aux':0.,'lr':1e-3,'epochs':180}),('D03_maponly64',{'pixel_aux':0.,'lr':1e-3,'epochs':180,'pixels':64,'map_batch':8}),('D04_aux001',{'pixel_aux':.01,'lr':1e-3,'epochs':180,'pixels':64,'map_batch':8}),('D05_huber128',{'hidden':[128,32],'act':'gelu','drop':.1,'lr':1e-3,'epochs':180,'pixels':64,'map_batch':8,'sample':'bin8','train_pool':'bin8','test_pool':'bin8','loss':'huber','pixel_aux':0.}),('D06_huber256',{'act':'gelu','drop':.1,'lr':1e-3,'epochs':180,'pixels':64,'map_batch':8,'sample':'bin8','train_pool':'bin8','test_pool':'bin8','loss':'huber','pixel_aux':0.})]
 out=[]
 for n,u in ch:q=b.copy();q.update(u);q['name']=n;out.append(q)
 return out

def main():
 tm.OUT.mkdir(parents=True,exist_ok=True);X,Y,signal,mixrows,blanks=tm.load_data();tm.ratio_key=lambda c:tuple(np.asarray(c,float));folds=tm.grouped_folds(mixrows);ex=[];t0=time.time()
 for ci,cfg in enumerate(cfgs(),1):
  print(f"START {ci}/7 {cfg['name']}",flush=True);ct=time.time();out=[]
  for fi,held in enumerate(folds,1):
   te=[r for r in mixrows if tm.ratio_key(r['conc']) in held];tr=[r for r in mixrows if tm.ratio_key(r['conc']) not in held]+blanks;out.extend(train_dynamic(X,Y,signal,tr,te,cfg,7100+fi*101))
  T,P,k=tm.aggregate(out);met=tm.metrics(T,P,k);ex.append({'name':cfg['name'],'config':cfg,'metrics':met});print(f"DONE {cfg['name']} all={met['all']['mae_pp']:.3f} binary={met['binary']['mae_pp']:.3f} ternary={met['ternary']['mae_pp']:.3f} sec={time.time()-ct:.1f}",flush=True)
 payload={'protocol':{'conditions':92,'split':'absolute-condition-grouped 5-fold','pixel_sampling':'dynamic each epoch','features':'1290-channel log1p full spectrum'},'experiments':ex,'elapsed_s':time.time()-t0};(tm.OUT/'dynamic_condition_results.json').write_text(json.dumps(payload,indent=2),encoding='utf8')
 rows=[]
 for e in ex:
  for s,m in e['metrics'].items():rows.append({'name':e['name'],'subset':s,**m})
 with (tm.OUT/'dynamic_condition_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(sorted(ex,key=lambda e:e['metrics']['all']['mae_pp']),indent=2),flush=True)
if __name__=='__main__':main()
