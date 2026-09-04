"""Train-only PLSR distillation for the full-spectrum per-pixel composition MLP."""
import csv,json,sys,time
from pathlib import Path
import numpy as np,torch
from sklearn.cross_decomposition import PLSRegression
ROOT=Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
sys.path.insert(0,str(ROOT/"documentation/scripts"))
import tune_mlp_composition_92 as tm

def train_fold(X,Y,signal,train_rows,test_rows,cfg,seed):
 ids=[]
 for i,r in enumerate(train_rows):ids.extend(tm.choose_pixels(r['idx'],signal,64,'random',seed+17*i).tolist())
 teacher=PLSRegression(n_components=8,max_iter=1000,tol=1e-7,scale=True).fit(X[ids].astype(float),Y[ids].astype(float));tq=teacher.predict(X.astype(float));tq=np.clip(tq,0,None);tq=tq/(tq.sum(1,keepdims=True)+1e-12);tq=torch.tensor(tq.astype(np.float32))
 torch.manual_seed(seed);net=tm.Net(X.shape[1],4,(256,64),'relu',.15);opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-3);rng=np.random.default_rng(seed)
 for ep in range(180):
  order=rng.permutation(len(train_rows));net.train()
  for st in range(0,len(order),8):
   batch=[train_rows[i] for i in order[st:st+8]];pxs=[tm.choose_pixels(r['idx'],signal,64,'random',int(rng.integers(2**31))) for r in batch];allpx=np.concatenate(pxs);prob=torch.softmax(net(torch.tensor(X[allpx])),1);losses=[];pos=0
   for r,px in zip(batch,pxs):
    p=prob[pos:pos+len(px)];pos+=len(px);target=torch.tensor(r['target']);pm=p.mean(0);loss=tm.map_loss(pm[None],target[None],'l1',2.)[0]+cfg['distill']*(p-tq[px]).abs().sum(1).mean();losses.append(loss)
   loss=torch.stack(losses).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
 out=[];net.eval()
 with torch.no_grad():
  for r in test_rows:
   pp=[]
   for st in range(0,len(r['idx']),256):pp.append(torch.softmax(net(torch.tensor(X[r['idx'][st:st+256]])),1))
   out.append((r,torch.cat(pp).mean(0).numpy()))
 return out

def main():
 tm.OUT.mkdir(parents=True,exist_ok=True);X,Y,signal,mixrows,blanks=tm.load_data();tm.ratio_key=lambda c:tuple(np.asarray(c,float));folds=tm.grouped_folds(mixrows);ex=[];t0=time.time()
 for wi,w in enumerate([.02,.05,.1,.2],1):
  cfg={'name':f'T{wi:02d}_distill_{w:g}','distill':w};out=[];ct=time.time();print('START',cfg['name'],flush=True)
  for fi,held in enumerate(folds,1):
   te=[r for r in mixrows if tm.ratio_key(r['conc']) in held];tr=[r for r in mixrows if tm.ratio_key(r['conc']) not in held]+blanks;out.extend(train_fold(X,Y,signal,tr,te,cfg,9100+fi*101))
  T,P,k=tm.aggregate(out);met=tm.metrics(T,P,k);ex.append({'name':cfg['name'],'config':cfg,'metrics':met});print(f"DONE {cfg['name']} all={met['all']['mae_pp']:.3f} binary={met['binary']['mae_pp']:.3f} ternary={met['ternary']['mae_pp']:.3f} sec={time.time()-ct:.1f}",flush=True)
 payload={'protocol':{'conditions':92,'split':'absolute-condition-grouped 5-fold','teacher':'train-fold-only PLSR8','inference':'MLP only; teacher not required','features':'1290-channel log1p full spectrum'},'experiments':ex,'elapsed_s':time.time()-t0};(tm.OUT/'distill_condition_results.json').write_text(json.dumps(payload,indent=2),encoding='utf8')
 rows=[]
 for e in ex:
  for s,m in e['metrics'].items():rows.append({'name':e['name'],'subset':s,**m})
 with (tm.OUT/'distill_condition_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(sorted(ex,key=lambda e:e['metrics']['all']['mae_pp']),indent=2),flush=True)
if __name__=='__main__':main()
