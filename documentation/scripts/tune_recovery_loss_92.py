"""Recovery-aware composition-loss search on the fixed 92-condition protocol."""
import csv,json,sys,time
from pathlib import Path
import numpy as np,torch
from sklearn.cross_decomposition import PLSRegression
ROOT=Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
sys.path.insert(0,str(ROOT/"documentation/scripts"))
import tune_mlp_composition_92 as tm

def augment(T,P,k,met):
 for subset,mask in [('all',np.ones(len(T),bool)),('binary',k==2),('ternary',k==3)]:
  means=[]
  for j in range(3):
   z=mask&(T[:,j]>0);means.append(float(100*np.mean(P[z,j]/T[z,j])))
  met[subset]['component_mean_recovery_pct']=means;met[subset]['macro_abs_mean_bias_pct']=float(np.mean(np.abs(np.asarray(means)-100)))
 return met

def train_fold(X,Y,signal,train_rows,test_rows,cfg,seed):
 ids=[]
 for i,r in enumerate(train_rows):ids.extend(tm.choose_pixels(r['idx'],signal,64,'random',seed+17*i).tolist())
 teacher=PLSRegression(n_components=8,max_iter=1000,tol=1e-7,scale=True).fit(X[ids].astype(float),Y[ids].astype(float));tq=teacher.predict(X.astype(float));tq=np.clip(tq,0,None);tq=tq/(tq.sum(1,keepdims=True)+1e-12);tq=torch.tensor(tq.astype(np.float32))
 torch.manual_seed(seed);net=tm.Net(X.shape[1],4,(256,64),'relu',.15);opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-3);rng=np.random.default_rng(seed);eps=1e-3
 for _ in range(180):
  order=rng.permutation(len(train_rows));net.train()
  for st in range(0,len(order),8):
   batch=[train_rows[i] for i in order[st:st+8]];pxs=[tm.choose_pixels(r['idx'],signal,64,'random',int(rng.integers(2**31))) for r in batch];allpx=np.concatenate(pxs);prob=torch.softmax(net(torch.tensor(X[allpx])),1);losses=[];pools=[];targets=[];pos=0
   for r,px in zip(batch,pxs):
    p=prob[pos:pos+len(px)];pos+=len(px);target=torch.tensor(r['target']);pm=p.mean(0);base=tm.map_loss(pm[None],target[None],'l1',2.)[0];present=target[:3]>0
    rec=torch.abs(torch.log((pm[:3][present]+eps)/(target[:3][present]+eps))).mean() if present.any() else torch.tensor(0.)
    loss=base+cfg['recovery_weight']*rec+.2*(p-tq[px]).abs().sum(1).mean();losses.append(loss);pools.append(pm);targets.append(target)
   if cfg['bias_weight']>0:
    pp=torch.stack(pools)[:,:3];yy=torch.stack(targets)[:,:3];bias=torch.abs((pp-yy).mean(0)).sum();losses.append(cfg['bias_weight']*bias)
   loss=torch.stack(losses).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
 out=[];net.eval()
 with torch.no_grad():
  for r in test_rows:
   pp=[]
   for st in range(0,len(r['idx']),256):pp.append(torch.softmax(net(torch.tensor(X[r['idx'][st:st+256]])),1))
   out.append((r,torch.cat(pp).mean(0).numpy()))
 return out

def run(cfg,seed,X,Y,signal,mixrows,blanks,folds):
 out=[]
 for fi,held in enumerate(folds,1):
  te=[r for r in mixrows if tm.ratio_key(r['conc']) in held];tr=[r for r in mixrows if tm.ratio_key(r['conc']) not in held]+blanks;out.extend(train_fold(X,Y,signal,tr,te,cfg,seed+fi*101))
 T,P,k=tm.aggregate(out);return augment(T,P,k,tm.metrics(T,P,k)),T,P,k

def main():
 tm.OUT.mkdir(parents=True,exist_ok=True);X,Y,signal,mixrows,blanks=tm.load_data();tm.ratio_key=lambda c:tuple(np.asarray(c,float));folds=tm.grouped_folds(mixrows);cfgs=[('R00',0.,0.),('R01',.02,0.),('R02',.05,0.),('R03',.1,0.),('R04',.05,.1)];ex=[];t0=time.time()
 for name,rw,bw in cfgs:
  cfg={'name':name,'recovery_weight':rw,'bias_weight':bw};ct=time.time();print('START',name,cfg,flush=True);met,*_=run(cfg,19100,X,Y,signal,mixrows,blanks,folds);ex.append({'name':name,'config':cfg,'metrics':met});print(f"DONE {name} all={met['all']['mae_pp']:.3f} tight={met['all']['within80_120_pct']:.2f} bias={met['all']['macro_abs_mean_bias_pct']:.2f} binary={met['binary']['mae_pp']:.3f} sec={time.time()-ct:.1f}",flush=True)
 payload={'protocol':{'conditions':92,'split':'absolute-condition-grouped 5-fold','base':'weighted map L1 + train-fold PLSR8 distillation 0.2','recovery_loss':'mean absolute log(pred/true) over present analytes','selection':'maximize within80-120 then minimize macro bias without material MAE increase'},'experiments':ex,'elapsed_s':time.time()-t0};(tm.OUT/'recovery_loss_results.json').write_text(json.dumps(payload,indent=2),encoding='utf8')
 rows=[]
 for e in ex:
  for s,m in e['metrics'].items():rows.append({'name':e['name'],'subset':s,**{k:v for k,v in m.items() if not isinstance(v,list)}})
 with (tm.OUT/'recovery_loss_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(sorted(ex,key=lambda e:(-e['metrics']['all']['within80_120_pct'],e['metrics']['all']['macro_abs_mean_bias_pct'],e['metrics']['all']['mae_pp'])),indent=2),flush=True)
if __name__=='__main__':main()
