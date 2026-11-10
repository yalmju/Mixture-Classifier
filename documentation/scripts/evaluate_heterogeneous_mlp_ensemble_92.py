"""One unified ensemble: constant distill + linear decay + pure map-only MLP."""
import json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
sys.path.insert(0,str(ROOT/"documentation/scripts"))
import tune_mlp_composition_92 as tm
import tune_mlp_distill_schedule_92 as ts
import tune_mlp_dynamic_condition_92 as tdyn

def run_member(name,seed,X,Y,signal,mixrows,blanks,folds):
 out=[]
 for fi,held in enumerate(folds,1):
  te=[r for r in mixrows if tm.ratio_key(r['conc']) in held];tr=[r for r in mixrows if tm.ratio_key(r['conc']) not in held]+blanks
  if name in ('constant','linear'):out.extend(ts.train_fold(X,Y,signal,tr,te,name,seed+fi*101))
  else:
   cfg={'hidden':[256,64],'act':'relu','drop':.15,'lr':1e-3,'wd':1e-3,'epochs':180,'pixels':64,'map_batch':8,'sample':'random','train_pool':'mean','test_pool':'mean','loss':'l1','alpha':2.,'pixel_aux':0.,'binary_weight':1.}
   out.extend(tdyn.train_dynamic(X,Y,signal,tr,te,cfg,seed+fi*101))
 return tm.aggregate(out)

def main():
 tm.OUT.mkdir(parents=True,exist_ok=True);X,Y,signal,mixrows,blanks=tm.load_data();tm.ratio_key=lambda c:tuple(np.asarray(c,float));folds=tm.grouped_folds(mixrows);members=[];T0=k0=None;t0=time.time()
 for name,seed in [('constant',16100),('linear',17100),('maponly',18100)]:
  ct=time.time();print('START',name,flush=True);T,P,k=run_member(name,seed,X,Y,signal,mixrows,blanks,folds)
  if T0 is None:T0,k0=T,k
  else:assert np.allclose(T,T0) and np.array_equal(k,k0)
  met=tm.metrics(T,P,k);members.append({'name':name,'seed':seed,'metrics':met,'pred':P});print(f"DONE {name} all={met['all']['mae_pp']:.3f} binary={met['binary']['mae_pp']:.3f} sec={time.time()-ct:.1f}",flush=True)
 P=np.mean([m['pred'] for m in members],axis=0);ens=tm.metrics(T0,P,k0)
 payload={'protocol':{'conditions':92,'split':'absolute-condition-grouped 5-fold','ensemble':'equal mean of constant-distilled, linear-decay-distilled, and map-only full-spectrum MLPs','inference':'single unified ensemble; no condition-specific switch'},'members':[{k:v for k,v in m.items() if k!='pred'} for m in members],'ensemble_metrics':ens,'elapsed_s':time.time()-t0};(tm.OUT/'heterogeneous_ensemble_results.json').write_text(json.dumps(payload,indent=2),encoding='utf8');print(json.dumps(payload,indent=2),flush=True)
if __name__=='__main__':main()
