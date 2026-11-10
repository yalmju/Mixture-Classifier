"""Three-seed confirmation and OOF ensemble for the selected distillation weight."""
import json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
sys.path.insert(0,str(ROOT/"documentation/scripts"))
import tune_mlp_composition_92 as tm
import tune_mlp_pls_distill_92 as td
def main():
 X,Y,signal,mixrows,blanks=tm.load_data();tm.ratio_key=lambda c:tuple(np.asarray(c,float));folds=tm.grouped_folds(mixrows);seeds=[12100,13100,14100];vals=[];Ps=[];T0=k0=None;t0=time.time();cfg={'name':'distill_0.2','distill':.2}
 for si,seed in enumerate(seeds,1):
  out=[];ct=time.time();print(f'START seed {si}/3 {seed}',flush=True)
  for fi,held in enumerate(folds,1):
   te=[r for r in mixrows if tm.ratio_key(r['conc']) in held];tr=[r for r in mixrows if tm.ratio_key(r['conc']) not in held]+blanks;out.extend(td.train_fold(X,Y,signal,tr,te,cfg,seed+fi*101))
  T,P,k=tm.aggregate(out)
  if T0 is None:T0,k0=T,k
  else:assert np.allclose(T,T0) and np.array_equal(k,k0)
  met=tm.metrics(T,P,k);vals.append({'seed':seed,'metrics':met});Ps.append(P);print(f"DONE seed={seed} all={met['all']['mae_pp']:.3f} binary={met['binary']['mae_pp']:.3f} sec={time.time()-ct:.1f}",flush=True)
 avg={s:{m:float(np.mean([v['metrics'][s][m] for v in vals])) for m in vals[0]['metrics'][s]} for s in vals[0]['metrics']};ens=tm.metrics(T0,np.mean(Ps,axis=0),k0)
 payload={'protocol':{'conditions':92,'split':'absolute-condition-grouped 5-fold','distillation_weight':.2,'teacher':'train-fold-only PLSR8','inference':'MLP only per seed; ensemble averages three MLP outputs'},'seed_results':vals,'mean_single_model_metrics':avg,'three_seed_ensemble_metrics':ens,'elapsed_s':time.time()-t0};(tm.OUT/'confirm_distill02_condition_results.json').write_text(json.dumps(payload,indent=2),encoding='utf8');print(json.dumps(payload,indent=2),flush=True)
if __name__=='__main__':main()
