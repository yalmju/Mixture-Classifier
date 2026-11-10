"""Higher PLSR distillation weights after the low-weight screen."""
import json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
sys.path.insert(0,str(ROOT/"documentation/scripts"))
import tune_mlp_composition_92 as tm
import tune_mlp_pls_distill_92 as td
def main():
 X,Y,signal,mixrows,blanks=tm.load_data();tm.ratio_key=lambda c:tuple(np.asarray(c,float));folds=tm.grouped_folds(mixrows);ex=[];t0=time.time()
 for w in [.3,.5,1.0]:
  cfg={'name':f'TH_distill_{w:g}','distill':w};out=[];ct=time.time();print('START',cfg['name'],flush=True)
  for fi,held in enumerate(folds,1):
   te=[r for r in mixrows if tm.ratio_key(r['conc']) in held];tr=[r for r in mixrows if tm.ratio_key(r['conc']) not in held]+blanks;out.extend(td.train_fold(X,Y,signal,tr,te,cfg,11100+fi*101))
  T,P,k=tm.aggregate(out);met=tm.metrics(T,P,k);ex.append({'name':cfg['name'],'config':cfg,'metrics':met});print(f"DONE {cfg['name']} all={met['all']['mae_pp']:.3f} binary={met['binary']['mae_pp']:.3f} ternary={met['ternary']['mae_pp']:.3f} sec={time.time()-ct:.1f}",flush=True)
 payload={'protocol':{'conditions':92,'split':'absolute-condition-grouped 5-fold','teacher':'train-fold-only PLSR8','inference':'MLP only'},'experiments':ex,'elapsed_s':time.time()-t0};(tm.OUT/'distill_high_condition_results.json').write_text(json.dumps(payload,indent=2),encoding='utf8');print(json.dumps(sorted(ex,key=lambda e:e['metrics']['all']['mae_pp']),indent=2),flush=True)
if __name__=='__main__':main()
