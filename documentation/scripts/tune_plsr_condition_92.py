"""Tune PLSR latent components on the exact condition-grouped 92-condition split."""
import json,sys,time
from pathlib import Path
import numpy as np
from sklearn.cross_decomposition import PLSRegression
ROOT=Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
sys.path.insert(0,str(ROOT/"documentation/scripts"))
import tune_mlp_composition_92 as tm

def main():
 X,Y,signal,mixrows,blanks=tm.load_data();tm.ratio_key=lambda c:tuple(np.asarray(c,float));folds=tm.grouped_folds(mixrows)
 results=[];t0=time.time()
 for nc in [2,4,6,8,10,12]:
  out=[];ct=time.time();print('START PLSR',nc,flush=True)
  for fi,held in enumerate(folds,1):
   te=[r for r in mixrows if tm.ratio_key(r['conc']) in held];tr=[r for r in mixrows if tm.ratio_key(r['conc']) not in held]+blanks
   ids=[]
   for i,r in enumerate(tr):ids.extend(tm.choose_pixels(r['idx'],signal,64,'random',8100+fi*101+i).tolist())
   model=PLSRegression(n_components=nc,max_iter=1000,tol=1e-7,scale=True).fit(X[ids].astype(float),Y[ids].astype(float))
   for r in te:
    q=model.predict(X[r['idx']].astype(float));q=np.clip(q,0,None);q=q/(q.sum(1,keepdims=True)+1e-12);out.append((r,q.mean(0)))
  T,P,k=tm.aggregate(out);met=tm.metrics(T,P,k);results.append({'n_components':nc,'metrics':met});print(f"DONE PLSR{nc} all={met['all']['mae_pp']:.3f} binary={met['binary']['mae_pp']:.3f} sec={time.time()-ct:.1f}",flush=True)
 payload={'protocol':{'conditions':92,'maps':93,'split':'absolute-condition-grouped 5-fold','train_pixels_per_map':64,'test_pool':'all-pixel mean','features':'1290-channel log1p full spectrum'},'experiments':results,'elapsed_s':time.time()-t0}
 outpath=ROOT/'documentation/results/mlp_hparam_92_20260825/plsr_condition_results.json';outpath.write_text(json.dumps(payload,indent=2),encoding='utf8');print(json.dumps(sorted(results,key=lambda x:x['metrics']['all']['mae_pp']),indent=2),flush=True)
if __name__=='__main__':main()
