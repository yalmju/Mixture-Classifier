import pickle,re,sys
from pathlib import Path
import numpy as np
ROOT=Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
sys.path.insert(0,str(ROOT/"documentation/scripts"))
import autoresearch_lowgrid64_dl as ar

data=list(ar.condition_features());d,C,Pm,App,X,y,mi,keys=data
pls=pickle.load(open(ROOT/"outputs/models_20260825/UNMIXR_PLS8_4class_102mixtures_20260825.dlm","rb"));by={}
for p,n in zip(pls["loo_eval"]["pred"],pls["loo_eval"]["paths"]):
 c=ar.parse(Path(str(n)).name)
 if c is not None:
  q=np.clip(np.asarray(p[:3],float),0,None);q=q/(q.sum()+1e-12);by.setdefault(tuple(c),[]).append(q)
Pp=np.stack([np.mean(by[tuple(c)],axis=0) for c in C]);Pp=Pp/(Pp.sum(1,keepdims=True)+1e-12)
cfg={"hidden":(32,16),"act":"gelu","drop":.1,"beta":.25,"prior":.01,"wd":.01,"lr":3e-3,"epochs":400,"ensemble":5}
for name,Pie in [("MLP_ratio",Pm),("PLS_ratio",Pp)]:
 xx=X.copy()
 for r,m in enumerate(mi):
  j=r%3;xx[r,3:6]=Pie[m];xx[r,-1]=Pie[m,j]
 q=list(data);q[2]=Pie;q[4]=xx
 _,_,s=ar.run_config(cfg,tuple(q));print(name,s)
