"""Condition-grouped companion to tune_mlp_composition_92.py."""
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
sys.path.insert(0,str(ROOT/"documentation/scripts"))
import tune_mlp_composition_92 as tm

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=["screen","confirm"],default="screen");args=ap.parse_args()
 tm.OUT.mkdir(parents=True,exist_ok=True);X,Y,signal,mixrows,blanks=tm.load_data()
 tm.ratio_key=lambda c: tuple(np.asarray(c,float))
 folds=tm.grouped_folds(mixrows);base={q["name"]:q for q in tm.configs()}
 if args.stage=="screen":
  names=["B00_oldlike","V01_maponly","V02_bin8","V04_h128_32","V07_gelu256","V08_huber","V09_ce","V10_unweighted"]
  todo=[base[n] for n in names];seeds=[1700]
 else:
  old=json.loads((tm.OUT/"screen_condition_results.json").read_text(encoding="utf8"))
  top=[e["name"] for e in sorted(old["experiments"],key=lambda e:(e["metrics"]["all"]["mae_pp"],e["metrics"]["binary"]["mae_pp"]))[:2]]
  todo=[]
  for n in top:
   for tag,lr,wd,ep in [("a",3e-4,1e-3,220),("b",1e-3,1e-3,180)]:
    q=base[n].copy();q.update({"name":f"{n}_{tag}","lr":lr,"wd":wd,"epochs":ep});todo.append(q)
  seeds=[2700,3700,4700]
 ex=[];t0=time.time()
 for ci,cfg in enumerate(todo,1):
  vals=[];print(f"START {ci}/{len(todo)} {cfg['name']}",flush=True);ct=time.time()
  for seed in seeds:
   met,*_=tm.run_cfg(cfg,X,Y,signal,mixrows,blanks,folds,seed);vals.append(met)
  avg={s:{m:float(np.mean([v[s][m] for v in vals])) for m in vals[0][s]} for s in vals[0]}
  ex.append({"name":cfg["name"],"config":cfg,"seeds":seeds,"metrics":avg,"seed_metrics":vals})
  print(f"DONE {cfg['name']} all={avg['all']['mae_pp']:.3f} binary={avg['binary']['mae_pp']:.3f} ternary={avg['ternary']['mae_pp']:.3f} sec={time.time()-ct:.1f}",flush=True)
 payload={"protocol":{"conditions":92,"maps":93,"split":"absolute-condition-grouped 5-fold","features":"1290-channel log1p full spectrum","screening":"real NNLS-hit pixels only; no synthetic data","primary":"all-condition composition MAE (%p)","stage":args.stage},"experiments":ex,"elapsed_s":time.time()-t0}
 stem=f"{args.stage}_condition";(tm.OUT/f"{stem}_results.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf8")
 rows=[]
 for e in ex:
  for s,m in e["metrics"].items():rows.append({"name":e["name"],"subset":s,**m})
 with (tm.OUT/f"{stem}_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(sorted([{"name":e["name"],**e["metrics"]["all"],"binary_mae":e["metrics"]["binary"]["mae_pp"]} for e in ex],key=lambda x:x["mae_pp"]),indent=2),flush=True)
if __name__=="__main__":main()
