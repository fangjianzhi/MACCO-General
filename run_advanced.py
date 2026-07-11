import argparse,csv,time
from pathlib import Path
import numpy as np
from advanced_benchmarks import make_advanced_suite
from macco import minimize,minimize_delayed_hybrid,minimize_hybrid,minimize_low_rank
from strong_baselines import cma_es,lshade

ALGS={"MACCO":minimize,"MACCO_LR":minimize_low_rank,"MACCO_HLR":minimize_hybrid,"CMA_ES":cma_es,"L_SHADE":lshade}
ALGS["MACCO_DHLR"]=minimize_delayed_hybrid


def main():
 p=argparse.ArgumentParser(); p.add_argument("--dimensions",nargs="+",type=int,default=[30,50,100]); p.add_argument("--runs",type=int,default=20)
 p.add_argument("--algorithms",nargs="+",default=list(ALGS),choices=list(ALGS))
 p.add_argument("--population",type=int,default=40); p.add_argument("--budget-factor",type=int,default=1000); p.add_argument("--seed",type=int,default=20260711)
 p.add_argument("--output",type=Path,default=Path("advanced_results")); p.add_argument("--fresh",action="store_true",help="ignore an existing checkpoint")
 a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); raw_path=a.output/"raw_results.csv"; rows=[]
 if raw_path.exists() and not a.fresh:
  with raw_path.open(encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
  for r in rows:
   r.update(dimension=int(r['dimension']),run=int(r['run']),seed=int(r['seed']),best_f=float(r['best_f']),evaluations=int(r['evaluations']),seconds=float(r['seconds']))
 completed={(r['dimension'],r['function'],r['algorithm'],r['run']) for r in rows}
 selected={name:ALGS[name] for name in a.algorithms}; total=sum(len(make_advanced_suite(d))*len(selected)*a.runs for d in a.dimensions); done=0
 for di,d in enumerate(a.dimensions):
  suite=make_advanced_suite(d); budget=a.budget_factor*d
  for fi,(name,(fn,lb,ub)) in enumerate(suite.items()):
   for aname,opt in selected.items():
    for run in range(a.runs):
     if (d,name,aname,run) in completed: continue
     seed=a.seed+10_000_000*di+100_000*fi+run; start=time.perf_counter(); r=opt(fn,d,lb,ub,population_size=a.population,max_evaluations=budget,seed=seed)
     rows.append(dict(dimension=d,function=name,algorithm=aname,run=run,seed=seed,best_f=r.best_f,evaluations=r.evaluations,seconds=time.perf_counter()-start)); done+=1
     print(f"[{done}/{total}] D={d} {name:<15} {aname:<8} {r.best_f:.4e}",flush=True)
     with raw_path.open("w",newline="",encoding="utf-8-sig") as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
 summary=[]
 for d in a.dimensions:
  for name in make_advanced_suite(d):
   for aname in selected:
    g=[r for r in rows if r['dimension']==d and r['function']==name and r['algorithm']==aname]; v=np.array([r['best_f'] for r in g])
    summary.append(dict(dimension=d,function=name,algorithm=aname,mean=v.mean(),std=v.std(ddof=1) if len(v)>1 else 0,median=np.median(v),best=v.min(),worst=v.max(),mean_seconds=np.mean([r['seconds'] for r in g])))
 with (a.output/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f: w=csv.DictWriter(f,fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)


if __name__=="__main__": main()
