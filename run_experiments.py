"""Ablation, dimension-scaling, and equal-budget baseline experiments."""

import argparse,csv,time
from pathlib import Path
import numpy as np

from benchmark_suite import FUNCTIONS
from baselines import cbo,de,pso
from macco import minimize,minimize_delayed_hybrid,minimize_hybrid,minimize_low_rank


def macco_no_restart(*args,**kwargs):
    kwargs["restart_fraction"]=0.0
    return minimize(*args,**kwargs)


ALGORITHMS={"MACCO":minimize,"MACCO_LR":minimize_low_rank,"MACCO_HLR":minimize_hybrid,
            "MACCO_DHLR":minimize_delayed_hybrid,
            "MACCO_NO_RESTART":macco_no_restart,"DE":de,"PSO":pso,"CBO":cbo}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dimensions",nargs="+",type=int,default=[10,30,50,100])
    p.add_argument("--runs",type=int,default=20); p.add_argument("--population",type=int,default=40)
    p.add_argument("--budget-factor",type=int,default=1000,help="budget = factor * dimension")
    p.add_argument("--fixed-budget",type=int,default=None)
    p.add_argument("--functions",nargs="+",default=list(FUNCTIONS))
    p.add_argument("--algorithms",nargs="+",default=list(ALGORITHMS))
    p.add_argument("--seed",type=int,default=20260711); p.add_argument("--output",type=Path,default=Path("experiment_results"))
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); rows=[]
    total=len(a.dimensions)*len(a.functions)*len(a.algorithms)*a.runs; done=0
    for di,dim in enumerate(a.dimensions):
      budget=a.fixed_budget or a.budget_factor*dim
      for fi,name in enumerate(a.functions):
        function,lb,ub=FUNCTIONS[name]
        for aname in a.algorithms:
          optimizer=ALGORITHMS[aname]
          for run in range(a.runs):
            seed=a.seed+10_000_000*di+100_000*fi+run; start=time.perf_counter()
            result=optimizer(function,dim,lb,ub,population_size=a.population,max_evaluations=budget,seed=seed)
            rows.append(dict(dimension=dim,function=name,algorithm=aname,run=run,seed=seed,
                best_f=result.best_f,evaluations=result.evaluations,restarts=result.restarts,seconds=time.perf_counter()-start))
            done+=1; print(f"[{done}/{total}] D={dim:<3} {name:<11} {aname:<16} {result.best_f:.5e}",flush=True)
    raw=a.output/"raw_results.csv"
    with raw.open("w",newline="",encoding="utf-8-sig") as f:
      w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summary=[]
    for dim in a.dimensions:
      for name in a.functions:
        for aname in a.algorithms:
          g=[r for r in rows if r["dimension"]==dim and r["function"]==name and r["algorithm"]==aname]
          v=np.array([r["best_f"] for r in g]); sec=np.array([r["seconds"] for r in g])
          summary.append(dict(dimension=dim,function=name,algorithm=aname,mean=np.mean(v),
            std=np.std(v,ddof=1) if len(v)>1 else 0,median=np.median(v),best=np.min(v),worst=np.max(v),
            mean_seconds=np.mean(sec),mean_restarts=np.mean([r["restarts"] for r in g])))
    with (a.output/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
      w=csv.DictWriter(f,fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)


if __name__=="__main__": main()
