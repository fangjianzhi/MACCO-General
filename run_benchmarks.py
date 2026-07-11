import argparse, csv, time
from pathlib import Path
import numpy as np

from benchmark_suite import FUNCTIONS
from macco import minimize


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dim",type=int,default=30); p.add_argument("--runs",type=int,default=20)
    p.add_argument("--population",type=int,default=40); p.add_argument("--budget",type=int,default=20000)
    p.add_argument("--functions",nargs="+",default=list(FUNCTIONS)); p.add_argument("--seed",type=int,default=20260711)
    p.add_argument("--output",type=Path,default=Path("results")); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for fi,name in enumerate(a.functions):
        function,lb,ub=FUNCTIONS[name]
        for run in range(a.runs):
            seed=a.seed+100000*fi+run; start=time.perf_counter()
            r=minimize(function,a.dim,lb,ub,population_size=a.population,max_evaluations=a.budget,seed=seed)
            rows.append(dict(function=name,run=run,seed=seed,best_f=r.best_f,evaluations=r.evaluations,
                             iterations=r.iterations,restarts=r.restarts,seconds=time.perf_counter()-start))
            print(f"{name:<11} run={run+1:>2}/{a.runs} best={r.best_f:.6e}",flush=True)
    with (a.output/"raw_results.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summary=[]
    for name in a.functions:
        v=np.array([r["best_f"] for r in rows if r["function"]==name]); t=np.array([r["seconds"] for r in rows if r["function"]==name])
        summary.append(dict(function=name,mean=np.mean(v),std=np.std(v,ddof=1) if len(v)>1 else 0,
                            median=np.median(v),best=np.min(v),worst=np.max(v),mean_seconds=np.mean(t)))
    with (a.output/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)


if __name__=="__main__": main()
