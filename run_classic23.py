"""Checkpointed equal-budget runner for the supplied classic F1--F23 suite."""

import argparse, csv, time
from pathlib import Path
import numpy as np

from classic23_benchmarks import make_classic23_suite
from macco import minimize, minimize_delayed_hybrid, minimize_low_rank
from strong_baselines import cma_es, lshade

ALGS = {"MACCO": minimize, "MACCO_LR": minimize_low_rank,
        "MACCO_DHLR": minimize_delayed_hybrid,
        "CMA_ES": cma_es, "L_SHADE": lshade}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--population", type=int, default=40)
    p.add_argument("--budget-factor", type=int, default=1000)
    p.add_argument("--fixed-budget", type=int)
    p.add_argument("--functions", nargs="+", default=[f"F{i}" for i in range(1,24)])
    p.add_argument("--algorithms", nargs="+", choices=list(ALGS), default=list(ALGS))
    p.add_argument("--seed", type=int, default=20260711)
    p.add_argument("--noisy-f7", action="store_true")
    p.add_argument("--output", type=Path, default=Path("classic23_results"))
    p.add_argument("--fresh", action="store_true")
    a = p.parse_args(); suite = make_classic23_suite(a.noisy_f7, a.seed)
    a.output.mkdir(parents=True, exist_ok=True); raw = a.output / "raw_results.csv"
    rows = []
    if raw.exists() and not a.fresh:
        with raw.open(encoding="utf-8-sig") as f: rows = list(csv.DictReader(f))
    completed = {(r["function"],r["algorithm"],int(r["run"])) for r in rows}
    total = len(a.functions)*len(a.algorithms)*a.runs; done = len(completed)
    for fi, name in enumerate(a.functions):
        fn, lb, ub, dim = suite[name]; budget = a.fixed_budget or a.budget_factor*dim
        for aname in a.algorithms:
            for run in range(a.runs):
                if (name,aname,run) in completed: continue
                seed=a.seed+100_000*fi+run; start=time.perf_counter()
                r=ALGS[aname](fn,dim,lb,ub,population_size=a.population,
                              max_evaluations=budget,seed=seed)
                rows.append(dict(function=name,dimension=dim,algorithm=aname,
                    run=run,seed=seed,best_f=r.best_f,evaluations=r.evaluations,
                    seconds=time.perf_counter()-start)); done+=1
                print(f"[{done}/{total}] {name:<3} D={dim:<2} {aname:<11} {r.best_f:.5e}",flush=True)
                with raw.open("w",newline="",encoding="utf-8-sig") as f:
                    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    summary=[]
    for name in a.functions:
        for aname in a.algorithms:
            g=[r for r in rows if r["function"]==name and r["algorithm"]==aname]
            v=np.array([float(r["best_f"]) for r in g]); sec=np.array([float(r["seconds"]) for r in g])
            summary.append(dict(function=name,dimension=suite[name][3],algorithm=aname,
                mean=v.mean(),std=v.std(ddof=1) if len(v)>1 else 0,median=np.median(v),
                best=v.min(),worst=v.max(),mean_seconds=sec.mean()))
    with (a.output/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)


if __name__ == "__main__": main()
