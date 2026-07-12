"""Checkpointed GWO-vs-MACCO diagnostic for origin, shift, and rotation bias."""

from __future__ import annotations
import argparse, csv, time
from pathlib import Path
import numpy as np

from advanced_benchmarks import make_advanced_suite
from baselines import gwo
from classic23_benchmarks import make_classic23_suite
from macco import minimize, minimize_subspace

ALGORITHMS={"GWO":gwo,"MACCO":minimize,"MACCO_SUBSPACE":minimize_subspace}


def make_suite(dim=30, seed=20260711):
    classic=make_classic23_suite()
    suite={name:(fn,lb,ub,d) for name,(fn,lb,ub,d) in classic.items()}
    advanced=make_advanced_suite(dim,seed)
    rr,lb,ub=advanced["rot_rastrigin"]
    rb,rlb,rub=advanced["rot_rosenbrock"]
    shift=np.random.default_rng(seed+2*dim).uniform(-0.35,0.35,dim)

    def shifted_rastrigin(x):
        z=x/5.12-shift
        return float(10*dim+np.sum(z**2-10*np.cos(2*np.pi*z)))

    suite["shifted_rastrigin"]=(shifted_rastrigin,-5.12,5.12,dim)
    suite["rotated_rastrigin"]=(rr,lb,ub,dim)
    suite["rotated_rosenbrock"]=(rb,rlb,rub,dim)
    return suite


def write_rows(path,rows):
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",type=int,default=20)
    p.add_argument("--population",type=int,default=30)
    p.add_argument("--budget",type=int,default=30_000)
    p.add_argument("--seed",type=int,default=20260711)
    p.add_argument("--functions",nargs="+",default=[f"F{i}" for i in range(1,24)]+[
        "shifted_rastrigin","rotated_rastrigin","rotated_rosenbrock"])
    p.add_argument("--algorithms",nargs="+",choices=list(ALGORITHMS),default=list(ALGORITHMS))
    p.add_argument("--output",type=Path,default=Path("gwo_diagnostic_results"))
    p.add_argument("--fresh",action="store_true")
    a=p.parse_args();suite=make_suite(seed=a.seed);a.output.mkdir(parents=True,exist_ok=True)
    raw=a.output/"raw_results.csv";rows=[]
    if raw.exists() and not a.fresh:
        with raw.open(encoding="utf-8-sig") as f:rows=list(csv.DictReader(f))
    completed={(r["function"],r["algorithm"],int(r["run"])) for r in rows}
    total=len(a.functions)*len(a.algorithms)*a.runs;done=len(completed)
    for fi,name in enumerate(a.functions):
        fn,lb,ub,dim=suite[name]
        for aname in a.algorithms:
            for run in range(a.runs):
                if (name,aname,run) in completed:continue
                seed=a.seed+100_000*fi+run;start=time.perf_counter()
                result=ALGORITHMS[aname](fn,dim,lb,ub,population_size=a.population,
                    max_evaluations=a.budget,seed=seed)
                rows.append(dict(function=name,dimension=dim,algorithm=aname,run=run,
                    seed=seed,best_f=result.best_f,evaluations=result.evaluations,
                    seconds=time.perf_counter()-start));done+=1
                print(f"[{done}/{total}] {name:<22} {aname:<15} {result.best_f:.5e}",flush=True)
                write_rows(raw,rows)
    summary=[]
    for name in a.functions:
        for aname in a.algorithms:
            g=[r for r in rows if r["function"]==name and r["algorithm"]==aname]
            v=np.array([float(r["best_f"]) for r in g]);sec=np.array([float(r["seconds"]) for r in g])
            summary.append(dict(function=name,dimension=suite[name][3],algorithm=aname,
                mean=v.mean(),std=v.std(ddof=1) if len(v)>1 else 0,median=np.median(v),
                best=v.min(),worst=v.max(),mean_seconds=sec.mean()))
    write_rows(a.output/"summary.csv",summary)


if __name__=="__main__":main()
