"""GWO-style visual comparison with equal budgets and paired repeated runs."""

from __future__ import annotations
import argparse, csv, time
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines import gwo, pso
from classic23_benchmarks import make_classic23_suite
from macco import minimize, minimize_subspace

ALGORITHMS = {
    "PSO": pso,
    "GWO": gwo,
    "MACCO base": minimize,
    "MACCO subspace": minimize_subspace,
}


def resample(history, points=300):
    old=np.linspace(0,1,len(history)); new=np.linspace(0,1,points)
    return np.interp(new,old,history)


def terrain(objective, dim, lb, ub, points=100):
    lo=np.broadcast_to(np.asarray(lb,dtype=float),(dim,)); hi=np.broadcast_to(np.asarray(ub,dtype=float),(dim,))
    x=np.linspace(lo[0],hi[0],points); y=np.linspace(lo[1],hi[1],points)
    xx,yy=np.meshgrid(x,y); zz=np.empty_like(xx); anchor=(lo+hi)/2
    for i in range(points):
        for j in range(points):
            vector=anchor.copy();vector[0]=xx[i,j];vector[1]=yy[i,j]
            zz[i,j]=objective(vector)
    return xx,yy,zz


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--function",default="F9",choices=[f"F{i}" for i in range(1,24)])
    p.add_argument("--runs",type=int,default=10)
    p.add_argument("--population",type=int,default=30)
    p.add_argument("--budget",type=int,default=30_000)
    p.add_argument("--seed",type=int,default=20260711)
    p.add_argument("--output",type=Path,default=Path("comparison_demo"))
    a=p.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install plotting support with: pip install -e .[plot]") from exc

    objective,lb,ub,dim=make_classic23_suite()[a.function]
    a.output.mkdir(parents=True,exist_ok=True); rows=[]; histories={name:[] for name in ALGORITHMS}
    for run in range(a.runs):
        seed=a.seed+run
        for name,opt in ALGORITHMS.items():
            start=time.perf_counter(); result=opt(objective,dim,lb,ub,
                population_size=a.population,max_evaluations=a.budget,seed=seed)
            elapsed=time.perf_counter()-start; histories[name].append(resample(result.history))
            rows.append(dict(function=a.function,algorithm=name,run=run,seed=seed,
                best_f=result.best_f,evaluations=result.evaluations,seconds=elapsed))
            print(f"run={run+1:02d}/{a.runs} {name:<15} best={result.best_f:.6e}",flush=True)

    with (a.output/"raw_results.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    summary=[]
    for name in ALGORITHMS:
        values=np.array([r["best_f"] for r in rows if r["algorithm"]==name])
        seconds=np.array([r["seconds"] for r in rows if r["algorithm"]==name])
        summary.append(dict(function=a.function,algorithm=name,mean=values.mean(),
            std=values.std(ddof=1) if len(values)>1 else 0,median=np.median(values),
            best=values.min(),worst=values.max(),mean_seconds=seconds.mean()))
    with (a.output/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)

    fig=plt.figure(figsize=(15,4.6)); ax=fig.add_subplot(1,3,1,projection="3d")
    xx,yy,zz=terrain(objective,dim,lb,ub);ax.plot_surface(xx,yy,zz,cmap="viridis",linewidth=0,antialiased=True)
    ax.set_title(f"{a.function} landscape (first 2 dimensions)");ax.set_xlabel("x1");ax.set_ylabel("x2");ax.set_zlabel("f(x)")
    ax=fig.add_subplot(1,3,2); progress=np.linspace(0,1,300)
    for name,data in histories.items():
        h=np.asarray(data); med=np.median(h,axis=0);q1,q3=np.quantile(h,[.25,.75],axis=0)
        ax.plot(progress,med,label=name,linewidth=2);ax.fill_between(progress,q1,q3,alpha=.12)
    positive=all(np.min(data)>0 for data in histories.values())
    if positive: ax.set_yscale("log")
    else: ax.set_yscale("symlog",linthresh=1e-8)
    ax.grid(True,which="both",alpha=.25)
    ax.set_title(f"Convergence ({a.runs} paired seeds)");ax.set_xlabel("Evaluation-budget progress");ax.set_ylabel("Best objective");ax.legend(fontsize=8)
    ax=fig.add_subplot(1,3,3);labels=list(ALGORITHMS);values=[[r["best_f"] for r in rows if r["algorithm"]==name] for name in labels]
    ax.boxplot(values,tick_labels=labels,showmeans=True);ax.tick_params(axis="x",rotation=25);ax.grid(True,axis="y",alpha=.25)
    ax.set_title("Final objective distribution");ax.set_ylabel("Best objective")
    fig.suptitle(f"Equal-budget optimizer comparison: {a.function}, D={dim}, B={a.budget}",fontsize=14)
    fig.tight_layout();output=a.output/f"{a.function}_comparison.png";fig.savefig(output,dpi=180);print(output)


if __name__=="__main__":main()
