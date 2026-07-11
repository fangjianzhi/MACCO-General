"""Rank and paired-sign analysis for MACCO experiment CSV files."""

import argparse,csv,math
from collections import defaultdict
from pathlib import Path
import numpy as np


def sign_p_value(wins,losses):
    n=wins+losses
    if n==0:return 1.0
    k=min(wins,losses)
    return min(1.0,2*sum(math.comb(n,i) for i in range(k+1))/(2**n))


def main():
    p=argparse.ArgumentParser(); p.add_argument("raw_csv",type=Path); p.add_argument("--reference",default="MACCO"); p.add_argument("--output",type=Path,default=None); a=p.parse_args()
    with a.raw_csv.open(encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    for r in rows:
        r["dimension"]=int(r["dimension"]); r["run"]=int(r["run"]); r["best_f"]=float(r["best_f"])
    algs=sorted({r["algorithm"] for r in rows}); cases=sorted({(r["dimension"],r["function"]) for r in rows})
    mean_ranks=defaultdict(list); pair={(r["dimension"],r["function"],r["run"],r["algorithm"]):r["best_f"] for r in rows}
    report=[]
    for d,fn in cases:
        means={alg:np.mean([r["best_f"] for r in rows if r["dimension"]==d and r["function"]==fn and r["algorithm"]==alg]) for alg in algs}
        ordered=sorted(means,key=means.get)
        for rank,alg in enumerate(ordered,1): mean_ranks[alg].append(rank)
        report.append(f"D={d} {fn}: "+", ".join(f"{alg}={means[alg]:.6e} (rank {ordered.index(alg)+1})" for alg in algs))
    report.append("\nMean ranks: "+", ".join(f"{alg}={np.mean(mean_ranks[alg]):.3f}" for alg in sorted(algs,key=lambda x:np.mean(mean_ranks[x]))))
    if a.reference in algs:
        report.append(f"\nPaired sign tests versus {a.reference}:")
        for alg in algs:
            if alg==a.reference:continue
            wins=ties=losses=0
            for d,fn in cases:
                runs=sorted({r["run"] for r in rows if r["dimension"]==d and r["function"]==fn})
                for run in runs:
                    ref=pair[(d,fn,run,a.reference)]; other=pair[(d,fn,run,alg)]
                    wins+=ref<other; ties+=ref==other; losses+=ref>other
            report.append(f"{a.reference} vs {alg}: W/T/L={wins}/{ties}/{losses}, exact sign p={sign_p_value(wins,losses):.6g}")
    text="\n".join(report); print(text)
    if a.output:a.output.write_text(text,encoding="utf-8")


if __name__=="__main__":main()
