"""Transparent equal-budget baselines for MACCO development experiments."""

import numpy as np

from macco.optimizer import MACCOResult, _bounds, _evaluate, _reflect, _rank_weights


def de(objective, dim, lb, ub, *, population_size=40, max_evaluations=20000, seed=None):
    lo,hi=_bounds(lb,ub,dim); span=hi-lo; rng=np.random.default_rng(seed)
    x=lo+rng.random((population_size,dim))*span; f=_evaluate(objective,x); ev=population_size
    idx=int(np.argmin(f)); best_x,best_f=x[idx].copy(),float(f[idx]); history=[best_f]; it=0
    while ev+population_size<=max_evaluations:
        it+=1; trial=np.empty_like(x)
        for i in range(population_size):
            pool=np.delete(np.arange(population_size),i); a,b,c=rng.choice(pool,3,replace=False)
            mutant=x[a]+0.5*(x[b]-x[c]); cross=rng.random(dim)<0.9; cross[rng.integers(dim)]=True
            trial[i]=np.where(cross,mutant,x[i])
        trial=_reflect(trial,lo,hi); tf=_evaluate(objective,trial); ev+=population_size
        good=tf<f; x[good],f[good]=trial[good],tf[good]; idx=int(np.argmin(f))
        if f[idx]<best_f: best_x,best_f=x[idx].copy(),float(f[idx])
        history.append(best_f)
    return MACCOResult(best_x,best_f,np.asarray(history),ev,it,seed,0)


def pso(objective, dim, lb, ub, *, population_size=40, max_evaluations=20000, seed=None):
    lo,hi=_bounds(lb,ub,dim); span=hi-lo; rng=np.random.default_rng(seed)
    x=lo+rng.random((population_size,dim))*span; v=rng.uniform(-.1,.1,x.shape)*span
    f=_evaluate(objective,x); ev=population_size; pb=x.copy(); pbf=f.copy()
    idx=int(np.argmin(f)); best_x,best_f=x[idx].copy(),float(f[idx]); history=[best_f]; it=0
    while ev+population_size<=max_evaluations:
        it+=1; progress=ev/max_evaluations; w=.9-.5*progress
        v=w*v+1.7*rng.random(x.shape)*(pb-x)+1.7*rng.random(x.shape)*(best_x-x)
        x=_reflect(x+v,lo,hi); f=_evaluate(objective,x); ev+=population_size
        good=f<pbf; pb[good],pbf[good]=x[good],f[good]; idx=int(np.argmin(pbf))
        if pbf[idx]<best_f: best_x,best_f=pb[idx].copy(),float(pbf[idx])
        history.append(best_f)
    return MACCOResult(best_x,best_f,np.asarray(history),ev,it,seed,0)


def gwo(objective, dim, lb, ub, *, population_size=40, max_evaluations=20000, seed=None):
    """Grey Wolf Optimizer with a strict objective-evaluation budget."""
    if population_size < 3:
        raise ValueError("GWO requires at least three search agents")
    lo,hi=_bounds(lb,ub,dim); span=hi-lo; rng=np.random.default_rng(seed)
    x=lo+rng.random((population_size,dim))*span; f=_evaluate(objective,x); ev=population_size
    order=np.argsort(f); alpha,beta,delta=x[order[:3]].copy()
    best_x,best_f=alpha.copy(),float(f[order[0]]); history=[best_f]; it=0
    while ev+population_size<=max_evaluations:
        it+=1; progress=ev/max_evaluations; a=2*(1-progress)
        leaders=(alpha,beta,delta); proposals=[]
        for leader in leaders:
            r1=rng.random(x.shape); r2=rng.random(x.shape)
            aa=2*a*r1-a; c=2*r2
            proposals.append(leader-aa*np.abs(c*leader-x))
        x=_reflect(sum(proposals)/3,lo,hi); f=_evaluate(objective,x); ev+=population_size
        order=np.argsort(f); alpha,beta,delta=x[order[:3]].copy()
        if f[order[0]]<best_f: best_x,best_f=alpha.copy(),float(f[order[0]])
        history.append(best_f)
    return MACCOResult(best_x,best_f,np.asarray(history),ev,it,seed,0)


def cbo(objective, dim, lb, ub, *, population_size=40, max_evaluations=20000, seed=None):
    """Lightweight discrete consensus-based optimization reference."""
    lo,hi=_bounds(lb,ub,dim); span=hi-lo; rng=np.random.default_rng(seed)
    x=lo+rng.random((population_size,dim))*span; f=_evaluate(objective,x); ev=population_size
    idx=int(np.argmin(f)); best_x,best_f=x[idx].copy(),float(f[idx]); history=[best_f]; it=0
    while ev+population_size<=max_evaluations:
        it+=1; progress=ev/max_evaluations; weights=_rank_weights(f,6.0); consensus=weights@x
        drift=1.2*rng.random((population_size,1))*(consensus-x)
        diffusion=.25*(1-progress)*rng.standard_normal(x.shape)*np.abs(x-consensus)
        trial=_reflect(x+drift+diffusion,lo,hi); tf=_evaluate(objective,trial); ev+=population_size
        good=tf<f; x[good],f[good]=trial[good],tf[good]; idx=int(np.argmin(f))
        if f[idx]<best_f: best_x,best_f=x[idx].copy(),float(f[idx])
        history.append(best_f)
    return MACCOResult(best_x,best_f,np.asarray(history),ev,it,seed,0)
