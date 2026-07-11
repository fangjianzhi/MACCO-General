"""Experimental MACCO-LR 0.3 with diagonal-plus-low-rank search geometry."""

from __future__ import annotations
import numpy as np
from .optimizer import MACCOResult,_bounds,_evaluate,_rank_weights,_reflect


def minimize_low_rank(objective,dim,lb,ub,*,population_size=40,max_evaluations=20000,
                      seed=None,scout_fraction=.20,polish_fraction=.10,
                      consensus_pressure=6.0,rank=5,low_rank_weight=.55,callback=None):
    """Experimental optimizer for rotated/non-separable continuous problems.

    Geometry is represented as diag(v) + U diag(s^2) U.T, where U contains at
    most ``rank`` elite principal directions. No D-by-D covariance is stored.
    """
    if dim<1 or population_size<8: raise ValueError("invalid dimension or population")
    if rank<1: raise ValueError("rank must be positive")
    lo,hi=_bounds(lb,ub,dim); span=hi-lo; rng=np.random.default_rng(seed)
    pop=lo+rng.random((population_size,dim))*span; cost=_evaluate(objective,pop); ev=population_size
    idx=int(np.argmin(cost)); best_x,best_f=pop[idx].copy(),float(cost[idx]); hist=[best_f]; it=0
    variance=np.ones(dim); mean_step=.18; main_limit=int(max_evaluations*(1-polish_fraction)); main_limit-=main_limit%population_size
    directions=np.empty((0,dim)); singular=np.empty(0)
    while ev+population_size<=main_limit:
        it+=1; progress=ev/max(main_limit,1); weights=_rank_weights(cost,consensus_pressure); consensus=weights@pop
        elite_n=max(3,population_size//4); elite_idx=np.argsort(cost)[:elite_n]
        elite_norm=(pop[elite_idx]-consensus)/span; observed=np.mean(elite_norm**2,axis=0)
        floor=1e-12+2e-3*(1-progress)**3; variance=.85*variance+.15*np.maximum(observed,floor)
        # Thin SVD costs O(elite_n^2 D), avoiding a D-by-D eigendecomposition.
        centered=elite_norm-elite_norm.mean(axis=0)
        if np.any(centered):
            _,s,vt=np.linalg.svd(centered/np.sqrt(max(elite_n-1,1)),full_matrices=False)
            k=min(rank,len(s),dim); directions=vt[:k]; singular=np.maximum(s[:k],1e-12)
        order=np.argsort(cost); n_scout=max(2,int(round(scout_fraction*(1-.6*progress)*population_size)))
        scouts,developers=order[-n_scout:],order[:-n_scout]; trial=pop.copy(); old=pop.copy()
        ne=len(developers); step=mean_step*max(.03,(1-progress)**.8)
        diag_noise=rng.standard_normal((ne,dim))*np.sqrt(variance)
        if len(directions):
            lr_noise=(rng.standard_normal((ne,len(directions)))*singular)@directions
            geometry=(1-low_rank_weight)*diag_noise+low_rank_weight*lr_noise
        else: geometry=diag_noise
        attraction=rng.uniform(.6,1.5,(ne,1)); trial[developers]=pop[developers]+attraction*(consensus-pop[developers])+step*geometry*span
        for i in scouts:
            pool=np.delete(np.arange(population_size),i); a,b,c=rng.choice(pool,3,replace=False)
            mutant=pop[a]+rng.uniform(.35,.85)*(pop[b]-pop[c]); cross=rng.random(dim)<.8; cross[rng.integers(dim)]=True
            trial[i]=np.where(cross,mutant,pop[i])
        trial=_reflect(trial,lo,hi); tf=_evaluate(objective,trial); ev+=population_size; good=tf<cost
        if np.any(good):
            gain=np.mean((cost[good]-tf[good])/(np.abs(cost[good])+1e-12)); mean_step*=1.02 if gain>1e-3 else .98
            mean_step=float(np.clip(mean_step,1e-4,.5)); pop[good],cost[good]=trial[good],tf[good]
        idx=int(np.argmin(cost))
        if cost[idx]<best_f: best_x,best_f=pop[idx].copy(),float(cost[idx])
        hist.append(best_f)
        if callback is not None: callback(it,best_x.copy(),best_f)

    remaining=max_evaluations-ev; batch=min(max(6,population_size//4),remaining); center,center_f=best_x.copy(),best_f
    sigma=.03; scale=np.ones(dim)
    while remaining>=batch and batch>0:
        diag=rng.standard_normal((batch,dim))*np.sqrt(scale)
        if len(directions):
            lr=(rng.standard_normal((batch,len(directions)))*singular)@directions
            noise=(1-low_rank_weight)*diag+low_rank_weight*lr
        else: noise=diag
        offspring=_reflect(center+sigma*noise*span,lo,hi); values=_evaluate(objective,offspring); ev+=batch; remaining-=batch
        j=int(np.argmin(values))
        if values[j]<center_f:
            delta=(offspring[j]-center)/span; scale=.9*scale+.1*np.maximum(delta**2/max(sigma**2,1e-30),1e-12)
            center,center_f=offspring[j].copy(),float(values[j]); sigma*=1.03
            if center_f<best_f: best_x,best_f=center.copy(),center_f
        else:sigma*=.88
        sigma=float(np.clip(sigma,1e-14,.2)); it+=1; hist.append(best_f)
    for _ in range(remaining):
        candidate=center.copy(); j=int(rng.integers(dim)); candidate[j]+=rng.normal()*sigma*span[j]; candidate=_reflect(candidate,lo,hi)
        value=float(_evaluate(objective,candidate[None])[0]); ev+=1
        if value<center_f:
            center,center_f=candidate,value
            if value<best_f:best_x,best_f=candidate.copy(),value
        hist.append(best_f)
    return MACCOResult(best_x,best_f,np.asarray(hist),ev,it,seed,0)
