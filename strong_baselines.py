"""Research baselines based on standard CMA-ES and L-SHADE mechanisms."""

import numpy as np
from macco.optimizer import MACCOResult,_bounds,_evaluate,_reflect


def cma_es(objective,dim,lb,ub,*,population_size=40,max_evaluations=20000,seed=None):
    lo,hi=_bounds(lb,ub,dim); span=hi-lo; rng=np.random.default_rng(seed)
    lam=max(4,population_size); mu=lam//2; w=np.log(mu+.5)-np.log(np.arange(1,mu+1)); w/=w.sum()
    mueff=1/np.sum(w**2); cc=(4+mueff/dim)/(dim+4+2*mueff/dim)
    cs=(mueff+2)/(dim+mueff+5); c1=2/((dim+1.3)**2+mueff)
    cmu=min(1-c1,2*(mueff-2+1/mueff)/((dim+2)**2+mueff)); damps=1+2*max(0,np.sqrt((mueff-1)/(dim+1))-1)+cs
    mean=np.full(dim,.5); sigma=.3; C=np.eye(dim); pc=np.zeros(dim); ps=np.zeros(dim)
    chi=np.sqrt(dim)*(1-1/(4*dim)+1/(21*dim**2)); ev=0; hist=[]; it=0; best_f=np.inf; best_x=None
    while ev+lam<=max_evaluations:
        it+=1; vals,vecs=np.linalg.eigh(.5*(C+C.T)); vals=np.clip(vals,1e-20,1e20); B=vecs; D=np.sqrt(vals)
        arz=rng.standard_normal((lam,dim)); ary=(arz*D)@B.T; arx=np.clip(mean+sigma*ary,0,1)
        real=lo+arx*span; fit=_evaluate(objective,real); ev+=lam; order=np.argsort(fit)
        if fit[order[0]]<best_f: best_f=float(fit[order[0]]); best_x=real[order[0]].copy()
        old=mean.copy(); mean=w@arx[order[:mu]]; y=(mean-old)/sigma
        invsqrt=(B*(1/D))@B.T; ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mueff)*(invsqrt@y)
        hsig=float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*it))/chi < 1.4+2/(dim+1))
        pc=(1-cc)*pc+hsig*np.sqrt(cc*(2-cc)*mueff)*y
        artmp=(arx[order[:mu]]-old)/sigma
        C=((1-c1-cmu)*C+c1*(np.outer(pc,pc)+(1-hsig)*cc*(2-cc)*C)
           +cmu*sum(wi*np.outer(yi,yi) for wi,yi in zip(w,artmp)))
        sigma*=np.exp((cs/damps)*(np.linalg.norm(ps)/chi-1)); sigma=float(np.clip(sigma,1e-14,2))
        hist.append(best_f)
    while ev<max_evaluations:
        x=lo+np.clip(mean+sigma*rng.standard_normal(dim)*np.sqrt(np.diag(C)),0,1)*span
        f=float(_evaluate(objective,x[None])[0]); ev+=1
        if f<best_f: best_f,best_x=f,x.copy()
        hist.append(best_f)
    return MACCOResult(best_x,best_f,np.asarray(hist),ev,it,seed,0)


def lshade(objective,dim,lb,ub,*,population_size=40,max_evaluations=20000,seed=None):
    lo,hi=_bounds(lb,ub,dim); span=hi-lo; rng=np.random.default_rng(seed)
    np_init=max(18,population_size); np_min=4; x=lo+rng.random((np_init,dim))*span; f=_evaluate(objective,x); ev=np_init
    H=6; MF=np.full(H,.5); MCR=np.full(H,.5); mem=0; archive=np.empty((0,dim)); hist=[]; it=0
    idx=int(np.argmin(f)); best_x,best_f=x[idx].copy(),float(f[idx])
    while ev+x.shape[0]<=max_evaluations:
        it+=1; n=x.shape[0]; trial=np.empty_like(x); Fs=np.empty(n); CRs=np.empty(n)
        for i in range(n):
            r=rng.integers(H); F=MF[r]+.1*np.tan(np.pi*(rng.random()-.5))
            while F<=0: F=MF[r]+.1*np.tan(np.pi*(rng.random()-.5))
            F=min(F,1); CR=float(np.clip(rng.normal(MCR[r],.1),0,1)); Fs[i]=F; CRs[i]=CR
            pnum=max(2,int(np.ceil(.2*n))); pbest=x[rng.choice(np.argsort(f)[:pnum])]
            r1=rng.choice(np.delete(np.arange(n),i)); pool=np.vstack((x,archive)) if archive.size else x
            banned={i,r1}; choices=[k for k in range(pool.shape[0]) if not (k<n and k in banned)]; r2=rng.choice(choices)
            mutant=x[i]+F*(pbest-x[i])+F*(x[r1]-pool[r2]); cross=rng.random(dim)<CR; cross[rng.integers(dim)]=True
            trial[i]=np.where(cross,mutant,x[i])
        trial=_reflect(trial,lo,hi); tf=_evaluate(objective,trial); ev+=n; good=tf<f
        if np.any(good):
            gain=f[good]-tf[good]; sw=gain/gain.sum(); sf=Fs[good]; scr=CRs[good]
            MF[mem]=np.sum(sw*sf**2)/np.sum(sw*sf); MCR[mem]=np.sum(sw*scr**2)/max(np.sum(sw*scr),1e-12); mem=(mem+1)%H
            archive=np.vstack((archive,x[good])); maxa=n
            if len(archive)>maxa: archive=archive[rng.choice(len(archive),maxa,replace=False)]
            x[good],f[good]=trial[good],tf[good]
        idx=int(np.argmin(f));
        if f[idx]<best_f: best_x,best_f=x[idx].copy(),float(f[idx])
        target=int(round(np_init+(np_min-np_init)*ev/max_evaluations)); target=max(np_min,min(target,n))
        if target<n:
            keep=np.argsort(f)[:target]; x,f=x[keep],f[keep]; archive=archive[:min(len(archive),target)]
        hist.append(best_f)
    while ev<max_evaluations:
        candidate=lo+rng.random(dim)*span; value=float(_evaluate(objective,candidate[None])[0]); ev+=1
        if value<best_f: best_x,best_f=candidate.copy(),value
        hist.append(best_f)
    return MACCOResult(best_x,best_f,np.asarray(hist),ev,it,seed,0)
