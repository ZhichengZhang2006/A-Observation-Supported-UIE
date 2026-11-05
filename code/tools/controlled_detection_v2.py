"""Independent model-conditional sensitivity/detectability diagnostic.

Fresh analytic scenes, independent direct/backscatter coefficients, no network
or benchmark GT. Finite observed differences and independent noise draws test
local sensitivity predictions. This cannot establish risk for real enhancers.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
for key in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np
from scipy.stats import norm, rankdata


def scene(seed, size=16):
    rng=np.random.default_rng(seed)
    yy,xx=np.mgrid[:size,:size]/float(size)
    J=np.stack([.30+.12*np.sin(2*np.pi*xx), .40+.09*np.cos(4*np.pi*yy),
                .46+.07*np.sin(2*np.pi*(xx+yy))],axis=-1)
    J+=rng.uniform(-.04,.04,J.shape)
    d=.7+.15*np.sin(2*np.pi*xx)*np.cos(2*np.pi*yy)
    return J,d


def render(J,d,beta,kappa,b,gain,gamma=1.):
    T=np.exp(-d[...,None]*beta)
    S=1-np.exp(-d[...,None]*kappa)
    z=gain*(J*T+b*S)
    return np.maximum(z,0)**(1./gamma)


def auc(a,b):
    # Mann-Whitney AUC; independent class samples; ties receive average ranks.
    n,m=len(a),len(b)
    ranks=rankdata(np.r_[a,b])
    return float((ranks[n:].sum()-m*(m+1)/2.)/(m*n))


def run():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out-dir',default='artifacts/controlled_detection_v2')
    ap.add_argument('--draws',type=int,default=256)
    args=ap.parse_args()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    start=time.time()
    rows=[]
    # Fixed before evaluation; reference ds=1, sigma=.012.
    conditions=[('attenuation',s,.012,False,1.,1) for s in [.4,1.,2.5]]
    conditions += [('noise',1.,s,False,1.,1) for s in [.004,.012,.036]]
    conditions += [('brightness_matched',s,.012,True,1.,1) for s in [.4,1.,2.5]]
    conditions += [('extra_view',1.,.012,False,1.,n) for n in [1,2]]
    conditions += [('response_mismatch',1.,.012,False,g,1) for g in [1.,1.6]]
    for seed in range(1000,1008):
        J,d=scene(seed)
        yy,xx=np.mgrid[:16,:16]/16.
        for ax,scale,sig,match,gamma,nview in conditions:
            beta=np.array([1.3,.8,.55])*scale
            kappa=np.array([.35,.55,.7])  # independently varied form, NOT beta
            b=np.array([.08,.16,.24])
            gain=1.
            if match:
                gain=.35/float(render(J,d,beta,kappa,b,1.).mean())
            I0=render(J,d,beta,kappa,b,gain,gamma)
            # Assumed linear model; intentionally wrong in gamma mismatch branch.
            T=np.exp(-d[...,None]*beta)
            S=1-np.exp(-d[...,None]*kappa)
            # Four nuisance directions: channel backscatter b and a global depth shift.
            B=[]
            for c in range(3):
                v=np.zeros_like(J);v[...,c]=gain*S[...,c];B.append(v.ravel())
            depth=gain*(-beta*J*T+b*kappa*np.exp(-d[...,None]*kappa))
            B.append(depth.ravel())
            B=np.stack(B,axis=1)/sig
            # Extra view uses a distinct, known-depth offset with the SAME nuisance
            # coefficients. Never a second hidden image in a single-image model.
            ds=[d] if nview==1 else [d,d+.4]
            Bs=[]
            for dv in ds:
                tv=np.exp(-dv[...,None]*beta);sv=1-np.exp(-dv[...,None]*kappa)
                bc=[]
                for c in range(3):
                    v=np.zeros_like(J);v[...,c]=gain*sv[...,c];bc.append(v.ravel())
                bc.append((gain*(-beta*J*tv+b*kappa*np.exp(-dv[...,None]*kappa))).ravel())
                Bs.append(np.stack(bc,axis=1)/sig)
            B=np.concatenate(Bs,axis=0)
            for component in ['colour_r','band_r']:
                v=np.zeros_like(J)
                v[...,0]=1. if component=='colour_r' else np.sin(4*np.pi*xx)
                v/=np.linalg.norm(v)
                amplitude=.01*np.sqrt(16*16)  # normalized finite perturbation
                assumed=np.concatenate([(gain*np.exp(-dv[...,None]*beta)*v).ravel() for dv in ds])/sig
                actual=np.concatenate([(render(J+amplitude*v,dv,beta,kappa,b,gain,gamma)-
                                         render(J,dv,beta,kappa,b,gain,gamma)).ravel() for dv in ds])/sig
                coef=np.linalg.lstsq(B,assumed,rcond=1e-10)[0]
                efficient=assumed-B@coef
                for regime,direction in [('known',assumed),('nuisance_projected',efficient)]:
                    normdir=np.linalg.norm(direction)
                    if normdir<1e-12:
                        rows.append(dict(seed=seed,axis=ax,scale=scale,sigma=sig,match=match,
                                         gamma=gamma,nview=nview,component=component,regime=regime,
                                         q=0.,auc_pred=.5,auc_mc=.5,dprime_actual=0.,observed_mean=float(I0.mean())))
                        continue
                    direction=direction/normdir
                    # Independent draws of the full multivariate observation noise.
                    # Match-filter score, not scalar synthetic labels supplied to a model.
                    rng=np.random.default_rng(seed*10000+len(rows))
                    z0=rng.normal(size=(args.draws,len(direction)))@direction
                    z1=(rng.normal(size=(args.draws,len(direction)))+actual)@direction
                    dprime=float(actual@direction)
                    q=float(normdir**2)
                    rows.append(dict(seed=seed,axis=ax,scale=scale,sigma=sig,match=match,
                                     gamma=gamma,nview=nview,component=component,regime=regime,
                                     q=q,auc_pred=float(norm.cdf(amplitude*np.sqrt(q)/np.sqrt(2))),
                                     auc_mc=auc(z0,z1),dprime_actual=dprime,
                                     observed_mean=float(I0.mean())))
    with open(out/'per_condition.csv','w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    linear=[r for r in rows if r['gamma']==1.]
    errors=[abs(r['auc_mc']-r['auc_pred']) for r in linear]
    mismatch=[r for r in rows if r['gamma']!=1.]
    res=dict(n_rows=len(rows),n_scenes=8,draws_per_class=args.draws,
             linear_mean_absolute_auc_error=float(np.mean(errors)),
             linear_max_absolute_auc_error=float(max(errors)),
             mismatch_mean_absolute_auc_error=float(np.mean([abs(r['auc_mc']-r['auc_pred']) for r in mismatch])),
             data_space='linear radiance by construction; gamma mismatch is deliberately unmodelled',
             interpretation='instrument validation only; NOT proof of H1 enhancer-risk prediction, real-water validity or novelty',
             source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             elapsed_sec=time.time()-start,conditions=conditions,scene_seeds=list(range(1000,1008)))
    (out/'summary.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
    print(json.dumps(res,indent=2))

if __name__=='__main__':
    run()
