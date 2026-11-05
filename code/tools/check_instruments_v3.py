"""Versioned implementation diagnostics; not scientific acceptance or test edits."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from obsui import support as S
from obsui import gauge as G

rows = []
wm = S.default_water()
J = np.array([[.05, .4, .4]])
r = G.colour_budget(J, np.ones(1), wm, 1.)
Jp, dp = G.gauge_pair(J, np.ones(1), wm, np.array([r['dd_used']]))
assert not r['admissible'] and Jp.min() >= -1e-12
rows.append({'check':'negative_signal_floor', 'result':r, 'min_Jprime':float(Jp.min())})
J = np.full((2, 3), .2)
d = np.array([.1, 3.])
lo, hi = G.admissible_range(J, d, wm)
_, dp = G.gauge_pair(J, d, wm, np.full(2, hi))
assert (wm.beta[None,:] * dp[:,None]).max() <= S.TAU_MAX + 1e-12
rows.append({'check':'maximum_depth_tau', 'lo':lo, 'hi':hi})
rng = np.random.default_rng(781)
max_diff = 0.
for i in range(24):
    J = rng.uniform(0,1,(13,3))
    d = rng.uniform(.01, S.TAU_MAX / wm.beta.max(),13)
    lo, hi = G.admissible_range(J,d,wm)
    assert lo <= 0 <= hi
    for step in [lo, (lo+hi)/2, hi]:
        dd = np.full(13,step)
        Jp, dp = G.gauge_pair(J,d,wm,dd)
        assert Jp.min() >= -1e-12 and Jp.max() <= 1+1e-12
        assert dp.min() >= -1e-12
        assert (dp[:,None]*wm.beta[None,:]).max() <= S.TAU_MAX+1e-12
        diff = G.verify_exact(J,d,wm,dd)['max_abs_diff']
        max_diff = max(max_diff,diff)
assert max_diff < 1e-12
rows.append({'check':'range_endpoints_and_render', 'cases':24,'max_render_error':max_diff})
for h,w,k in [(16,16,9),(9,13,4),(1,1,1)]:
    comps=S.components(h,w,k)
    V=np.stack([v.ravel() for v in comps.values()],axis=1)
    err=float(np.max(np.abs(V.T@V-np.eye(V.shape[1]))))
    assert err<1e-12
    rank=np.linalg.matrix_rank(S.depth_basis(h,w,k)-S.depth_basis(h,w,k).mean(axis=0))
    assert len(comps)==3+3*rank
    rows.append({'check':'rank_aware_directions','shape':[h,w],'k':k,'low_rank':int(rank),'orthogonality_error':err})
# An independent finite-difference derivative with non-unit channel gains.
h,w=4,5
J=rng.uniform(.1,.8,(h*w,3)); phi=S.depth_basis(h,w,4)
d=phi@np.array([.2,.4,.7,.9]); water=S.WaterModel(wm.beta.copy(),np.array([.8,2.,1.4]),wm.b_inf.copy())
tau=d[:,None]*water.beta[None,:]; sig=np.full_like(J,.02)
I=S.render(J,tau,water)
fit={'water':water,'phi':phi,'depth':d,'sigma':sig,'tau':tau}
an=S.analyse(I,J,fit); v=rng.normal(size=J.shape); v/=np.linalg.norm(v)
eps=1e-6
rv=(S.render(J+eps*v,tau,water)-S.render(J-eps*v,tau,water))/(2*eps)/sig
q_fd=float(np.sum(rv**2)); q=S.direction_info(an,v)['q']
assert abs(q-q_fd)/q_fd < 1e-8
rows.append({'check':'gain_aware_direction_finite_difference','q':q,'q_finite_difference':q_fd})
# Oracle PSNR selection is a maximisation, and a grid family is not the full
# observation-compatible set (the latter also admits pixel-varying depth).
J=np.full((2,3),.3); d=np.ones(2)
Jg,_=G.gauge_pair(J,d,wm,np.full(2,.1))
f=G.family_spread(J,Jg,wm,d_ref=d,dd_lo=-.2,dd_hi=.2,n_grid=41)
assert f['psnr_best']>=f['psnr_at_zero'] and f['psnr_spread']>0 and abs(f['dd_best']-.1)<1e-12
rows.append({'check':'oracle_psnr_extremum','result':f})
Jp,_=G.gauge_pair(J,d,wm,np.array([.1,-.1]))
r=G.best_step(J,Jp,wm)
exact=G.verify_exact(J,d,wm,np.array([.1,-.1]))['max_abs_diff']
assert exact<1e-12 and r['frac']<.01
rows.append({'check':'family_external_is_not_observation_inconsistent','global_family_fraction':r['frac'],'max_render_error':exact})
print(json.dumps({'version':'20261105.v3','status':'implementation_checks_passed','not_scientific_acceptance':True,'checks':rows,'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(S.__file__),Path(G.__file__)]}},indent=2))
