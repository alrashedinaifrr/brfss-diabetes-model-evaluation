"""Independent 2026 BMI-domain probability-error and calibration CI checks.
Independent weighted logistic IRLS, independent linearized influence functions,
independent stratified-PSU Taylor SE, and nonlinear Rao-Wu bootstrap of fits.
"""
from pathlib import Path
import os,time
import numpy as np,pandas as pd,joblib
from scipy.special import expit
from scipy.stats import norm
root=Path(__file__).resolve().parent
out=root/'outputs'/'independent_ci_verification';out.mkdir(exist_ok=True,parents=True)
p=joblib.load(root/'outputs/reconstructed_scored_parent_2025.joblib');N=len(p)
w=p._LLCPWT.to_numpy(float)
y=p.target.to_numpy(float);bmi=p.bmi.to_numpy(float)
s=p._STSTR.to_numpy();order=np.argsort(s,kind='mergesort');sort_s=s[order]
start=np.flatnonzero(np.r_[True,sort_s[1:]!=sort_s[:-1]]);end=np.r_[start[1:],N]
hn=end-start
factor_var=np.repeat(np.where(hn>1,hn/np.maximum(hn-1,1),0),hn)
single=np.array([start[i] for i in range(len(start)) if hn[i]==1],int)

def independent_taylor(v):
    x=v[order];group_sums=np.add.reduceat(x,start)
    mu=np.repeat(group_sums/hn,hn)
    extra=x[single]-x.mean()
    return np.sqrt(np.sum((x-mu)**2*factor_var)+np.sum(extra*extra))

def newton_fit(y,z,w,beta):
    b=np.array(beta,dtype=float)
    for j in range(20):
        prob=expit(b[0]+b[1]*z)
        grad=np.array([np.sum(w*(y-prob)),np.sum(w*(y-prob)*z)])
        v=w*prob*(1-prob)
        h=np.array([[v.sum(),np.sum(v*z)],[np.sum(v*z),np.sum(v*z*z)]])
        delta=np.linalg.solve(h,grad)
        b+=delta
        if np.max(np.abs(delta))<1e-10:return b
    raise RuntimeError(f'GLM iteration did not converge: {b}')

archive=pd.read_csv(root/'outputs/study1_estimates_and_CIs.csv').set_index('metric')
cells={}
for g,mask in [('low',np.isfinite(y)&np.isfinite(bmi)&(bmi<25)),('middle',np.isfinite(y)&np.isfinite(bmi)&(bmi>=25)&(bmi<30)),('high',np.isfinite(y)&np.isfinite(bmi)&(bmi>=30))]:
    idx=np.flatnonzero(mask)
    for model in ('full','no_age'):
        prob=np.clip(p['prob_'+model].to_numpy(float)[idx],1e-9,1-1e-9)
        z=np.log(prob/(1-prob))
        yy=y[idx];ww=w[idx]
        beta=np.array([archive.loc[f'{g}_calibration_intercept_{model}','estimate'],archive.loc[f'{g}_calibration_slope_{model}','estimate']])
        ourbeta=newton_fit(yy,z,ww,beta)
        assert max(abs(ourbeta-beta))<1e-8
        mu=expit(ourbeta[0]+ourbeta[1]*z)
        r=ww*(yy-mu);v=ww*mu*(1-mu)
        h00=v.sum();h01=(v*z).sum();h11=(v*z*z).sum()
        inv=np.linalg.inv(np.array([[h00,h01],[h01,h11]]))
        ui=np.column_stack((r,r*z))@inv.T
        IF=np.zeros((N,2));IF[idx,:]=ui
        for col,measure in enumerate(('intercept','slope')):
            metric=f'{g}_calibration_{measure}_{model}'
            se=independent_taylor(IF[:,col]);orig=archive.loc[metric,'se']
            assert abs(se-orig)<1e-8,(metric,se,orig)
        cells[(g,model)]=(idx,yy,z,ww,beta)
    # independent BMI Brier difference influence
    ix=np.flatnonzero(mask);yf=y[ix]
    er_full=(yf-p.prob_full.to_numpy()[ix])**2
    er_no=(yf-p.prob_no_age.to_numpy()[ix])**2
    ef=np.average(er_full,weights=w[ix]);en=np.average(er_no,weights=w[ix])
    d=er_no-er_full;dbar=np.average(d,weights=w[ix])
    IF=np.zeros(N);IF[ix]=w[ix]*(d-dbar)/w[ix].sum()
    metric=f'{g}_brier_noage_minus_full';se=independent_taylor(IF)
    assert abs(se-archive.loc[metric,'se'])<1e-11,(metric,se,archive.loc[metric,'se'])

# Fully nonlinear confidence validation for 6 joint calibration models,
# resampled under 2025 stratified PSU design. Single PSU strata held fixed.
seed=20261119;rng=np.random.default_rng(seed)
B=int(os.environ.get('CAL_BOOT_REPS','299'))
results=[];vnames=[f'{g}_{m}_{pr}' for g in ('low','middle','high') for m in ('full','no_age') for pr in ('intercept','slope')]
mat=np.zeros((B,len(vnames)),float);t0=time.perf_counter()
for b in range(B):
  f=np.ones(N)
  for a,z in zip(start,end):
    n=z-a
    if n==1:continue
    draw=rng.integers(0,n,size=n-1); c=np.bincount(draw,minlength=n)
    f[order[a:z]]=c*n/(n-1)
  cc=0
  for g in ('low','middle','high'):
    for model in ('full','no_age'):
       idx,yy,z,ww,beta=cells[(g,model)]
       mat[b,cc:cc+2]=newton_fit(yy,z,ww*f[idx],beta);cc+=2
  if (b+1)%50==0:print('calibration resamples',b+1,'of',B,'elapsed',round(time.perf_counter()-t0,1),flush=True)
for col,key in enumerate(vnames):
 g,model,measure=key.rsplit('_',2) if False else (None,None,None)
 # name mapping below
 for gg in ('low','middle','high'):
  for mm in ('full','no_age'):
   for pr in ('intercept','slope'):
    if key==f'{gg}_{mm}_{pr}':g,model,measure=gg,mm,pr
 metric=f'{g}_calibration_{measure}_{model}'
 base=archive.loc[metric]
 std=float(mat[:,col].std(ddof=1));quant=np.quantile(mat[:,col],[.025,.975]);
 results.append({'metric':metric,'estimate':base['estimate'],'taylor_se':base['se'],'direct_bootstrap_se':std,'se_ratio':std/base['se'],'taylor_lo':base['ci95_low'],'taylor_hi':base['ci95_high'],'bootstrap_lo':quant[0],'bootstrap_hi':quant[1]})
 print(metric,'Taylor SE',round(base['se'],6),'bootstrap SE',round(std,6),'ratio',round(std/base['se'],3),flush=True)
pd.DataFrame(results).to_csv(out/'calibration_bootstrap_crosschecks.csv',index=False)
print('done',len(results),'parameters',flush=True)
