"""Direct paired stratified PSU bootstrap for three BMI Brier contrasts."""
from pathlib import Path
import numpy as np,pandas as pd,joblib
root=Path(__file__).resolve().parent
p=joblib.load(root/'outputs/reconstructed_scored_parent_2025.joblib')
N=len(p);y=p.target.to_numpy(float);w=p._LLCPWT.to_numpy(float);bmi=p.bmi.to_numpy(float)
valid=np.isfinite(y)&np.isfinite(bmi)
s=p._STSTR.to_numpy();ord=np.argsort(s,kind='mergesort');ss=s[ord]
start=np.flatnonzero(np.r_[True,ss[1:]!=ss[:-1]]);end=np.r_[start[1:],N]
groups={'low':valid&(bmi<25),'middle':valid&(bmi>=25)&(bmi<30),'high':valid&(bmi>=30)}
error=(y-p.prob_no_age.to_numpy(float))**2-(y-p.prob_full.to_numpy(float))**2
B=699;rng=np.random.default_rng(20261125);values=np.zeros((B,3))
for b in range(B):
 f=np.ones(N)
 for a,z in zip(start,end):
  m=z-a
  if m==1:continue
  ix=rng.integers(0,m,size=m-1);f[ord[a:z]]=np.bincount(ix,minlength=m)*m/(m-1)
 wf=w*f
 for i,mask in enumerate(groups.values()):values[b,i]=np.average(error[mask],weights=wf[mask])
print('Brier 699-replicate independent CI audit',flush=True)
rows=[];arch=pd.read_csv(root/'outputs/study1_estimates_and_CIs.csv').set_index('metric')
for i,g in enumerate(groups):
 m=g+'_brier_noage_minus_full';a=arch.loc[m];q=np.quantile(values[:,i],[.025,.975]);sd=values[:,i].std(ddof=1)
 rows.append({'metric':m,'original_CI_low':a['ci95_low'],'original_CI_high':a['ci95_high'],'bootstrap_CI_low':q[0],'bootstrap_CI_high':q[1],'taylor_SE':a['se'],'bootstrap_SE':sd,'SE_ratio':sd/a['se']})
 print(g,'point',round(a['estimate'],7),'original',round(a['ci95_low'],7),round(a['ci95_high'],7),'bootstrap',np.round(q,7),'SE_ratio',round(sd/a['se'],4),flush=True)
pd.DataFrame(rows).to_csv(root/'outputs/independent_ci_verification/bmi_brier_bootstrap_crosschecks.csv',index=False)
