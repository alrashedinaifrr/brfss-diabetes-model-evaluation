"""Independent replication of BRFSS Study I confidence intervals.

Stratified Rao-Wu (m_h-1) PSU rescaled bootstrap for fixed classifier scores,
plus an independent stratified-PSU Taylor covariance implementation.
Singleton PSUs are held fixed for the direct bootstrap; report exact separate
singleton contribution and do not mislabel bootstrap CI as fully design-inclusive.
No training refits occur within resamples.
"""
from __future__ import annotations
import os, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from study1_extension import auc_influence, brier_influence, age_standardized_group,calibration
root=Path(__file__).resolve().parent
out=root/'outputs'/'independent_ci_verification'
out.mkdir(parents=True,exist_ok=True)
p=joblib.load(root/'outputs/reconstructed_scored_parent_2025.joblib')
n=len(p);yraw=p.target.to_numpy();eligible=np.isfinite(yraw)
wt=p._LLCPWT.to_numpy(float)
age=p.age_group.to_numpy(float);bmi=p.bmi.to_numpy(float)
low=eligible & np.isfinite(age)&(age>=1)&(age<=13)&np.isfinite(bmi)&(bmi<25)
high=eligible & np.isfinite(age)&(age>=1)&(age<=13)&np.isfinite(bmi)&(bmi>=30)
pooled=low|high
# Pre-sort original weights and outcomes by model scores for each estimator.
class AUCPrepared:
 def __init__(self,mask,model):
  ix=np.flatnonzero(mask)
  self.ix=ix;self.y=p.target.to_numpy()[ix].astype(np.int8)
  values=p['prob_'+model].to_numpy(float)[ix]
  self.order=np.argsort(values,kind='mergesort')
  s=values[self.order]
  self.starts=np.flatnonzero(np.r_[True,s[1:]!=s[:-1]])
 def score(self,weights):
  w=weights[self.ix][self.order]
  y=self.y[self.order]
  a=np.add.reduceat(np.where(y==1,w,0.),self.starts)
  b=np.add.reduceat(np.where(y==0,w,0.),self.starts)
  return float(np.sum(a*(np.cumsum(b)-.5*b))/(a.sum()*b.sum()))
auc_preps={(m,g):AUCPrepared(g, m) for m in ('full','no_age') for g in ["all","low","high"] for g in ([] if False else [g])} if False else {}
for m in ('full','no_age'):
 for tag,mask in [('all',eligible),('low',low),('high',high)]:auc_preps[(m,tag)]=AUCPrepared(mask,m)
idxl=np.flatnonzero(low);idxh=np.flatnonzero(high);idxp=np.flatnonzero(pooled)
y_all=yraw[eligible].astype(int);y_p=p.target.to_numpy()[idxp].astype(int)
y_l=p.target.to_numpy()[idxl].astype(int);y_h=p.target.to_numpy()[idxh].astype(int)
cell_l=y_l*13+age[idxl].astype(int)-1;cell_h=y_h*13+age[idxh].astype(int)-1
cell_p=y_p*13+age[idxp].astype(int)-1
W=p._LLCPWT.to_numpy()

def standardized_gap(ww,model):
  pooled_mass=np.bincount(cell_p,weights=ww[idxp],minlength=26)
  if (pooled_mass<=0).any():return float('nan')
  references=pooled_mass.copy();references[:13]/=pooled_mass[:13].sum();references[13:]/=pooled_mass[13:].sum()
  outv=[]
  for idx,cell,key in [(idxl,cell_l,'low'),(idxh,cell_h,'high')]:
    mass=np.bincount(cell,weights=ww[idx],minlength=26)
    if (mass<=0).any():return float('nan')
    adjusted=np.zeros(n) # simple implementation, precompute would improve speed
    adjusted[idx]=ww[idx]*(references[cell]/mass[cell])
    outv.append(auc_preps[(model,key)].score(adjusted))
  return float(outv[0]-outv[1])

# Strictly original estimates with independent definitions.
def collect(ww):
  ua=auc_preps[('full','all')].score(ww)-auc_preps[('no_age','all')].score(ww)
  yf=yraw[eligible];wf=ww[eligible];full=p.prob_full.to_numpy()[eligible];no=p.prob_no_age.to_numpy()[eligible]
  delta=np.average((yf-no)**2-(yf-full)**2,weights=wf)
  sfull=standardized_gap(ww,'full');sno=standardized_gap(ww,'no_age')
  return np.array([ua,delta,sfull,sno,sfull-sno])

v=collect(W)
print('nonlinear independent estimates:',v,flush=True)
archived=pd.read_csv(root/'outputs/study1_estimates_and_CIs.csv').set_index('metric')
metrics=['rq1_auc_full_minus_noage','rq1_brier_noage_minus_full','standardized_gap_full','standardized_gap_no_age','std_bmi_gap_reduction']
for m,z in zip(metrics,v):
 old=archived.loc[m,'estimate']
 print('check',m,'diff',z-old,flush=True)
 assert abs(old-z)<1e-8,(m,old,z)

# Independently compute paired linearized PSU covariance; use NumPy group factorization
# without invoking the portfolio's design_covariance implementation.
if not np.all(np.isfinite(p._STSTR.to_numpy())):raise ValueError('missing strata')
strata=p._STSTR.to_numpy();psu=p._PSU.to_numpy()
design=pd.DataFrame({'s':strata,'p':psu})
ids,unique_pairs=pd.factorize(pd.MultiIndex.from_frame(design))
strata_per_psu=np.array(unique_pairs.get_level_values(0))
counts=pd.Series(strata_per_psu).value_counts()
print('design  n_strata',len(counts),'n_psu',len(unique_pairs),'singleton',sum(counts==1),flush=True)
assert len(unique_pairs)==len(p)

ui={}
for m in ('full','no_age'):
 a,ua=auc_influence(y_all,p.loc[eligible,'prob_'+m].to_numpy(),W[eligible]); vec=np.zeros(n);vec[eligible]=ua;ui[f'auc_{m}']=vec
 b,ub=brier_influence(y_all,p.loc[eligible,'prob_'+m].to_numpy(),W[eligible]);vec=np.zeros(n);vec[eligible]=ub;ui[f'brier_{m}']=vec
 al,ah,gap,ustd,_,_=age_standardized_group(p,low,high,'prob_'+m);ui[f'stdgap_{m}']=ustd
influences=np.column_stack((ui['auc_full']-ui['auc_no_age'],ui['brier_no_age']-ui['brier_full'],ui['stdgap_full'],ui['stdgap_no_age'],ui['stdgap_full']-ui['stdgap_no_age']))

# Compute vector covariance with an alternative algorithm using sorted arrays and vectorized group centers.
order=np.argsort(strata,kind='mergesort');s=strata[order];U=influences[order]
start=np.flatnonzero(np.r_[True,s[1:]!=s[:-1]]);end=np.r_[start[1:],n]
mean=np.repeat(np.add.reduceat(U,start,axis=0)/(end-start)[:,None],end-start,axis=0)
center=U-mean
factors=np.repeat(np.where((end-start)>1,(end-start)/np.maximum(end-start-1,1),0),end-start)
Vnon=(center*np.sqrt(factors)[:,None]).T@(center*np.sqrt(factors)[:,None])
# grand mean used by the current portfolio's singleton policy
single=np.array([start[i] for i in range(len(start)) if end[i]-start[i]==1],int)
overall=U.mean(axis=0)
Vsingle=(U[single]-overall).T@(U[single]-overall)
Vadj=Vnon+Vsingle
selog=np.sqrt(np.diag(Vadj));senon=np.sqrt(np.diag(Vnon))
results=[]
for i,m in enumerate(metrics):
  old=archived.loc[m,'se']
  results.append({'metric':m,'point':float(v[i]),'original_taylor_se':float(old),'independent_taylor_se':float(selog[i]),'no_singleton_se':float(senon[i]),'singleton_variance_pct':float(100*Vsingle[i,i]/Vadj[i,i]),'absolute_se_difference':float(abs(old-selog[i]))})
  print('independent Taylor',m,'archived',old,'new',selog[i],'without singleton',senon[i],'singleton variance pct',round(100*Vsingle[i,i]/Vadj[i,i],3),flush=True)
  assert abs(old-selog[i])<1e-11,(m,old,selog[i])

# Bootstrap: directly re-evaluate nonlinear AUROC/standardized-AUROC and Brier
# using independently generated stratified PSU replicate weights. The singletons
# are fixed (the bootstrap is thus conservatively not claimed to mirror all singleton
# adjustment details). The comparison with no-singleton Taylor SE is appropriate.
rng=np.random.default_rng(20261118)
B=int(os.getenv('BOOT_REPS','399'))
reps=np.zeros((B,5))
t0=time.perf_counter()
for b in range(B):
  factor=np.ones(n,float)
  for a,z in zip(start,end):
    m=z-a
    if m==1:continue
    draw=rng.integers(0,m,size=m-1)
    c=np.bincount(draw,minlength=m)
    factor[order[a:z]]=c*(m/(m-1))
  reps[b]=collect(W*factor)
  if (b+1)%50==0: print('bootstrap completed',b+1,'of',B,'seconds',round(time.perf_counter()-t0,1),flush=True)
bsse=np.std(reps,axis=0,ddof=1)
for i,m in enumerate(metrics):
 qlo,qhi=np.percentile(reps[:,i],[2.5,97.5]);
 results[i].update({'bootstrap_reps':B,'bootstrap_se':float(bsse[i]),'bootstrap_se_to_no_singleton_taylor':float(bsse[i]/senon[i]),'bootstrap_p2_5':float(qlo),'bootstrap_p97_5':float(qhi),'taylor_ci_low':float(v[i]-norm.ppf(.975)*selog[i]),'taylor_ci_high':float(v[i]+norm.ppf(.975)*selog[i])})
 print('BOOT',m,'SE',bsse[i],'Taylor non-singleton',senon[i],'ratio',bsse[i]/senon[i],'percentile',[qlo,qhi],flush=True)

# Independent weighted calibration via Statsmodels GLM for each BMI domain.
cal=[]
for name,mask in [('low',eligible&(bmi<25)),('middle',eligible&(bmi>=25)&(bmi<30)),('high',eligible&(bmi>=30))]:
 for model in ('full','no_age'):
  yp=p.loc[mask,'target'].to_numpy(float)
  prob=np.clip(p.loc[mask,'prob_'+model].to_numpy(float),1e-9,1-1e-9)
  z=np.log(prob/(1-prob))
  x=np.column_stack((np.ones(len(z)),z))
  w=W[mask];sw=w/w.mean()
  g=sm.GLM(yp,x,family=sm.families.Binomial(),freq_weights=sw).fit(maxiter=100,disp=0)
  e0=archived.loc[f'{name}_calibration_intercept_{model}','estimate'];e1=archived.loc[f'{name}_calibration_slope_{model}','estimate']
  dif=np.abs(g.params-np.array([e0,e1]))
  cal.append({'group':name,'model':model,'glm_intercept':float(g.params[0]),'glm_slope':float(g.params[1]),'diff_intercept':float(dif[0]),'diff_slope':float(dif[1])})
  print('GLM',name,model,'params',g.params,'abs error',dif,flush=True)
  assert max(dif)<1e-5

# Full 49-interval arithmetic check
for _,row in archived.iterrows():
 se=row['se'];est=row['estimate']
 if pd.notna(se):
  assert abs(est-norm.ppf(.975)*se-row['ci95_low'])<1e-10
  assert abs(est+norm.ppf(.975)*se-row['ci95_high'])<1e-10
pd.DataFrame(results).to_csv(out/'independent_core_CI_crosschecks.csv',index=False)
pd.DataFrame(cal).to_csv(out/'independent_calibration_glm_crosschecks.csv',index=False)
pd.DataFrame(reps,columns=metrics).to_csv(out/'replicate_statistics.csv',index=False)
(out/'README.txt').write_text('Fixed trained models. Stratified PSU rescaled (m_h-1) direct nonlinear bootstrap; single PSU strata held fixed in bootstrap. Full linearized singleton-adjusted Taylor SEs independently recomputed. R survey package was not executed. Bootstrap percentile intervals are sensitivity checks, not replacements for survey design-based linearized intervals.\n',encoding='utf-8')
print('FINISHED',out,flush=True)
