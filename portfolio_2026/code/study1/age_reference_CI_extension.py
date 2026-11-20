"""Add paired Taylor-linearization CIs for alternative BMI age-reference targets.

The 2025 respondent-level scored parent is local to the research workspace. Never
redistribute it. CIs condition on already fitted 2024 models and use the same
public-use PSU/stratum variance approximation as the original Study I.

Computes independent references: age dist among low BMI, among high BMI,
and uniform age categories (fixed population). The outcome-specific case and
control age distributions are harmonized separately for each reference.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results';OUT.mkdir(parents=True,exist_ok=True)
from study1_extension import AGE,cell_concordance_with_standard_reference,design_covariance,age_standardized_group
PARENT=ROOT/'code/study1/outputs/reconstructed_scored_parent_2025.joblib'
parent=joblib.load(PARENT)
n=len(parent);ok=parent.target.notna().to_numpy();bmi=parent.bmi.to_numpy();age=parent.age_group.to_numpy()
low=ok & np.isfinite(bmi)&(bmi<25)&np.isin(age,AGE)
high=ok & np.isfinite(bmi)&(bmi>=30)&np.isin(age,AGE)
mask={'low':low,'high':high};pooled=low|high
refs={}
for r in ('low','high','uniform'):
    rr=np.zeros((2,13))
    if r=='uniform': rr[:]=1/13
    else:
        q=parent.loc[mask[r]];y=q.target.to_numpy(int);w=q._LLCPWT.to_numpy(float);a=q.age_group.to_numpy(int)
        for cls in (0,1):
            for cat in AGE:rr[cls,cat-1]=w[(y==cls)&(a==cat)].sum()
            rr[cls]/=rr[cls].sum()
    refs[r]=rr

results=[];cimaps={}
for r in ('low','high','uniform'):
    ref=refs[r];by_model={}; vectors=[];labels=[]
    for model in ('full','no_age'):
        values={};means={};ifgroup={}
        for g in ('low','high'):
            q=parent.loc[mask[g]];y=q.target.to_numpy(int);a=q.age_group.to_numpy(int);w=q._LLCPWT.to_numpy(float)
            auc,u,case,control=cell_concordance_with_standard_reference(y,q['prob_'+model].to_numpy(float),w,a,ref[1],ref[0])
            values[g]=auc;means[g]=(case,control)
            ui=np.zeros(n);ui[mask[g]]=u
            if r!='uniform':
                rp=parent.loc[mask[r]];ry=rp.target.to_numpy(int);ra=rp.age_group.to_numpy(int);rw=rp._LLCPWT.to_numpy(float)
                wy={c:rw[ry==c].sum() for c in (0,1)}
                extra=np.where(ry==1,rw/wy[1]*(case[ra-1]-auc),rw/wy[0]*(control[ra-1]-auc))
                ui[mask[r]]+=extra
            ifgroup[g]=ui
        key=f'{r}_{model}';v=ifgroup['low']-ifgroup['high'];by_model[model]=values['low']-values['high']
        labels.append(key);vectors.append(v)
        results.append({'reference':r,'estimand':model+'_BMI_gap','estimate':by_model[model],'n_low':int(low.sum()),'n_high':int(high.sum())})
    # Compute paired covariance and delta uncertainty for same reference
    V,meta=design_covariance(parent,np.stack(vectors,axis=1))
    for j,model in enumerate(('full','no_age')):
        se=np.sqrt(max(V[j,j],0));row=next(x for x in results if x['reference']==r and x['estimand']==model+'_BMI_gap')
        row.update({'se_taylor':se,'ci95_low':row['estimate']-1.95996398454*se,'ci95_high':row['estimate']+1.95996398454*se})
    d=by_model['full']-by_model['no_age'];se=np.sqrt(max(V[0,0]+V[1,1]-2*V[0,1],0))
    results.append({'reference':r,'estimand':'full_minus_no_age_BMI_gap','estimate':d,'se_taylor':se,
                    'ci95_low':d-1.95996398454*se,'ci95_high':d+1.95996398454*se,
                    'n_low':int(low.sum()),'n_high':int(high.sum())})
    print(r,[(x['estimand'],round(x['estimate'],6),round(x['ci95_low'],6),round(x['ci95_high'],6)) for x in results if x['reference']==r],flush=True)
    cimaps[r]={'estimate_full':by_model['full'],'estimate_no_age':by_model['no_age'],'cov':V.tolist(),'design':meta}

# Independent numerical spot check: pooled reference reproduces published value.
a,b,gg,ui,_,_=age_standardized_group(parent,low,high,'prob_full')
assert abs(gg-0.00732518205080579)<1e-7,gg
# Verify self-reference becomes raw weighted AUC, up to numerical tolerance.
from study1_extension import auc_influence
for r,g in [('low','low'),('high','high')]:
    q=parent.loc[mask[g]]
    raw,_=auc_influence(q.target.to_numpy(int),q.prob_full.to_numpy(),q._LLCPWT.to_numpy())
    # Compute A group standardized to own age case/control distribution.
    rr=refs[r];a,_,_,_=cell_concordance_with_standard_reference(q.target.to_numpy(int),q.prob_full.to_numpy(),q._LLCPWT.to_numpy(),q.age_group.to_numpy(int),rr[1],rr[0])
    assert abs(raw-a)<1e-10,(r,raw,a)

pd.DataFrame(results).to_csv(OUT/'study1_alternative_age_reference_taylor_CIs.csv',index=False)
(OUT/'study1_reference_CI_diagnostics.json').write_text(json.dumps(cimaps,indent=2))
print('DONE',flush=True)
