"""Independent stratified PSU rescaled-bootstrap spot-check of alternative ref CIs.

Uses sklearn roc_auc_score rather than the sorted-pair concordance implementation
used for Taylor linearization. Only fixed 2024 predictions are resampled.
"""
import json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.metrics import roc_auc_score
root=Path(__file__).resolve().parents[2];out=root/'results';out.mkdir(exist_ok=True,parents=True)
x=joblib.load(root/'code/study1/outputs/reconstructed_scored_parent_2025.joblib')
valid=(x.target.notna() & x.age_group.between(1,13)&x.bmi.notna()).to_numpy()
l=(valid & (x.bmi.to_numpy()<25)); h=(valid & (x.bmi.to_numpy()>=30));m={'low':l,'high':h}
vals={g:{'y':x.loc[s,'target'].to_numpy(int),'a':x.loc[s,'age_group'].to_numpy(int)-1,
         'w':x.loc[s,'_LLCPWT'].to_numpy(float),'p_full':x.loc[s,'prob_full'].to_numpy(float),
         'p_no_age':x.loc[s,'prob_no_age'].to_numpy(float),'indices':np.flatnonzero(s)} for g,s in m.items()}
original=pd.read_csv(out/'study1_alternative_age_reference_taylor_CIs.csv')
rng=np.random.default_rng(312709)
# Group of PSUs: 2025 public-use data each record has an independent PSU id within a stratum.
strata=x._STSTR.to_numpy();psu=x._PSU.to_numpy()
order=np.argsort(strata,kind='stable');sorted_s=strata[order];cuts=np.r_[0,np.flatnonzero(np.diff(sorted_s))+1,len(order)]
assert pd.MultiIndex.from_frame(x[['_STSTR','_PSU']]).nunique()==len(psu),'Check nested PSU multiplicity before resampling'

B=149;rows=[];start=time.time()
for b in range(B):
    mult=np.zeros(len(x),dtype=float)
    for begin,end in zip(cuts[:-1],cuts[1:]):
        ids=order[begin:end]; k=len(ids)
        if k==1:mult[ids]=1.;continue
        counts=np.bincount(rng.integers(0,k,size=k-1),minlength=k)
        mult[ids]=counts*k/(k-1)
    ref_values={}
    groupW={g:vals[g]['w']*mult[vals[g]['indices']] for g in ('low','high')}
    for r in ('low','high','uniform'):
        if r=='uniform':ref=np.full((2,13),1/13)
        else:
            v=vals[r];ref=np.bincount(v['y']*13+v['a'],weights=groupW[r],minlength=26).reshape(2,13)
            ref=ref/ref.sum(axis=1,keepdims=True)
        gap={}
        for model in ('full','no_age'):
            AU={}
            for g in ('low','high'):
                v=vals[g]; w=groupW[g];d=np.bincount(v['y']*13+v['a'],weights=w,minlength=26).reshape(2,13)
                assert np.all(d>0)
                adjusted=w*ref[v['y'],v['a']]/d[v['y'],v['a']]
                selected=adjusted>0
                AU[g]=roc_auc_score(v['y'][selected],v['p_'+model][selected],sample_weight=adjusted[selected])
            gap[model]=AU['low']-AU['high']
        rows.append({'replicate':b,'reference':r,'full_BMI_gap':gap['full'],
                     'no_age_BMI_gap':gap['no_age'],'full_minus_no_age_BMI_gap':gap['full']-gap['no_age']})
    if (b+1)%25==0:print('bootstrap',b+1,'of',B,'seconds',round(time.time()-start,1),flush=True)
res=pd.DataFrame(rows);res.to_csv(out/'study1_reference_bootstrap_replicates.csv',index=False)
comparison=[]
for (r,m),g in [( (r,k), res[res.reference==r][k].to_numpy()) for r in ('low','high','uniform') for k in ('full_BMI_gap','no_age_BMI_gap','full_minus_no_age_BMI_gap')]:
    orig=original[(original.reference==r)&(original.estimand==m)].iloc[0]
    ci=np.quantile(g,[.025,.975]);bootse=g.std(ddof=1);ratio=bootse/orig.se_taylor
    comparison.append({'reference':r,'estimand':m,'taylor_low':orig.ci95_low,'taylor_high':orig.ci95_high,
                       'bootstrap_ci95_low':ci[0],'bootstrap_ci95_high':ci[1],
                       'taylor_se':orig.se_taylor,'bootstrap_se':bootse,'ratio_boot_to_taylor_se':ratio})
result=pd.DataFrame(comparison);result.to_csv(out/'study1_alternative_reference_bootstrap_crosscheck.csv',index=False)
print(result.to_string(index=False),flush=True)
