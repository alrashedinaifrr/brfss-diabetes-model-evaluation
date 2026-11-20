"""BRFSS 2024--2025 inferential extension for Naif Alrashedi's Study I.

Usage
-----
python study1_extension.py --train-xpt LLCP2024.XPT --evaluation-xpt LLCP2025.XPT \
  --out-dir outputs/diabetes
python study1_extension.py --scored-csv scored_parent_design.csv \
  --out-dir outputs/diabetes

The scored CSV must contain target, age_group, bmi, _LLCPWT, _STSTR, _PSU,
prob_full, prob_no_age. Records outside eligible analysis domains must stay
in the parent design, have target/probability missing, and retain the design
identifiers. Inference is conditional on the 2024 fitted classifiers.
"""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ['age_group', 'bmi', 'sex', 'exercise', 'smoked100', 'education']
RAW = ['DIABETE4', '_LLCPWT', '_STSTR', '_PSU', '_AGEG5YR', '_BMI5',
       'SEXVAR', 'EXERANY2', 'SMOKE100', 'EDUCA']
AGE = np.arange(1,14)
Z = float(norm.ppf(.975))


def extract_parent(xpt, chunksize=40000):
    """Keep *all* evaluation PSUs in the parent design (not just cases)."""
    chunks=[]
    for raw in pd.read_sas(xpt, format='xport', encoding='utf-8',chunksize=chunksize):
        needed=set(RAW)-set(raw.columns)
        if needed: raise ValueError(f'Columns missing in {xpt}: {sorted(needed)}')
        d=raw[RAW].copy()
        eligible=d['DIABETE4'].isin([1,3]) & d['_LLCPWT'].gt(0)
        d['target']=np.where(eligible,(d['DIABETE4']==1).astype(float),np.nan)
        d['age_group']=d['_AGEG5YR'].where(d['_AGEG5YR'].between(1,13))
        d['bmi']=(d['_BMI5']/100).where((d['_BMI5']/100).between(10,80))
        for src,dst,values in [('SEXVAR','sex',[1,2]),('EXERANY2','exercise',[1,2]),
                               ('SMOKE100','smoked100',[1,2]),('EDUCA','education',[1,2,3,4,5,6])]:
            d[dst]=d[src].where(d[src].isin(values))
        chunks.append(d[['target',*FEATURES,'_LLCPWT','_STSTR','_PSU']])
    return pd.concat(chunks,ignore_index=True)


def model_for(features):
    n=['bmi']; cats=[x for x in features if x!='bmi']
    prep=ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),
                        ('standardize',StandardScaler())]),n),
              ('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
                       ('encode',OneHotEncoder(handle_unknown='ignore'))]),cats)])
    return Pipeline([('prep',prep),('model',LogisticRegression(C=1.0,solver='lbfgs',max_iter=500))])


def fit_and_score(train_xpt, eval_xpt):
    tr=extract_parent(train_xpt); te=extract_parent(eval_xpt)
    eligible_tr=tr.target.notna(); eligible_te=te.target.notna()
    if eligible_tr.sum()<100 or eligible_te.sum()<100: raise ValueError('Too few eligible cases')
    te['prob_full']=np.nan;te['prob_no_age']=np.nan
    fitted={}
    for label,feat in [('full',FEATURES),('no_age',[x for x in FEATURES if x!='age_group'])]:
        m=model_for(feat)
        m.fit(tr.loc[eligible_tr,feat],tr.loc[eligible_tr,'target'].astype(int))
        te.loc[eligible_te,'prob_'+label]=m.predict_proba(te.loc[eligible_te,feat])[:,1]
        fitted[label]=m
    return te, {'n_train_eligible':int(eligible_tr.sum()),
                'n_eval_parent':int(len(te)), 'n_eval_eligible':int(eligible_te.sum())}


def _weighted_concordances(y, score, w):
    """AUC and each observation's opposite-class concordance, with half ties.

    Complexity O(n log n) from sorting; values are conditional on fixed scores.
    """
    y=np.asarray(y,int);score=np.asarray(score,float);w=np.asarray(w,float)
    assert len(y)==len(score)==len(w)
    if not np.isin(y,[0,1]).all() or not (w>0).all(): raise ValueError('Invalid outcome or weight')
    is1=y==1;is0=~is1
    if not is1.any() or not is0.any(): raise ValueError('Both outcome classes required')
    def against(query, opposite_scores, opposite_weights, greater):
        order=np.argsort(opposite_scores,kind='mergesort')
        s=opposite_scores[order];ww=opposite_weights[order]
        c=np.r_[0,np.cumsum(ww)];left=np.searchsorted(s,query,side='left');right=np.searchsorted(s,query,side='right')
        ties=c[right]-c[left]
        favorable=(c[-1]-c[right]) if greater else c[left]
        return (favorable+.5*ties)/c[-1]
    conc=np.empty(len(y))
    conc[is1]=against(score[is1],score[is0],w[is0],False)
    conc[is0]=against(score[is0],score[is1],w[is1],True)
    auc=np.average(conc[is1],weights=w[is1])
    return float(auc),conc


def auc_influence(y,score,w):
    auc,concord=_weighted_concordances(y,score,w)
    y=np.asarray(y,int);w=np.asarray(w,float)
    wy=np.where(y==1,w[y==1].sum(),w[y==0].sum())
    return auc,w*(concord-auc)/wy


def brier_influence(y,score,w):
    y=np.asarray(y,float);score=np.asarray(score,float);w=np.asarray(w,float)
    error=(y-score)**2;value=np.average(error,weights=w)
    return float(value),w*(error-value)/w.sum()


def design_covariance(df, influences):
    """Stratified, with-replacement PSU linearization on the *parent* design.

    influences is n_parent x p; rows outside analytic domain must be zero.
    Single-PSU strata use a centered-overall-PSU-mean approximation consistent
    with the convention described in the source portfolio, NOT a replicate-
    weight analysis. Treat singleton handling as a sensitivity parameter.
    """
    if np.asarray(influences).ndim==1: influences=np.asarray(influences)[:,None]
    influences=np.asarray(influences,float)
    if len(df)!=len(influences):raise ValueError('Parent design/IF mismatch')
    if not np.isfinite(influences).all():raise ValueError('Nonfinite influence')
    if df[['_STSTR','_PSU']].isna().any().any():raise ValueError('Parent strata or PSU missing')
    frame=pd.DataFrame({'_STSTR':df['_STSTR'].to_numpy(), '_PSU':df['_PSU'].to_numpy()})
    # aggregate within (stratum, PSU) rather than by a bare PSU identifier
    idx,uniques=pd.factorize(pd.MultiIndex.from_frame(frame),sort=False)
    totals=np.stack([np.bincount(idx,weights=influences[:,j],minlength=len(uniques))
                     for j in range(influences.shape[1])],axis=1)
    sdf=pd.DataFrame({'stratum':uniques.get_level_values(0).to_numpy(), 'pidx':np.arange(len(uniques))})
    cov=np.zeros((influences.shape[1],influences.shape[1]));single=[]
    overall=totals.mean(axis=0)
    for _,g in sdf.groupby('stratum',sort=False):
        v=totals[g.pidx.to_numpy()]
        m=len(v)
        if m==1:single.append(v[0]);continue
        centered=v-v.mean(axis=0)
        cov+= (m/(m-1))*(centered.T@centered)
    if single:
        singleton=np.asarray(single)-overall
        cov+=singleton.T@singleton
    return cov,{'strata':int(sdf.stratum.nunique()),'psus':int(len(uniques)),
                'singleton_strata':len(single),'singleton_center':'overall PSU mean'}


def cell_concordance_with_standard_reference(y,s,w,age,refcase,refcontrol):
    """Compute standardized AUC and IF conditional on reference r_1,r_0.

    r_1(a) and r_0(a) are pooled age distributions separately among cases
    and controls in the two compared BMI groups. Standardization is by age
    conditional on case/control status, preserving survey weights *within*
    each age x outcome cell. This is not a causal interpretation.
    """
    y=np.asarray(y,int);s=np.asarray(s,float);w=np.asarray(w,float);age=np.asarray(age,int)
    wy=np.zeros((2,13));
    for c in (0,1):
        for a in AGE: wy[c,a-1]=w[(y==c)&(age==a)].sum()
    for c,reference in [(1,refcase),(0,refcontrol)]:
        if np.any((reference>0)&(wy[c]==0)):
            raise ValueError('Age/outcome/group common support violation')
    adj=np.where(y==1,w*refcase[age-1]/wy[1,age-1],w*refcontrol[age-1]/wy[0,age-1]);
    A,ci=_weighted_concordances(y,s,adj)
    mean_case=np.zeros(13);mean_control=np.zeros(13)
    for a in AGE:
        for c,store in ((1,mean_case),(0,mean_control)):
            sel=(y==c)&(age==a)
            store[a-1]=np.average(ci[sel],weights=w[sel]) if sel.any() else np.nan
    # Within-age mean-centering removes reference-weight estimation effect
    # inside group. The extra reference-distribution IF enters outside.
    centered=np.where(y==1,ci-mean_case[age-1],ci-mean_control[age-1])
    fixed=adj*centered
    return A,fixed,mean_case,mean_control


def age_standardized_group(parent, group_low, group_high, model):
    """Return standardized group AUCs, gap, full parent-design IF.

    The reference distribution is estimated from pooled extreme-BMI group
    cases/controls; its influence is propagated through the common r's.
    """
    low=np.asarray(group_low,bool);high=np.asarray(group_high,bool)
    pooled=low|high; p=parent.loc[pooled]
    y=p.target.to_numpy().astype(int);age=p.age_group.to_numpy().astype(int)
    w=p._LLCPWT.to_numpy(float)
    W=np.zeros((2,13))
    for c in (0,1):
        for a in AGE:W[c,a-1]=w[(y==c)&(age==a)].sum()
    r0=W[0]/W[0].sum();r1=W[1]/W[1].sum()
    A={};influence={};case_means={};control_means={}
    for name,mask in [('low',low),('high',high)]:
        q=parent.loc[mask];yi=q.target.to_numpy(int);agei=q.age_group.to_numpy(int)
        val,ui,case,control=cell_concordance_with_standard_reference(
            yi,q[model].to_numpy(float),q._LLCPWT.to_numpy(float),agei,r1,r0)
        A[name]=val;case_means[name]=case;control_means[name]=control
        influence[name]=np.zeros(len(parent));influence[name][mask]=ui
        # Pooled reference IF: every respondent in extreme BMI groups enters
        # the pooled age distribution, even when not in this target group.
        idx=np.flatnonzero(pooled)
        extra=np.where(y==1,
                 w/W[1].sum()*(case[age-1]-val),
                 w/W[0].sum()*(control[age-1]-val))
        influence[name][idx]+=extra
    return (A['low'], A['high'], A['low']-A['high'],
            influence['low']-influence['high'], influence['low'], influence['high'])


def calibration(y,p,w,eps=1e-9):
    """Weighted joint calibration intercept and slope with score IFs.

    Estimates solve sum w*x*(y-expit(beta0+beta1*logit(p)))=0; IF is
    H^{-1} w*x*(y-mu), with H the weighted Hessian. Scaling weights does
    not alter the estimate or its linearized IF.
    """
    from scipy.optimize import minimize
    y=np.asarray(y,float);p=np.asarray(p,float);w=np.asarray(w,float)
    z=logit(np.clip(p,eps,1-eps));x=np.c_[np.ones(len(p)),z];ws=w/w.mean()
    def fun(beta):
        t=x@beta;ll=np.logaddexp(0,t)-y*t
        return np.sum(ws*ll)
    def jac(beta):
        return x.T@(ws*(expit(x@beta)-y))
    fit=minimize(fun,np.array([0.,1.]),jac=jac,method='BFGS',options={'gtol':1e-7})
    if not np.isfinite(fit.x).all() or np.linalg.norm(jac(fit.x))>1e-4:
        # valid optimum may warn on loss precision due to scale
        warnings.warn('Calibration optimization did not achieve strict tolerance')
    mu=expit(x@fit.x);h=x.T@((ws*mu*(1-mu))[:,None]*x)
    scores=((ws*(y-mu))[:,None]*x)
    if np.linalg.cond(h)>1e12:raise ValueError('Near-singular calibration Hessian')
    ui=np.linalg.solve(h,scores.T).T
    return fit.x,ui


def validate_inputs(df):
    df=df.copy();need={'target','age_group','bmi','_LLCPWT','_STSTR','_PSU','prob_full','prob_no_age'}
    absent=need-set(df)
    if absent:raise ValueError('Missing columns '+str(absent))
    if df[['_STSTR','_PSU']].isna().any().any():raise ValueError('Parent design records missing ids')
    if not np.isfinite(df['_LLCPWT']).all() or (df['_LLCPWT']<=0).any():
        raise ValueError('Parent weights must be finite and >0')
    eligible=df.target.notna()
    if not df.loc[eligible,'target'].isin([0,1]).all():raise ValueError('Invalid target')
    for c in ['prob_full','prob_no_age']:
        p=df.loc[eligible,c]
        if p.isna().any() or not np.isfinite(p).all() or not p.between(0,1).all():
            raise ValueError('Invalid predictions '+c)
    return df


def summarize(parent,out_dir):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    parent=validate_inputs(parent)
    n=len(parent);ok=parent.target.notna().to_numpy();
    if not (parent.loc[ok,'target'].nunique()==2):raise ValueError('Both classes required')
    estimates=[];infs=[];name_to_i={}
    def emit(metric,est,ui,meta=None):
        col=np.asarray(ui,float);assert col.shape==(n,)
        name_to_i[metric]=len(infs);infs.append(col)
        estimates.append({'metric':metric,'estimate':float(est),**(meta or {})})
    def pooled_inf(metric,func,mask,score):
        v,local=func(parent.loc[mask,'target'],parent.loc[mask,score],parent.loc[mask,'_LLCPWT'])
        ui=np.zeros(n);ui[mask]=local
        emit(metric,v,ui)
    for model in ['full','no_age']:
        pooled_inf(f'all_auc_{model}',auc_influence,ok,'prob_'+model)
        pooled_inf(f'all_brier_{model}',brier_influence,ok,'prob_'+model)
    l=parent.bmi.to_numpy(dtype=float);age=parent.age_group.to_numpy(float)
    bmi_masks={
        'low':ok&np.isfinite(l)&(l<25),
        'middle':ok&np.isfinite(l)&(l>=25)&(l<30),
        'high':ok&np.isfinite(l)&(l>=30)
    }
    for gn,mask in bmi_masks.items():
        if mask.sum()==0:continue
        for model in ['full','no_age']:
            y=parent.loc[mask,'target'];pr=parent.loc[mask,'prob_'+model];w=parent.loc[mask,'_LLCPWT']
            if y.nunique()==2:
                pooled_inf(f'{gn}_auc_{model}',auc_influence,mask,'prob_'+model)
            pooled_inf(f'{gn}_brier_{model}',brier_influence,mask,'prob_'+model)
            beta,U=calibration(y,pr,w)
            for j,measure in enumerate(['intercept','slope']):
                ui=np.zeros(n);ui[mask]=U[:,j]
                emit(f'{gn}_calibration_{measure}_{model}',beta[j],ui,{'n':int(mask.sum())})
    # For age standardization use the same complete-age, extreme-BMI cohort
    # as in the original decomposition (RQ2), rather than extrapolate to missing age.
    valid_age=np.isin(age,AGE)
    low=bmi_masks['low']&valid_age;high=bmi_masks['high']&valid_age
    # Reproduce the portfolio RQ2 contrast on age- and BMI-complete
    # respondents. BMI-wide calibration intentionally uses more observations.
    for group_name, mask in [('low',low),('high',high)]:
        for model in ['full','no_age']:
            pooled_inf(f'age_complete_{group_name}_auc_{model}',auc_influence,mask,'prob_'+model)
    for model in ['full','no_age']:
        al,ah,gap,ui,uil,uih=age_standardized_group(parent,low,high,'prob_'+model)
        emit(f'standardized_gap_{model}',gap,ui,{'n_low':int(low.sum()),'n_high':int(high.sum())})
        emit(f'standardized_auc_low_{model}',al,uil,{'n':int(low.sum())})
        emit(f'standardized_auc_high_{model}',ah,uih,{'n':int(high.sum())})
    vectors=np.stack(infs,axis=1)
    cov,design=design_covariance(parent,vectors)
    for row in estimates:
        if row['metric'] not in name_to_i: continue
        i=name_to_i[row['metric']];se=float(np.sqrt(max(cov[i,i],0)))
        row.update({'se':se,'ci95_low':row['estimate']-Z*se,'ci95_high':row['estimate']+Z*se})
    contrasts=[
        ('rq1_auc_full_minus_noage','all_auc_full',1,'all_auc_no_age',-1),
        ('rq1_brier_noage_minus_full','all_brier_no_age',1,'all_brier_full',-1),
        ('raw_bmi_gap_full','age_complete_low_auc_full',1,'age_complete_high_auc_full',-1),
        ('raw_bmi_gap_noage','age_complete_low_auc_no_age',1,'age_complete_high_auc_no_age',-1),
        ('raw_bmi_gap_reduction','age_complete_low_auc_full',1,'age_complete_high_auc_full',-1),
        ('std_bmi_gap_reduction','standardized_gap_full',1,'standardized_gap_no_age',-1)
    ]
    emap={r['metric']:r['estimate'] for r in estimates}
    for title,a,ca,b,cb in contrasts:
        if a not in name_to_i or b not in name_to_i:continue
        xa=name_to_i[a];xb=name_to_i[b]
        if title=='raw_bmi_gap_reduction':
            names=['age_complete_low_auc_full','age_complete_high_auc_full',
                   'age_complete_low_auc_no_age','age_complete_high_auc_no_age']
            coeff=np.array([1,-1,-1,1.]); value=sum(c*emap[m] for c,m in zip(coeff,names))
            ids=[name_to_i[x] for x in names];var=float(coeff@cov[np.ix_(ids,ids)]@coeff)
        else:
            value=ca*emap[a]+cb*emap[b]
            var=float(ca*ca*cov[xa,xa]+cb*cb*cov[xb,xb]+2*ca*cb*cov[xa,xb])
        se=float(np.sqrt(max(var,0)))
        estimates.append({'metric':title,'estimate':value,'se':se,'ci95_low':value-Z*se,'ci95_high':value+Z*se})
    # Additional paired domain comparisons, retaining shared PSU covariance.
    extra_contrasts = {}
    for group in ('low','middle','high'):
        extra_contrasts[f'{group}_brier_noage_minus_full'] = {
            f'{group}_brier_no_age': 1., f'{group}_brier_full': -1.}
    for model in ('full','no_age'):
        extra_contrasts[f'raw_minus_standardized_gap_{model}'] = {
            f'age_complete_low_auc_{model}': 1.,
            f'age_complete_high_auc_{model}': -1.,
            f'standardized_gap_{model}': -1.}
    for metric, weights in extra_contrasts.items():
        metrics=list(weights)
        ids=[name_to_i[m] for m in metrics]
        coeff=np.asarray([weights[m] for m in metrics])
        value=float(sum(weights[m]*emap[m] for m in metrics))
        variance=float(coeff@cov[np.ix_(ids,ids)]@coeff)
        se=float(np.sqrt(max(variance,0)))
        estimates.append({'metric':metric,'estimate':value,'se':se,
                          'ci95_low':value-Z*se,'ci95_high':value+Z*se})
    results=pd.DataFrame(estimates)
    results.to_csv(out/'study1_estimates_and_CIs.csv',index=False)
    (out/'study1_run_metadata.json').write_text(json.dumps({
        'n_parent':n,'n_eligible':int(ok.sum()),'n_low_complete_age':int(low.sum()),
        'n_high_complete_age':int(high.sum()),'design':design,
        'standardization':'Pooled extreme-BMI age distribution separately among cases and controls',
        'CI':'normal approximation based on first-order PSU Taylor linearization',
        'conditional_on':'2024 fitted models; no training uncertainty',
        'evaluation_wave_reused_during_development':True},indent=2))
    print(results.to_string(index=False))
    return results


def main():
    ap=argparse.ArgumentParser()
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--scored-csv');g.add_argument('--train-xpt')
    ap.add_argument('--evaluation-xpt');ap.add_argument('--out-dir',default='outputs/diabetes')
    args=ap.parse_args()
    if args.scored_csv:
        parent=pd.read_csv(args.scored_csv)
        meta={'input_mode':'scored CSV'}
    else:
        if not args.evaluation_xpt:ap.error('--evaluation-xpt required with --train-xpt')
        parent,meta=fit_and_score(args.train_xpt,args.evaluation_xpt)
    od=Path(args.out_dir);od.mkdir(parents=True,exist_ok=True)
    summarize(parent,od)
    (od/'input_scope.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
