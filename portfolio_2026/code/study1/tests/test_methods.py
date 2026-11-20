from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from study1_extension import (auc_influence,brier_influence,age_standardized_group,
                              design_covariance,calibration,summarize)
from study2_extension import (qlike,har_panel,har_fit_predict,pairwise_inference,
                              model_confidence_set,long_to_panel,run_inference)


def synthetic_parent(seed=17,n_per_cell=16):
    rg=np.random.default_rng(seed);rows=[]
    for g,bmi in [('low',22.),('middle',27.),('high',33.)]:
        for age in range(1,14):
            for y in [0,1]:
                for k in range(n_per_cell):
                    # strong age-related rank with overlapping groups
                    p=1/(1+np.exp(-(-2.3+.28*age+.4*(g=='high')+.11*y+rg.normal(0,.5))))
                    q=1/(1+np.exp(-(-2.+15*(bmi-25)/100+rg.normal(0,.3))))
                    rows.append({'target':y,'age_group':age,'bmi':bmi,
                                 '_LLCPWT':rg.uniform(.5,1.5),
                                 '_STSTR':1+(k%8),'_PSU':len(rows)+1,
                                 'prob_full':p,'prob_no_age':q})
    return pd.DataFrame(rows)


def test_weighted_auc_identical_sklearn():
    rng=np.random.default_rng(55); y=np.array([0,1,0,1,1,0,1,0,1,0])
    p=np.array([.6,.4,.3,.5,.5,.3,.7,.6,.5,.4]);w=rng.uniform(.3,1.8,len(y))
    val,ui=auc_influence(y,p,w)
    assert abs(val-roc_auc_score(y,p,sample_weight=w))<1e-13
    assert abs(ui.sum())<1e-10
    for i in [0,3,7]:
        h=1e-5;w1=w.copy();w1[i]*=1+h
        v2,_=auc_influence(y,p,w1)
        assert abs((v2-val)/h-ui[i])<2e-5


def test_brier_if_numerical():
    y=np.array([0,1,1,0,0]);p=np.array([.1,.6,.4,.2,.5]);w=np.array([1,2,3,1,4.])
    v,ui=brier_influence(y,p,w)
    assert np.isclose(ui.sum(),0,atol=1e-14)
    for i in range(5):
        wr=w.copy();wr[i]*=1+1e-6
        updated,_=brier_influence(y,p,wr)
        assert abs((updated-v)/1e-6-ui[i])<1e-6


def test_standardized_auc_delta_and_IF():
    df=synthetic_parent(n_per_cell=8); low=df.bmi.lt(25).to_numpy();high=df.bmi.ge(30).to_numpy()
    a,b,d,ui,uil,uih=age_standardized_group(df,low,high,'prob_full')
    assert 0<=a<=1 and 0<=b<=1
    assert np.isclose(d,a-b)
    assert np.isclose(ui.sum(),0,atol=2e-10)
    for i in [0,20,len(df)-3]:
        h=1e-5;tmp=df.copy();tmp.loc[i,'_LLCPWT']*=1+h
        _,_,d2,_,_,_=age_standardized_group(tmp,low,high,'prob_full')
        assert abs((d2-d)/h-ui[i])<3e-5


def test_calibration_score_and_design_cov():
    df=synthetic_parent(n_per_cell=4)
    beta,ui=calibration(df.target,df.prob_full,df._LLCPWT)
    assert np.isfinite(beta).all() and np.isfinite(ui).all()
    assert np.allclose(ui.sum(axis=0),0,atol=1e-5)
    cov,metadata=design_covariance(df,ui)
    assert np.all(np.linalg.eigvalsh(cov)>-1e-12)
    assert metadata['strata']==4


def test_study1_end_to_end(tmp_path):
    df=synthetic_parent(n_per_cell=3)
    # Add parent-design records not in analysis population
    extra=df.iloc[:20].copy();extra.target=np.nan;extra.prob_full=np.nan;extra.prob_no_age=np.nan
    extra._PSU=np.arange(90000,90020)
    result=summarize(pd.concat([df,extra],ignore_index=True),tmp_path)
    assert (tmp_path/'study1_estimates_and_CIs.csv').exists()
    for metric in ['rq1_auc_full_minus_noage','rq1_brier_noage_minus_full',
                   'std_bmi_gap_reduction','raw_bmi_gap_reduction']:
        assert metric in set(result.metric)


def test_qlike_known():
    y=np.array([.25,.1]);h=np.array([.5,.2])
    assert np.allclose(qlike(y,h),np.log(h)+y/h)


def test_har_panel_causal(tmp_path):
    rng=np.random.default_rng(22); n=260
    dates=pd.bdate_range('2022-01-01',periods=n)
    r=rng.normal(0,.01,n);prices=200*np.exp(np.cumsum(r))
    f=tmp_path/'prices.csv';pd.DataFrame({'Date':dates,'Close':prices}).to_csv(f,index=False)
    x=har_panel(f)
    assert len(x)>150
    # changing future prices cannot change features/targets at origin t-2
    p2=prices.copy();p2[200:]*=1.7
    pd.DataFrame({'Date':dates,'Close':p2}).to_csv(f,index=False)
    xx=har_panel(f)
    assert np.allclose(x.loc[:100,['rv1','rv5','rv22','y']],
                       xx.loc[:100,['rv1','rv5','rv22','y']])
    pred,beta=har_fit_predict(x.iloc[:130],x.iloc[131:])
    assert (pred>0).all() and len(pred)==len(x.iloc[131:])


def test_bootstrap_and_mcs():
    rng=np.random.default_rng(17);n=100
    l=np.column_stack([rng.normal(0,1,n),rng.normal(.02,1,n),
                       rng.normal(4,.1,n)])
    tab=pairwise_inference(l,['A','B','Bad'],replicates=200,block=5)
    assert len(tab)==3 and (tab.ci95_low<=tab.ci95_high).all()
    keep,hist=model_confidence_set(l,['A','B','Bad'],replicates=200,block=5)
    assert 'Bad' not in keep and len(keep)>=1


def test_finance_end_to_end(tmp_path):
    rng=np.random.default_rng(43);records=[]
    for market in ['SP500','TASI']:
        for i in range(65):
            y=abs(rng.normal(.00008,.00009));date=f'2026-04-{(i%28)+1:02d}-{i//28}'
            for name,factor in [('EWMA',1.),('Poisson',1.15),('HAR_SR2',1.35)]:
                records.append({'market':market,'fold':'2026','date':date,'model':name,
                                'target_sq_return':y,'pred_var':max(1e-9,.00008*factor)})
    df=long_to_panel(pd.DataFrame(records))
    scores,mcs=run_inference(df,tmp_path,replicates=100,block=5)
    assert len(scores)==6 and len(mcs)==2
    assert (tmp_path/'study2_pairwise_qlike_DM_bootstrap.csv').exists()
