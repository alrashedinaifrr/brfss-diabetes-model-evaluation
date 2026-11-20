from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from study1_extension import (auc_influence,brier_influence,age_standardized_group,
                              design_covariance,calibration,summarize)


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


