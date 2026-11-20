"""Small independent software checks; these do not replace real survey inference."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from study1_extension import (auc_influence,brier_influence,calibration,
                              age_standardized_group,design_covariance)


def parent(seed=7,n=5):
    rng=np.random.default_rng(seed); rows=[]
    for group,bmi in [('low',21.),('mid',27.),('high',34.)]:
        for age in range(1,14):
            for y in [0,1]:
                for k in range(n):
                    p=1/(1+np.exp(-(-3.7+0.24*age+0.45*(group=='high')+rng.normal(0,.45))))
                    q=1/(1+np.exp(-(-2+.16*(group=='high')+rng.normal(0,.3))))
                    rows.append({'_STSTR':age,'_PSU':len(rows)+1,'_LLCPWT':rng.uniform(.2,2),
                                 'target':y,'bmi':bmi,'age_group':age,
                                 'prob_full':p,'prob_no_age':q})
    return pd.DataFrame(rows)


def test_auc_and_derivative():
    rng=np.random.default_rng(4); y=np.array([0,1,1,0,1,0,0,1,1,0]);s=np.array([.2,.3,.3,.1,.9,.2,.2,.4,.7,.9]);w=rng.uniform(.3,1.5,10)
    value, inf=auc_influence(y,s,w)
    assert abs(value-roc_auc_score(y,s,sample_weight=w))<1e-12
    for i in [0,3,6,9]:
        wp=w.copy();wp[i]*=1.000001
        v,_=auc_influence(y,s,wp)
        assert abs((v-value)/1e-6-inf[i])<1e-5


def test_brier_derivative():
    y=np.array([1,0,1,0,0]);p=np.array([.7,.3,.4,.2,.8]);w=np.array([.2,2,1,3,2.])
    v,inf=brier_influence(y,p,w)
    for i in range(len(y)):
        new=w.copy();new[i]*=1.000001
        vp,_=brier_influence(y,p,new)
        assert abs((vp-v)/1e-6-inf[i])<1e-5


def test_standardized_references_and_influence():
    df=parent()
    l=df.bmi.lt(25).to_numpy();h=df.bmi.ge(30).to_numpy()
    a,b,gap,ui,_,_=age_standardized_group(df,l,h,'prob_full')
    assert abs(a-b-gap)<1e-12
    for i in [0,3,25,len(df)-7]:
        d=df.copy();d.loc[i,'_LLCPWT']*=1.00001
        _,_,g2,*_=age_standardized_group(d,l,h,'prob_full')
        assert abs((g2-gap)/1e-5-ui[i])<1e-4


def test_calibration_score_and_derivative():
    df=parent(n=4);beta,inf=calibration(df.target,df.prob_full,df._LLCPWT)
    assert np.linalg.norm(inf.sum(axis=0))<1e-5
    for i in [1,10,22]:
        d=df.copy();d.loc[i,'_LLCPWT']*=1.0001
        beta2,_=calibration(d.target,d.prob_full,d._LLCPWT)
        assert np.linalg.norm((beta2-beta)/1e-4-inf[i])<2e-3


def test_design_covariance_nonnegative():
    df=parent();p=np.arange(len(df))*.002
    influence=np.column_stack((p-p.mean(),np.sin(p)))
    cov,meta=design_covariance(df,influence)
    assert meta['strata']==13 and meta['psus']==len(df)
    assert min(np.linalg.eigvalsh(cov))>=-1e-10
