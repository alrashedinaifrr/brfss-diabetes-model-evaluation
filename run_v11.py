"""Reproducible V11 sensitivity analysis. No fitting or writing on import."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

RAW = ['DIABETE4','_LLCPWT','_AGEG5YR','_BMI5','SEXVAR','EXERANY2','SMOKE100','EDUCA']
FEATURES = ['age_group','bmi','sex','exercise','smoked100','education']
CATS = [x for x in FEATURES if x != 'bmi']

def transform(chunk):
    d = chunk.loc[chunk.DIABETE4.isin([1,3]) & chunk._LLCPWT.gt(0), RAW].copy()
    d['target'] = (d.DIABETE4 == 1).astype(int)
    d['age_group'] = d._AGEG5YR.where(d._AGEG5YR.between(1,13))
    d['bmi'] = (d._BMI5 / 100).where((d._BMI5 / 100).between(10,80))
    for raw, new, valid in [('SEXVAR','sex',[1,2]),('EXERANY2','exercise',[1,2]),('SMOKE100','smoked100',[1,2]),('EDUCA','education',[1,2,3,4,5,6])]:
        d[new] = d[raw].where(d[raw].isin(valid))
    return d[['target',*FEATURES,'_LLCPWT']]

def load_xpt(path, chunksize=40000):
    if not path.is_file(): raise FileNotFoundError(f'Missing XPT: {path}')
    return pd.concat((transform(c) for c in pd.read_sas(path,format='xport',chunksize=chunksize)),ignore_index=True)

def model_for(strategy):
    if strategy == 'complete_case':
        numeric=Pipeline([('scale',StandardScaler())]); categorical=OneHotEncoder(handle_unknown='ignore')
    else:
        numeric=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=(strategy=='indicator'))),('scale',StandardScaler())])
        categorical=Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))])
    return Pipeline([('pre',ColumnTransformer([('num',numeric,['bmi']),('cat',categorical,CATS)])),('lr',LogisticRegression(max_iter=350,solver='lbfgs'))])

def evaluate(train, test):
    common=test.loc[test[FEATURES].notna().all(axis=1)].copy()
    rows=[]
    for strategy in ['median','indicator','complete_case']:
        fitdata = train if strategy != 'complete_case' else train.dropna(subset=FEATURES)
        model=model_for(strategy)
        model.fit(fitdata[FEATURES],fitdata.target)
        for cohort, data in [('all_eligible',test),('common_complete_case',common)]:
            if strategy=='complete_case' and cohort=='all_eligible': continue
            y=data.target.to_numpy(); p=model.predict_proba(data[FEATURES])[:,1]; w=data._LLCPWT.to_numpy()
            rows.append(dict(strategy=strategy,cohort=cohort,train_n=len(fitdata),test_n=len(data),auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),brier=brier_score_loss(y,p),log_loss=log_loss(y,p),weighted_auroc=roc_auc_score(y,p,sample_weight=w),weighted_brier=brier_score_loss(y,p,sample_weight=w),mean_pred=float(p.mean()),observed_prevalence=float(y.mean())))
    return {'protocol':'2024 train; 2025 evaluated repeatedly during development; common complete-case comparison; no design-correct confidence intervals', 'n_train_2024':len(train),'n_test_2025':len(test),'n_common_test_2025':len(common),'results':rows,'limitations':['2025 not a pristine final holdout','different training cohorts across strategies','survey design uncertainty not verified','cross-sectional self-report; no clinical diagnosis or future incidence']}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',type=Path,required=True,help='Directory containing LLCP2024.XPT and LLCP2025.XPT'); p.add_argument('--output-dir',type=Path,default=Path('results_v11')); a=p.parse_args()
    train=load_xpt(a.data_dir/'LLCP2024.XPT'); test=load_xpt(a.data_dir/'LLCP2025.XPT'); result=evaluate(train,test)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'common_cohort_comparison_2025.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    pd.DataFrame(result['results']).to_csv(a.output_dir/'common_cohort_comparison_2025.csv',index=False)
    print('Done:',a.output_dir,'training:',len(train),'evaluation:',len(test))
if __name__=='__main__': main()
