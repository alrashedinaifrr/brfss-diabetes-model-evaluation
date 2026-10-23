import sys,zipfile,json,shutil,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
root=Path('data/raw'); dest=Path('results'); dest.mkdir(parents=True, exist_ok=True)
cols=['DIABETE4','_LLCPWT','_AGEG5YR','_BMI5','SEXVAR','EXERANY2','SMOKE100','EDUCA']
def load(year):
    frames=[]
    with zipfile.ZipFile(root/f'LLCP{year}XPT.zip') as z:
      with z.open(z.namelist()[0]) as f:
       for chunk in pd.read_sas(f,format='xport',chunksize=30000):
        d=chunk[cols].copy(); d=d[d.DIABETE4.isin([1,3]) & d._LLCPWT.gt(0)].copy()
        d['target']=(d.DIABETE4==1).astype(int)
        d['age_group']=d._AGEG5YR.where(d._AGEG5YR.between(1,13))
        d['bmi']=(d._BMI5/100).where((d._BMI5/100).between(10,80))
        for raw,new,valid in [('SEXVAR','sex',[1,2]),('EXERANY2','exercise',[1,2]),('SMOKE100','smoked100',[1,2]),('EDUCA','education',[1,2,3,4,5,6])]:d[new]=d[raw].where(d[raw].isin(valid))
        frames.append(d[['target','age_group','bmi','sex','exercise','smoked100','education','_LLCPWT']])
    return pd.concat(frames,ignore_index=True)
a=load(2024);b=load(2025); print('loaded',len(a),len(b),flush=True)
features=['age_group','bmi','sex','exercise','smoked100','education'];cat=[x for x in features if x!='bmi'];
def build(strategy):
    if strategy=='complete_case':
      numeric=Pipeline([('scale',StandardScaler())]); categorical=OneHotEncoder(handle_unknown='ignore')
    else:
      numeric=Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=strategy=='indicator')),('scale',StandardScaler())]);categorical=Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))])
    pre=ColumnTransformer([('num',numeric,['bmi']),('cat',categorical,cat)])
    return Pipeline([('pre',pre),('lr',LogisticRegression(max_iter=350,solver='lbfgs'))])
rows=[]
for strategy in ['median','indicator','complete_case']:
    train=a if strategy!='complete_case' else a.dropna(subset=features)
    test=b if strategy!='complete_case' else b.dropna(subset=features)
    m=build(strategy);start=time.time();m.fit(train[features],train.target)
    p=m.predict_proba(test[features])[:,1];y=test.target.to_numpy();w=test._LLCPWT.to_numpy()
    row={'strategy':strategy,'train_n':len(train),'test_n':len(test),'test_population':'all eligible' if strategy!='complete_case' else 'complete cases only (not directly comparable)', 'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p)),'weighted_auroc':float(roc_auc_score(y,p,sample_weight=w)),'weighted_brier':float(brier_score_loss(y,p,sample_weight=w)),'elapsed_seconds':round(time.time()-start,1)}
    rows.append(row);print(row,flush=True)
(dest/'imputation_sensitivity_2025.json').write_text(json.dumps({'experiment':'pre-specified sensitivity; train on 2024, evaluate 2025; no tuning on 2025','results':rows,'warnings':['Complete-case test population differs; scores are not directly comparable','No survey-design valid confidence intervals']},indent=2))
(dest/'imputation_sensitivity_2025.csv').write_text(pd.DataFrame(rows).to_csv(index=False))
print('Results saved to',dest,flush=True)
