import json, time, shutil, pathlib, pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
base=pathlib.Path('/mnt/data'); out=base/'health_risk_portfolio_v10'; shutil.copytree(base/'health_risk_portfolio_v9',out,dirs_exist_ok=True); (out/'src').mkdir(exist_ok=True)
cols=['DIABETE4','_LLCPWT','_AGEG5YR','_BMI5','SEXVAR','EXERANY2','SMOKE100','EDUCA']
def load(year):
 chunks=[]
 for ch in pd.read_sas(base/'brfss_raw'/f'LLCP{year}.XPT',format='xport',chunksize=40000):
  d=ch[cols].copy();d=d[d.DIABETE4.isin([1,3]) & d._LLCPWT.gt(0)].copy()
  d['target']=(d.DIABETE4==1).astype(int)
  d['age_group']=d._AGEG5YR.where(d._AGEG5YR.between(1,13))
  d['bmi']=(d._BMI5/100).where((d._BMI5/100).between(10,80))
  for raw,new,valid in [('SEXVAR','sex',[1,2]),('EXERANY2','exercise',[1,2]),('SMOKE100','smoked100',[1,2]),('EDUCA','education',[1,2,3,4,5,6])]:d[new]=d[raw].where(d[raw].isin(valid))
  chunks.append(d[['target','age_group','bmi','sex','exercise','smoked100','education','_LLCPWT']])
 return pd.concat(chunks,ignore_index=True)
a=load(2024);b=load(2025);features=['age_group','bmi','sex','exercise','smoked100','education'];cat=[x for x in features if x!='bmi']; mask=b[features].notna().all(axis=1);common=b.loc[mask].copy();rows=[]
for strategy in ['median','indicator','complete_case']:
 train=a if strategy!='complete_case' else a.dropna(subset=features)
 numeric=Pipeline([('scale',StandardScaler())]) if strategy=='complete_case' else Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=strategy=='indicator')),('scale',StandardScaler())])
 categorical=OneHotEncoder(handle_unknown='ignore') if strategy=='complete_case' else Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))])
 model=Pipeline([('pre',ColumnTransformer([('num',numeric,['bmi']),('cat',categorical,cat)])),('lr',LogisticRegression(max_iter=350,solver='lbfgs'))]);model.fit(train[features],train.target)
 for cohort,d in [('all_eligible',b),('common_complete_case',common)]:
  if strategy=='complete_case' and cohort=='all_eligible':continue
  y=d.target.to_numpy();p=model.predict_proba(d[features])[:,1];w=d._LLCPWT.to_numpy()
  rows.append({'strategy':strategy,'cohort':cohort,'train_n':len(train),'test_n':len(d),'auroc':roc_auc_score(y,p),'auprc':average_precision_score(y,p),'brier':brier_score_loss(y,p),'log_loss':log_loss(y,p),'weighted_auroc':roc_auc_score(y,p,sample_weight=w),'weighted_brier':brier_score_loss(y,p,sample_weight=w),'mean_pred':float(p.mean()),'observed_prevalence':float(y.mean())})
  print(strategy,cohort,rows[-1]['auroc'],rows[-1]['brier'],flush=True)
results={'protocol':'2024 training only; 2025 held-out test; common complete-case cohort for apples-to-apples comparison','n_train_2024':len(a),'n_test_2025':len(b),'n_common_test_2025':len(common),'results':rows,'limitations':['No design-correct confidence intervals','Complete-case cohort selection changes population','No prospective diabetes incidence prediction','No hyperparameter tuning on 2025']}
(out/'results'/'common_cohort_comparison_2025.json').write_text(json.dumps(results,indent=2));pd.DataFrame(rows).to_csv(out/'results'/'common_cohort_comparison_2025.csv',index=False)
(out/'src'/'common_cohort_comparison_v10.py').write_text(pathlib.Path(__file__).read_text())
(out/'README.md').write_text('''# Health Risk Modeling — V10\n\nCross-sectional self-reported diabetes classification using CDC BRFSS 2024 (training) and 2025 (held-out evaluation). **Not a diagnostic model and not a future-incidence forecast.**\n\n## V10: Fair comparison on the same evaluation population\n\nThe three missing-data strategies are evaluated on the identical 2025 complete-case subset. Median and missing-indicator strategies are additionally evaluated on the full eligible 2025 cohort. All preprocessing is fitted on 2024 only.\n\nSee `results/common_cohort_comparison_2025.csv` for exact metrics, and `src/common_cohort_comparison_v10.py` for reproduction. The input CDC SAS XPT files are not included.\n\n## Scientific limitations\n\n- Survey weights can be used for descriptive and weighted point estimates, but no design-correct uncertainty intervals are claimed.\n- No R survey cross-validation has been completed.\n- Comparisons are descriptive; 2025 was reused in earlier model inspections, so treat it as a held-out year but not a pristine one-time final test.\n- This is association/classification of *self-reported existing diabetes*, not clinical diagnosis, causality, or prospective risk prediction.\n- Any public release requires reviewing codebook definitions and reproducibility setup.\n\n## Run\n\nInstall `pandas numpy scikit-learn` and put `LLCP2024.XPT` and `LLCP2025.XPT` in `/mnt/data/brfss_raw` when running this specific script in its current environment. Adapt `base` for your local machine.\n''')
(out/'CHANGELOG.md').write_text('# V10\n\n- Added same-cohort comparison for median imputation, missingness indicator, and complete-case training.\n- Evaluated median and indicator models on both full eligible and complete-case 2025 cohorts.\n- Did not claim survey-design confidence intervals.\n')
shutil.make_archive(str(base/'health_risk_portfolio_v10'),'zip',base,'health_risk_portfolio_v10')
print('DONE',out)
