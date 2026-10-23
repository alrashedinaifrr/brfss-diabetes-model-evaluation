"""V13 audit: export individual predictions with BRFSS design variables, and descriptive diagnostics.
No design-based confidence intervals are computed or claimed.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from run_v11 import load_xpt, model_for, FEATURES

def load_design(path, chunksize=40000):
    # Use identical filtering as run_v11.transform; preserve original respondent order.
    cols = ['DIABETE4', '_LLCPWT', '_STSTR', '_PSU']
    chunks=[]
    for c in pd.read_sas(path, format='xport', chunksize=chunksize):
        missing=[x for x in cols if x not in c.columns]
        if missing: raise ValueError(f'Missing survey-design columns: {missing}')
        d=c.loc[c.DIABETE4.isin([1,3]) & c._LLCPWT.gt(0),cols]
        chunks.append(d.reset_index(drop=True))
    return pd.concat(chunks, ignore_index=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,default=Path('results_v13'))
    ap.add_argument('--save-predictions',action='store_true',help='Write respondent-level predictions; keep private')
    a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    train=load_xpt(a.data_dir/'LLCP2024.XPT'); test=load_xpt(a.data_dir/'LLCP2025.XPT')
    design=load_design(a.data_dir/'LLCP2025.XPT')
    if len(test)!=len(design) or not np.array_equal(test.target.to_numpy(),(design.DIABETE4.to_numpy()==1).astype(int)):
        raise RuntimeError('Survey design rows do not align with model evaluation rows')
    model=model_for('median');model.fit(train[FEATURES],train.target)
    y=test.target.to_numpy();p=model.predict_proba(test[FEATURES])[:,1]
    w=test._LLCPWT.to_numpy()
    summary={'n_train':len(train),'n_test':len(test), 'auroc_unweighted':float(roc_auc_score(y,p)),
        'auroc_weighted_point_estimate':float(roc_auc_score(y,p,sample_weight=w)),
        'brier_unweighted':float(brier_score_loss(y,p)),
        'brier_weighted_point_estimate':float(brier_score_loss(y,p,sample_weight=w)),
        'weighted_prevalence':float(np.average(y,weights=w)),
        'weighted_mean_prediction':float(np.average(p,weights=w)),
        'n_strata':int(design._STSTR.nunique()), 'n_psu_identifiers':int(design._PSU.nunique()),
        'limitations':['2025 reused during development','no design-based confidence intervals','no independent external validation','self-reported prevalent diabetes, not prospective incidence','no clinical utility established']}
    (a.output_dir/'v13_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    bins=pd.qcut(pd.Series(p),q=15,duplicates='drop')
    cal=pd.DataFrame({'bin':bins.astype(str),'y':y,'p':p,'w':w}).groupby('bin',sort=False).apply(
        lambda g:pd.Series({'n':len(g),'predicted_unweighted':g.p.mean(),'observed_unweighted':g.y.mean(),
            'predicted_weighted':np.average(g.p,weights=g.w),'observed_weighted':np.average(g.y,weights=g.w)}),
        include_groups=False).reset_index()
    cal.to_csv(a.output_dir/'calibration_bins_v13.csv',index=False)
    if a.save_predictions:
        pd.DataFrame({'target':y,'prediction':p,'weight':w,'stratum':design._STSTR,
            'psu':design._PSU,'sex':test.sex,'bmi':test.bmi}).to_csv(a.output_dir/'PRIVATE_predictions_2025.csv',index=False)
    print(json.dumps(summary,indent=2));print('Saved:',a.output_dir)
if __name__=='__main__':main()
