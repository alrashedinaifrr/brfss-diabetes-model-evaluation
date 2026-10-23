"""Design-based domain inference for BRFSS 2025, with explicit singleton policy.
This script computes weighted ratio estimators using the full eligible design
and zero contributions outside the domain. It does NOT validate CDC codebooks.
Requires the private prediction file from the V3 pipeline.
"""
from pathlib import Path
import argparse, json, zipfile
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser()
parser.add_argument('--data-dir',default='data/raw')
parser.add_argument('--results-dir',default='results')
args=parser.parse_args()
root=Path(args.results_dir)
cols=['DIABETE4','_LLCPWT','_STSTR','_PSU','SEXVAR','_BMI5']
parts=[]
with zipfile.ZipFile(Path(args.data_dir)/'LLCP2025XPT.zip') as z:
    with z.open(z.namelist()[0]) as f:
        for chunk in pd.read_sas(f,format='xport',chunksize=25000):
            chunk=chunk[cols]
            parts.append(chunk.loc[chunk.DIABETE4.isin([1,3]) & chunk._LLCPWT.gt(0)].copy())
d=pd.concat(parts,ignore_index=True)
d['y']=(d.DIABETE4==1).astype(float)
pred_path=root/'predictions_logistic_regression_2025.csv'
if not pred_path.exists():
    raise FileNotFoundError(f'{pred_path}: regenerate private predictions with the V3 pipeline')
pred=pd.read_csv(pred_path).predicted_probability.to_numpy(dtype=float)
if len(pred)!=len(d): raise ValueError('Prediction alignment mismatch')
d['p']=pred

# For each domain, keep every record in the original eligible sample design.
# A domain estimator is theta = sum(w * I_domain * z) / sum(w * I_domain).
# Linearized PSU contribution = sum(w * I_domain * (z - theta)) / sum(w * I_domain).
# Singleton strata: conservative explicit 'centered at overall PSU mean' sensitivity
# convention, NOT a CDC-endorsed inferential CI. No inferential CIs are emitted.
def descriptive_domain(domain, z, metric):
    w=d._LLCPWT.to_numpy(dtype=float)
    m=np.asarray(domain,dtype=bool)
    z=np.asarray(z,dtype=float)
    valid=m & np.isfinite(z) & np.isfinite(w) & (w>0)
    if not valid.any(): raise ValueError('Empty domain')
    theta=float(np.sum(w[valid]*z[valid])/np.sum(w[valid]))
    u=np.where(valid,w*(np.nan_to_num(z,nan=0)-theta),0.)
    psu=pd.DataFrame({'stratum':d._STSTR.to_numpy(), 'psu':d._PSU.to_numpy(), 'u':u}).groupby(['stratum','psu'],sort=False)['u'].sum().reset_index()
    strata=psu.groupby('stratum',sort=False)['u']
    sizes=strata.transform('size').to_numpy()
    centered=psu.u.to_numpy()-strata.transform('mean').to_numpy()
    variance_terms=np.where(sizes>1,sizes/np.maximum(sizes-1,1)*centered**2,0.)
    singletons=int((strata.size()==1).sum())
    return {'metric':metric,'domain_n':int(valid.sum()),'estimate':theta,
            'full_design_strata':int(strata.ngroups),'full_design_psus':int(len(psu)),
            'singleton_strata_full_design':singletons,
            'variance_excluding_singletons_sensitivity_only':float(variance_terms.sum()/np.sum(w[valid])**2),
            'se':None,'ci95':None,
            'status':'DESCRIPTIVE ONLY: unresolved singleton-stratum treatment and design verification'}

rows=[]
for name,mask in [('all',np.ones(len(d),dtype=bool)),('female',d.SEXVAR.to_numpy()==2),('male',d.SEXVAR.to_numpy()==1)]:
    for label,z in [('diabetes_prevalence',d.y.to_numpy()),('brier_logistic',(d.p-d.y).to_numpy()**2),('mean_calibration_error',(d.p-d.y).to_numpy())]:
        rows.append({'domain':name,**descriptive_domain(mask,z,label)})
report={'version':'V6','source':'CDC BRFSS 2025 supplied XPT','notes':[
 'Domain variance retains the full eligible sample design rather than dropping out-of-domain records.',
 'Reported variance excluding singleton strata is a sensitivity diagnostic only, not a valid SE.',
 'Do not use these results for population inference until singleton policy, codebooks and geographic comparability are reviewed.',
 '2025 holdout must not be used for hyperparameter tuning.'], 'results':rows}
(root/'domain_design_audit_v6.json').write_text(json.dumps(report,indent=2),encoding='utf8')
pd.DataFrame(rows).to_csv(root/'domain_design_audit_v6.csv',index=False)
print(json.dumps(report,indent=2))
