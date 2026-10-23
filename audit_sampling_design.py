import zipfile,json,sys
import pandas as pd,numpy as np
from pathlib import Path
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--data-dir',default='data/raw')
parser.add_argument('--out-dir',default='results')
args=parser.parse_args()
out=Path(args.out_dir).parent;(out/'results').mkdir(parents=True,exist_ok=True);(out/'src').mkdir(exist_ok=True)
report={}
for year in [2024,2025]:
 p=Path(args.data_dir)/f'LLCP{year}XPT.zip'; chunks=[]
 with zipfile.ZipFile(p) as z:
  with z.open(z.namelist()[0]) as f:
   for c in pd.read_sas(f,format='xport',chunksize=40000):
    cols=['DIABETE4','_LLCPWT','_STSTR','_PSU','_STATE','SEXVAR','_BMI5']
    chunks.append(c[cols])
 d=pd.concat(chunks,ignore_index=True)
 eligible=d.DIABETE4.isin([1,3]) & d._LLCPWT.gt(0)
 e=d.loc[eligible]
 def structure(x):
  n=x.groupby('_STSTR',dropna=False)['_PSU'].nunique(dropna=False)
  return {'records':len(x),'strata':int(len(n)),'psu_pairs':int(x.groupby(['_STSTR','_PSU'],dropna=False).ngroups),'singleton_strata':int((n==1).sum()),'min_psus_per_stratum':int(n.min()),'median_psus_per_stratum':float(n.median()),'max_psus_per_stratum':int(n.max()),'missing_stratum':int(x._STSTR.isna().sum()),'missing_psu':int(x._PSU.isna().sum())}
 report[str(year)]={'raw':structure(d),'eligible':structure(e),'eligible_fraction':float(eligible.mean()),'diabetes_weighted_eligible':float(np.average((e.DIABETE4==1).astype(float),weights=e._LLCPWT)),'raw_diabetes_code_counts':{str(k):int(v) for k,v in d.DIABETE4.value_counts(dropna=False).items()}}
 print(year,report[str(year)]['eligible'],flush=True)
(Path(args.out_dir)/'sampling_design_audit.json').write_text(json.dumps(report,indent=2))

