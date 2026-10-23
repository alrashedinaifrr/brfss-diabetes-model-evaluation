"""Reproducible descriptive missingness sensitivity, NOT design-based inference."""
import argparse, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('--data-dir',type=Path,default=Path('data/raw'))
p.add_argument('--out-dir',type=Path,default=Path('results'))
a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
report={}
for year in (2024,2025):
    path=a.data_dir/f'LLCP{year}XPT.zip'
    counts={}; total=0; yes=0; no=0; unknown=0; bmi_missing=0; bmi_missing_yes=0; bmi_missing_no=0
    w_yes=w_no=w_unknown=0.
    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as f:
            for c in pd.read_sas(f,format='xport',chunksize=30000):
                y=c['DIABETE4']; w=c['_LLCPWT'].to_numpy(dtype=float); bmi=c['_BMI5']
                valid=np.isfinite(w)&(w>0)
                pos=valid & y.eq(1).to_numpy(); neg=valid & y.eq(3).to_numpy(); unk=valid & ~(y.isin([1,3]).to_numpy())
                miss=bmi.isna().to_numpy() | bmi.isin([7777,9999]).to_numpy() | (bmi.to_numpy(dtype=float)<=0)
                total+=int(valid.sum());yes+=int(pos.sum());no+=int(neg.sum());unknown+=int(unk.sum())
                bmi_missing+=int((miss & valid).sum());bmi_missing_yes+=int((miss & pos).sum());bmi_missing_no+=int((miss & neg).sum())
                w_yes+=float(w[pos].sum());w_no+=float(w[neg].sum());w_unknown+=float(w[unk].sum())
    # Deliberately extreme worst-case bounds; unknown includes prediabetes and gestational diabetes,
    # so these are mathematical sensitivity bounds, NOT credible prevalence estimates.
    report[str(year)]={'positive_weight_records':yes,'negative_weight_records':no,'excluded_other_diabetes_response_records':unknown,
       'eligible_records':yes+no,'missing_or_invalid_bmi_eligible':bmi_missing_yes+bmi_missing_no,
       'bmi_missing_eligible_fraction':(bmi_missing_yes+bmi_missing_no)/(yes+no),
       'bmi_missing_yes_fraction':bmi_missing_yes/yes,'bmi_missing_no_fraction':bmi_missing_no/no,
       'eligible_weighted_prevalence':w_yes/(w_yes+w_no),
       'extreme_lower_bound_if_all_other_responses_no':w_yes/(w_yes+w_no+w_unknown),
       'extreme_upper_bound_if_all_other_responses_yes':(w_yes+w_unknown)/(w_yes+w_no+w_unknown),
       'warning':'Bounds are extreme coding sensitivity only; not prevalence estimates or survey-design confidence intervals.'}
    print(year,report[str(year)],flush=True)
(a.out_dir/'missingness_sensitivity.json').write_text(json.dumps(report,indent=2),encoding='utf8')
