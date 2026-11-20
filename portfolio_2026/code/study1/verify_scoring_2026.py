from pathlib import Path
import sys,time
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from study1_extension import fit_and_score
root=Path(__file__).resolve().parent
(root/'outputs').mkdir(exist_ok=True,parents=True)
t0=time.perf_counter()
p,meta=fit_and_score(root/'data/LLCP2024.XPT',root/'data/LLCP2025.XPT')
print('loaded and scored',meta,'time',round(time.perf_counter()-t0,1),flush=True)
mask=p.target.notna();y=p.loc[mask,'target'].astype(int);w=p.loc[mask,'_LLCPWT']
for m in ['full','no_age']:
 probs=p.loc[mask,'prob_'+m]
 print(m,'AUC',roc_auc_score(y,probs,sample_weight=w),'Brier',np.average((y-probs)**2,weights=w),flush=True)
joblib.dump(p,root/'outputs/reconstructed_scored_parent_2025.joblib',compress=3)
print('saved',root/'outputs/reconstructed_scored_parent_2025.joblib',round(time.perf_counter()-t0,1),flush=True)
