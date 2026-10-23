"""Generate honest model diagnostics from original CDC data. No design-based CIs."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from run_v11 import load_xpt, model_for, FEATURES

def save(fig,path):
    fig.tight_layout();fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('results_v12'))
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    train=load_xpt(args.data_dir/'LLCP2024.XPT');test=load_xpt(args.data_dir/'LLCP2025.XPT')
    model=model_for('median');model.fit(train[FEATURES],train.target)
    y=test.target.to_numpy();p=model.predict_proba(test[FEATURES])[:,1]
    fpr,tpr,_=roc_curve(y,p);precision,recall,_=precision_recall_curve(y,p)
    fig,ax=plt.subplots(figsize=(7,5));ax.plot(fpr,tpr,label=f'Logistic AUROC = {roc_auc_score(y,p):.4f}');ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='False positive rate',ylabel='True positive rate',title='ROC | 2025 evaluation (previously used in development)');ax.legend();save(fig,args.output_dir/'roc_2025.png')
    fig,ax=plt.subplots(figsize=(7,5));ax.plot(recall,precision,label=f'AP = {average_precision_score(y,p):.4f}');ax.axhline(y.mean(),linestyle='--',color='gray',label=f'Prevalence = {y.mean():.4f}');ax.set(xlabel='Recall',ylabel='Precision',title='Precision–Recall | 2025 evaluation');ax.legend();save(fig,args.output_dir/'precision_recall_2025.png')
    fig,ax=plt.subplots(figsize=(7,5));obs,pred=calibration_curve(y,p,n_bins=15,strategy='quantile');ax.plot(pred,obs,'o-',label=f'Logistic | Brier = {brier_score_loss(y,p):.4f}');ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='Mean predicted probability',ylabel='Observed fraction',title='Calibration | quantile bins, unweighted');ax.legend();save(fig,args.output_dir/'calibration_2025.png')
    fig,ax=plt.subplots(figsize=(7,5));ax.hist(p,bins=40,alpha=.8);ax.set(xlabel='Predicted probability',ylabel='Respondent count',title='Prediction distribution | 2025');save(fig,args.output_dir/'prediction_distribution_2025.png')
    rows=[]
    for name,sub in [('Female',test.sex.eq(2)),('Male',test.sex.eq(1)),('BMI < 25',test.bmi.lt(25)),('BMI >= 30',test.bmi.ge(30))]:
        mask=sub.fillna(False).to_numpy();yy=y[mask];pp=p[mask]
        if len(np.unique(yy))<2:continue
        rows.append({'group':name,'n':int(mask.sum()),'auroc':roc_auc_score(yy,pp),'auprc':average_precision_score(yy,pp),'brier':brier_score_loss(yy,pp),'prevalence':float(yy.mean()),'mean_prediction':float(pp.mean())})
    groups=pd.DataFrame(rows);groups.to_csv(args.output_dir/'subgroup_metrics_2025.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,5));ax.bar(groups.group,groups.auroc);ax.set(ylim=(0,1),ylabel='AUROC',title='Subgroup discrimination | descriptive, no confidence intervals');ax.tick_params(axis='x',rotation=15);save(fig,args.output_dir/'subgroup_auroc_2025.png')
    # DCA: thresholding is a hypothetical utility calculation, not clinical validation.
    thresholds=np.arange(.05,.51,.01);prev=y.mean();n=len(y)
    rows=[]
    for t in thresholds:
        pred=p>=t;tp=np.sum(pred & (y==1));fp=np.sum(pred & (y==0))
        nb=tp/n-fp/n*t/(1-t);all_nb=prev-(1-prev)*t/(1-t)
        rows.append({'threshold':t,'model_net_benefit':nb,'treat_all_net_benefit':all_nb,'treat_none_net_benefit':0.})
    dca=pd.DataFrame(rows);dca.to_csv(args.output_dir/'decision_curve_exploratory.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,5));ax.plot(dca.threshold,dca.model_net_benefit,label='Model');ax.plot(dca.threshold,dca.treat_all_net_benefit,label='Treat all');ax.axhline(0,color='gray',linestyle='--',label='Treat none');ax.set(xlabel='Hypothetical decision threshold',ylabel='Net benefit',title='Exploratory decision curve | NOT clinically validated');ax.legend();save(fig,args.output_dir/'decision_curve_exploratory.png')
    print('Saved charts and tables:',args.output_dir,'n_train=',len(train),'n_test=',len(test))
    print('AUROC=',roc_auc_score(y,p),'AP=',average_precision_score(y,p),'Brier=',brier_score_loss(y,p))
if __name__=='__main__':main()
