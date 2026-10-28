import sys, unittest
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_v11 import (
 transform, model_for, weighted_decision_analysis,
 weighted_age_auc_decomposition, evaluate,
)
class TestV11(unittest.TestCase):
 def test_transform(self):
  raw=pd.DataFrame({'DIABETE4':[1,3,2], '_LLCPWT':[2.,3.,4.], '_AGEG5YR':[4.,14.,3.], '_BMI5':[2500.,9999.,3000.], 'SEXVAR':[1.,2.,1.], 'EXERANY2':[1.,2.,1.], 'SMOKE100':[1.,2.,1.], 'EDUCA':[4.,6.,5.]})
  d=transform(raw)
  self.assertEqual(len(d),2)
  self.assertEqual(d.target.tolist(),[1,0])
  self.assertTrue(np.isnan(d.bmi.iloc[1]))
  self.assertTrue(np.isnan(d.age_group.iloc[1]))
 def test_pipeline_construction(self):
  for strategy in ['median','indicator','complete_case']:
   self.assertEqual(model_for(strategy).steps[-1][0],'lr')
 def test_weighted_decision_metrics(self):
  result=weighted_decision_analysis(
   y=[1,1,0,0], probability=[.8,.4,.6,.1], weights=[2.,1.,3.,4.], thresholds=[.5]
  )[0]
  self.assertAlmostEqual(result['weighted_sensitivity'],2/3)
  self.assertAlmostEqual(result['weighted_specificity'],4/7)
  self.assertAlmostEqual(result['weighted_precision'],2/5)
  self.assertAlmostEqual(result['weighted_flagged_fraction'],5/10)
  self.assertAlmostEqual(result['net_benefit_model'],-0.1)
  self.assertAlmostEqual(result['net_benefit_treat_all'],-0.4)
 def test_decision_threshold_validation(self):
  with self.assertRaises(ValueError):
   weighted_decision_analysis([0,1],[.2,.8],[1,1],[0.0])
  with self.assertRaises(ValueError):
   weighted_decision_analysis([0,.5],[.2,.8],[1,1],[.5])
 def test_age_pair_auc_decomposition_reconstructs_total(self):
  result=weighted_age_auc_decomposition(
   y=[0,1,0,1,0,1],
   score=[.1,.9,.4,.7,.2,.8],
   weight=[1,2,1,1,2,1],
   age_group=[1,1,2,2,3,3],
  )
  self.assertAlmostEqual(
   result['within_age_pair_share']*result['within_age_auc']
   + result['between_age_pair_share']*result['between_age_auc'],
   result['overall_auc'],
  )
  self.assertLess(result['reconstruction_error'],1e-12)
 def test_evaluate_returns_threshold_analysis(self):
  rng=np.random.default_rng(7)
  def cohort(n):
   age=rng.integers(1,14,n).astype(float)
   bmi=rng.normal(28,5,n)
   y=(rng.random(n)<(0.08+age*.015)).astype(int)
   return pd.DataFrame({
    'target':y,'age_group':age,'bmi':bmi,
    'sex':rng.integers(1,3,n).astype(float),
    'exercise':rng.integers(1,3,n).astype(float),
    'smoked100':rng.integers(1,3,n).astype(float),
    'education':rng.integers(1,7,n).astype(float),
    '_LLCPWT':rng.uniform(0.5,2,n),
   })
  train,test=cohort(160),cohort(80)
  result=evaluate(train,test,decision_thresholds=[.1,.3])
  full_decisions=[r for r in result['decision_analysis'] if r['model']=='full']
  self.assertEqual([r['threshold'] for r in full_decisions],[.1,.3])
  self.assertIn('net_benefit_model',full_decisions[0])
  self.assertEqual(result['n_test_2025'],80)
  self.assertIn('weighted_observed_prevalence',result['results'][0])
  self.assertIn('without_age',{r['model'] for r in result['results']})
  self.assertTrue(result['bmi_subgroup_auroc'])
  self.assertLess(max(r['reconstruction_error'] for r in result['bmi_gap_decomposition']),1e-9)
if __name__=='__main__':unittest.main()
