import sys, unittest
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_v11 import transform, model_for
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
if __name__=='__main__':unittest.main()
