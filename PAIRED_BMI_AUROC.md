# Reproduce the paired BMI AUROC analysis

[paired_bmi_auc.py](paired_bmi_auc.py) refits the full and age-omitted logistic models and evaluates both on the same BRFSS respondents. It reproduces the two BMI subgroup estimates and the paired survey-design confidence interval for the difference between the gaps.

## Inputs and environment

Download the SAS Transport archives from the CDC releases for [2024](https://www.cdc.gov/brfss/annual_data/annual_2024.html) and [2025](https://www.cdc.gov/brfss/annual_data/annual_2025.html). Extract the XPT member from each archive. Use the 2024 file for training and the 2025 file for evaluation.

The commands below run from the root of this repository revision with Python 3.12. The script uses the preprocessing and logistic models in the accompanying [run_v11.py](run_v11.py).

```bash
python -m pip install numpy==2.3.5 pandas==2.2.3 scipy==1.17.0 scikit-learn==1.8.0 joblib==1.5.3 threadpoolctl==3.6.0 matplotlib==3.10.8
python paired_bmi_auc.py \
  --training-xpt /path/to/LLCP2024.XPT \
  --evaluation-xpt /path/to/LLCP2025.XPT \
  --output results/brfss_paired_design_results.json
```

| Uncompressed input | Source rows | SHA-256 |
| --- | ---: | --- |
| 2024 XPT | 457,670 | `7d53cf088f858e02bcc029a51cf0408cc68f40a0ebeb02a3c0f73058aab36b1c` |
| 2025 XPT | 356,158 | `d99262f8854f018a600c3538ee78e4308569af700abe6b92b36180336c24dec2` |

## Check the output

In `contrasts.gap_reduction`, the reduction rounds to **0.0315** and the conditional 95% confidence interval rounds to **[0.0184, 0.0446]**. The analysis uses 97,914 respondents with BMI below 25 and 100,215 with BMI of at least 30. The confidence interval reflects evaluation-sample variability for the two fixed fits.
