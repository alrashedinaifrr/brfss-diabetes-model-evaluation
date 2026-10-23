# Technical report — BRFSS diabetes classification

## Objective
Evaluate logistic-regression discrimination for prevalent self-reported diabetes, compare BMI subgroup AUROC, and study how age-group information changes the observed BMI discrimination gap.

## Data and outcome
Training: BRFSS 2024 (441,934 eligible records). Evaluation: BRFSS 2025 (342,539 eligible records). Eligibility is based on valid DIABETE4 yes/no responses and positive survey weight. The binary outcome is DIABETE4=1 (reported diabetes) versus DIABETE4=3 (reported no diabetes). This is a cross-sectional self-report endpoint.

## Predictors and model
Age group, BMI, sex, exercise, smoking history and education. Primary implementation uses a scikit-learn logistic regression with numeric imputation/scaling and categorical imputation/one-hot encoding. Sensitivity checks compare median imputation, missingness indicators and complete-case training. Full and no-age model evaluations share the same eligible 2025 evaluation population for overall performance.

## Reproduced core results

| Primary model / cohort | Training N | Evaluation N | Weighted AUROC | Weighted Brier |
|---|---:|---:|---:|---:|
| Median imputation / all eligible | 441,934 | 342,539 | 0.795546 | 0.098758 |
| Missingness indicator / all eligible | 441,934 | 342,539 | 0.795419 | 0.098768 |
| Median imputation / common complete case | 441,934 | 298,869 | 0.799073 | 0.099553 |
| Missingness indicator / common complete case | 441,934 | 298,869 | 0.799066 | 0.099585 |
| Complete-case training / common complete case | 380,516 | 298,869 | 0.799084 | 0.099574 |

The complete-case row must not be directly ranked against all-eligible rows because the evaluation cohort differs.

## Age ablation

| Model | Weighted AUROC (all eligible) | Weighted Brier (all eligible) |
|---|---:|---:|
| Full | 0.795546 | 0.098758 |
| No age | 0.687464 | 0.108311 |

On the age-complete BMI<25 and BMI>=30 cohort (N=198,129), full-model AUROC was 0.806129 versus 0.745342; no-age AUROC was 0.650246 versus 0.620937. Hence the observed gap fell from 0.060787 to 0.029308, a decrease of 0.031479 (51.8%).

## Exact AUROC decomposition
The survey-weighted AUROC is the weighted probability that a case is scored higher than a noncase (with half-credit for ties). Partition positive-negative pairs by whether the two individuals share an age category:

`AUC = pi_same * AUC_same + pi_different * AUC_different`.

The BMI-group AUROC gap can be expressed as discrimination and pair-mass-composition components using a symmetric two-group algebraic decomposition. With 13 age categories:

| Full-model BMI<25 minus BMI>=30 component | Estimate |
|---|---:|
| Within-age discrimination | 0.000205 |
| Between-age discrimination | 0.058936 |
| Composition | 0.001646 |
| Total gap | 0.060787 |

The between-age discrimination term represents about 96.95% of this **particular decomposition** of the observed gap. This ratio changes when age categories are merged: approximately 89.75% (7 categories), 90.57% (4) and 83.40% (2). Thus the decomposition is partition-dependent, not an invariant causal attribution.

## Exploratory uncertainty
A paired evaluation-sample bootstrap with 400 replicates estimated the gap reduction at 0.031479 (bootstrap SE 0.006693; percentile interval [0.018888, 0.044492]). A separate 100-replicate training-sample stability exercise yielded SD about 0.000443 for gap reduction. A quadrature combination produced approximate SE 0.006708 and interval [0.018331, 0.044627]. This combination assumes negligible covariance across the separately assessed sources and does not establish design-correct coverage for BRFSS.

## Verification
A separate GitHub ZIP extraction and execution in Colab produced training N=441,934 and evaluation N=342,539 and reproduced the primary weighted AUROC and Brier score to six decimals. Dependency imports and syntax checks passed. Two basic pytest tests passed. These checks establish a working core pipeline, not end-to-end replication of all V13–V16 notebook analyses.

## Limitations
1. Survey design variance estimation remains uncertain, particularly because the apparent PSU identifiers may not represent actual multi-respondent sampling clusters in the extracted data.
2. The 2025 evaluation data were reused during development.
3. Outcome and covariates are survey-reported, and the target is prevalence rather than future incidence.
4. Excluding age is an ablation study, not a fairness intervention.
5. The decomposition is descriptive and algebraic; methodological novelty has not been demonstrated.
6. Clinical calibration and clinical decision usefulness require additional evaluation.
