# Diabetes Classification and Subgroup AUROC Analysis

## Project Overview

This project investigates diabetes classification and subgroup discrimination using CDC Behavioral Risk Factor Surveillance System (BRFSS) data from 2024 and 2025.

A logistic regression model was trained on 2024 data and evaluated on 2025 data. The analysis examines survey-weighted predictive performance, age-related AUROC decomposition, BMI subgroup differences, and the consequences of excluding age from the prediction model.

## Dataset

- **Training:** BRFSS 2024 — 441,934 eligible observations.
- **Evaluation:** BRFSS 2025 — 342,539 eligible observations.
- **Outcome:** Self-reported prevalent diabetes.
- **Predictors:** Age group, BMI, sex, exercise, smoking history, and education.

The raw datasets are available from the CDC and are not included in this repository.

## Key Results

| Metric | Full Model | No-Age Model |
|---|---:|---:|
| Weighted AUROC | 0.795546 | 0.687464 |
| Weighted Brier Score | 0.098758 | 0.108311 |

For participants with BMI below 25 or at least 30 and valid age information, the AUROC gap decreased from 0.060787 to 0.029308 after excluding age.

This represents an approximately **51.8% reduction in the observed subgroup AUROC gap**, accompanied by lower overall discrimination.

## Methods

- Logistic regression and missing-data handling
- Survey-weighted performance evaluation
- Within-age and between-age AUROC decomposition
- BMI subgroup comparisons
- Predictor ablation
- Bootstrap sensitivity and training stability analyses

## Interpretation and Limitations

The AUROC decomposition is an exact mathematical accounting identity, not a causal attribution method.

The uncertainty estimates are exploratory and have not been established as fully design-valid confidence intervals for BRFSS.

The 2025 evaluation dataset was reused during project development and is not an untouched final test set.

The analysis concerns self-reported prevalent diabetes rather than future disease incidence.

## Research Status

This repository documents an applied statistical and machine-learning investigation. Methodological novelty and journal-publication readiness have not been established.

## Reproducibility

Source code, dependency specifications, and reproducibility instructions will be organized in this repository. The original CDC datasets must be obtained separately.

## Data Source

Centers for Disease Control and Prevention (CDC), Behavioral Risk Factor Surveillance System (BRFSS).

https://www.cdc.gov/brfss/
## Visual Results

The following figures summarize the predictive performance, subgroup differences, and exploratory statistical analyses.

### 1. Model Performance

![Model Performance](01_model_performance.png)

### 2. Weighted ROC Curves

![Weighted ROC Curves](02_weighted_roc.png)

### 3. Weighted Calibration

![Weighted Calibration](03_weighted_calibration.png)

### 4. BMI Subgroup AUROC

![BMI Subgroup AUROC](04_bmi_subgroup_auroc.png)

### 5. AUROC Gap Decomposition

![AUROC Gap Decomposition](05_auroc_decomposition.png)

### 6. Age-Binning Sensitivity

![Age-Binning Sensitivity](06_age_binning_sensitivity.png)

### 7. Exploratory Uncertainty

![Exploratory Uncertainty](07_gap_reduction_uncertainty.png)

**Statistical note:** Uncertainty intervals are exploratory and have not been established as fully design-valid BRFSS confidence intervals.
