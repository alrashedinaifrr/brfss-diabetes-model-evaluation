# 2026 doctoral research portfolio extension — BRFSS diabetes model evaluation

This directory contains the **portfolio-specific extensions** to the original analysis. It does not replace the existing benchmark notebooks, original code, or full research manuscript.

## Scope and main outcomes

- 2024 BRFSS training and 2025 BRFSS evaluation with identical eligible evaluation respondents for the full and no-age models.
- Survey-weighted AUROC and Brier paired contrasts, with conditional Taylor (survey-PSU / stratum) confidence intervals and independent bootstrap verification.
- Case/control-specific age-standardized AUROC across extreme BMI groups (BMI <25 vs BMI >=30) using pooled, low-BMI, high-BMI, and uniform reference distributions; sensitivity estimates and intervals are provided.
- BMI-specific Brier scores and weighted calibration intercept/slope.

**Selected checked estimates**: full AUROC 0.795546; no-age AUROC 0.687464; full-minus-no-age AUROC 0.108083 (95% CI [0.103778, 0.112387]). Removing age increased Brier by 0.009553 (95% CI [0.009131, 0.009976]). The extreme-BMI full-model AUROC gap was 0.060787 before standardization and 0.007325 for the pooled reference. A uniform age reference gives a negative contrast; results depend on the target population.

**Interpretation**: sampling uncertainty conditions on the fitted 2024 models and public-use 2025 survey-design approximation. These observational analyses do not show causal effects or a clinical fairness improvement, and the 2025 evaluation sample was revisited during analysis.

## Reproduce from CDC public-use transport files

Obtain the original `LLCP2024.XPT` and `LLCP2025.XPT` transport files from the CDC BRFSS annual datasets; do **not** commit participant-level source files or the scored parent to this public repository. Put files in `code/study1/data/` relative to this directory. The exact XPT SHA-256 values from the audited run are:

- `LLCP2024.XPT`: `7d53cf088f858e02bcc029a51cf0408cc68f40a0ebeb02a3c0f73058aab36b1c`
- `LLCP2025.XPT`: `d99262f8854f018a600c3538ee78e4308569af700abe6b92b36180336c24dec2`

Run from this directory after `pip install -r requirements.txt`:

```bash
python code/study1/verify_scoring_2026.py
python code/study1/study1_extension.py --train-xpt code/study1/data/LLCP2024.XPT --evaluation-xpt code/study1/data/LLCP2025.XPT --out-dir results/recomputed_study1
python code/study1/age_reference_CI_extension.py
python code/study1/bootstrap_alternative_age_reference.py
PYTHONPATH=code/study1 pytest -q code/study1/test_study1_reanalysis.py code/study1/tests
```

Results in `results/` are aggregate tables and numerical audit checks, **not** respondent-level data. The detailed LaTeX study-body source under `docs/` is a manuscript fragment and is not standalone compilable without the main portfolio's LaTeX preamble, figures, and bibliography.

## Evidence and limits

The research report, code and result files should be read together. Confidence intervals are pointwise unless explicitly adjusted, and the age reference is a stated choice rather than an invariant feature of the data. Exact reproduction requires the original 2024/2025 CDC files, matching software versions, and the stated cleaning/survey-design conventions.

**Research context**: [Full Financial Volatility Forecasting extension](https://github.com/alrashedinaifrr/financial-volatility-forecasting/tree/main/portfolio_2026) (when merged).
