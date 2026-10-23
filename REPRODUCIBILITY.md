# Reproducibility status

## What is included
- `Health_Risk_Project_Analysis.ipynb`: 73-cell Colab analysis history, with all saved cell outputs and execution counts removed. Contains development steps from V11 through V16.7 and figure generation.
- `requirements.txt`: initial Python dependency list, **not version-pinned or validated**.

## Not yet included
- External source scripts and source ZIPs referenced by the notebook, including `run_v11.py`, `visualize_v12.py`, `audit_v13.py` and their dependencies.
- CDC BRFSS 2024 and 2025 XPT source datasets; download these from CDC BRFSS.
- Intermediate private respondent-level predictions and other derived intermediate files.

## Important caveats
The notebook includes sequential, exploratory Colab commands, uploads, paths rooted at `/content/health_project`, R survey validation commands, and dependencies on earlier outputs. Running it top to bottom in a fresh runtime is **not supported yet**. Its cells document the analysis history but do not independently reproduce all published figures without the missing scripts and intermediate artifacts.

2025 was repeatedly used during development. Uncertainty estimates are exploratory, not established as fully survey-design-valid. The study concerns self-reported prevalent diabetes, not future incidence.

## Next action
Export the source directory from Colab (`/content/health_project/v13/health_risk_portfolio_v13/src`) and the other source directories used by V11/V12; review and include them, then construct a single clean pipeline with explicit raw-data download and derived-data generation steps.
