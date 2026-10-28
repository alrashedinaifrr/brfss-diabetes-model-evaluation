# Validation status

| Check | Status | Evidence and scope |
|---|---|---|
| Original CDC archives extracted | PASS | Both 2024 and 2025 XPT files extracted in Colab; filenames normalized |
| Repository copied into independent workspace | PASS | GitHub ZIP unpacked and required files found |
| Dependencies installed and imports checked | PASS | Installation exit code 0; NumPy, pandas, SciPy, scikit-learn, Matplotlib, statsmodels imported |
| Python syntax | PASS | Nine source/test scripts passed syntax checks |
| Automated tests | PASS (limited) | Five tests in `test_v11.py` pass in the current working revision; coverage includes weighted threshold metrics and end-to-end decision-analysis output |
| Core pipeline run | PASS | Exit code 0; training N=441,934, evaluation N=342,539 |
| Core weighted metrics | PASS (6 d.p.) | Median/all-eligible AUROC 0.795546 and Brier 0.098758 reproduced |
| Complete V15/V16 workflow independently reproduced | NOT VERIFIED | Notebook analyses and derived figures not all independently regenerated from the uploaded repository |
| Survey-design-correct uncertainty | NOT ESTABLISHED | Exploratory variance methods only |
| Untouched external test | NOT AVAILABLE | 2025 reused during development |
| Clinical validation | NOT ESTABLISHED | No prospective or clinical deployment validation |
| Publication-level originality | NOT ESTABLISHED | Requires targeted literature review and novel contribution |
| New decision-analysis output on raw CDC cohorts | NOT RUN | CDC XPT files are not present in this repository; the new calculation is covered by synthetic-data tests |

**Important:** A PASS indicates only the specifically named check, not overall clinical or research validity.

## Current working-revision checks (28 October 2026)

| Check | Status | Evidence and scope |
|---|---|---|
| Automated tests | PASS (limited) | Six tests pass, including threshold calculations, model evaluation, and exact age-pair AUROC reconstruction |
| Synthetic full-analysis smoke test | PASS | Full/no-age/weighted-fit comparisons, BMI subgroup analysis, calibration, decision curve, JSON serialization, and four figure files generated |
| Full analysis on CDC 2024/2025 files | NOT RUN | Raw XPT files are not included in this workspace |
| Survey-design-correct uncertainty | NOT ESTABLISHED | Weighted metrics remain point estimates; PSU/stratum variance method still requires validation |
| Untouched final test wave | NOT AVAILABLE | 2025 was reused during development |

The synthetic smoke test checks software flow and table/figure generation only; it does not validate BRFSS estimates or clinical utility.
