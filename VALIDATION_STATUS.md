# Validation status

| Check | Status | Evidence and scope |
|---|---|---|
| Original CDC archives extracted | PASS | Both 2024 and 2025 XPT files extracted in Colab; filenames normalized |
| Repository copied into independent workspace | PASS | GitHub ZIP unpacked and required files found |
| Dependencies installed and imports checked | PASS | Installation exit code 0; NumPy, pandas, SciPy, scikit-learn, Matplotlib, statsmodels imported |
| Python syntax | PASS | Nine source/test scripts passed syntax checks |
| Automated tests | PASS (limited) | Two tests in `test_v11.py` passed; not a comprehensive test suite |
| Core pipeline run | PASS | Exit code 0; training N=441,934, evaluation N=342,539 |
| Core weighted metrics | PASS (6 d.p.) | Median/all-eligible AUROC 0.795546 and Brier 0.098758 reproduced |
| Complete V15/V16 workflow independently reproduced | NOT VERIFIED | Notebook analyses and derived figures not all independently regenerated from the uploaded repository |
| Survey-design-correct uncertainty | NOT ESTABLISHED | Exploratory variance methods only |
| Untouched external test | NOT AVAILABLE | 2025 reused during development |
| Clinical validation | NOT ESTABLISHED | No prospective or clinical deployment validation |
| Publication-level originality | NOT ESTABLISHED | Requires targeted literature review and novel contribution |

**Important:** A PASS indicates only the specifically named check, not overall clinical or research validity.
