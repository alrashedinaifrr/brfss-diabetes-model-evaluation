# Reproducibility status

## Supported clean run

The repository contains one maintained script for the core analysis:

```bash
python -m pip install -r requirements.txt
python -m pytest -q test_v11.py
python run_v11.py --data-dir /path/to/cdc_xpt --output-dir results_v11
```

The input directory must contain CDC BRFSS `LLCP2024.XPT` and `LLCP2025.XPT`. The raw CDC files are not redistributed here. The script creates:

- `analysis_2025.json` and `run_metadata.json`
- CSV tables for model comparison, decision thresholds, calibration, BMI subgroups, and AUROC decomposition
- Four analysis figures under `results_v11/figures/`

The run manifest records SHA-256 hashes, row counts, Python version, and installed library versions. This identifies the source files and runtime used for a given local run.

## Notebook scope

`Health_Risk_Project_Analysis.ipynb` preserves the original 73-cell Colab development history from V11 through V16.7, including its prior analyses and figures. It contains cells dependent on uploaded archives, intermediate private predictions, R survey tools, and earlier outputs, so it is not designed to run top-to-bottom in a fresh runtime. The new `Health_Risk_Project_Colab.ipynb` provides a clean Colab workflow for the maintained `run_v11.py` pipeline; the original analysis notebook remains available as the project history.

## Boundaries

- BRFSS 2025 was repeatedly used during development, so it is not an untouched final test wave.
- The current pipeline reports weighted point estimates but no design-correct confidence intervals. Validation of the PSU/stratum variance method remains open.
- Earlier notebook bootstrap intervals are exploratory and are not regenerated as design-valid uncertainty by the clean runner.
- The study classifies self-reported prevalent diabetes; it does not predict future incidence or establish clinical utility.
- The new pipeline has been tested on synthetic data for software flow and figures, but not rerun on the original XPT files in this workspace.
