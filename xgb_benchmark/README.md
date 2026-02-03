# XGBoost Diagnosis Benchmark

Comprehensive benchmark for XGBoost-based diagnosis prediction across multiple conditions, time horizons, and feature sets.

## Overview

This module systematically evaluates XGBoost classification performance for predicting various diagnoses using UK Biobank data. It generates a complete table of AUC scores across all combinations of:

- **11 Diagnosis types**: colorectal_cancer, lung_cancer, stomach_cancer, alzheimers, copd, hhd, ischemia, kidney, lower_resp, stroke, t2d
- **5 Time points**: 0yr (current), 1yr, 2yr, 5yr, 10yr
- **6 Feature sets**: Various combinations of demographics, protein, and blood biomarkers

## Time Point Definitions

| Time Point | Definition |
|------------|------------|
| 0yr (current) | Diagnosed at or before baseline (time_to_diagnosis ≤ 0) |
| 1yr | Not diagnosed at baseline, diagnosed within 1 year |
| 2yr | Not diagnosed at baseline, diagnosed within 2 years |
| 5yr | Not diagnosed at baseline, diagnosed within 5 years |
| 10yr | Not diagnosed at baseline, diagnosed within 10 years |

## Feature Sets

| Feature Set | Description |
|-------------|-------------|
| `demo_protein` | Demographics (6) + All protein features (~2920) |
| `demo_blood` | Demographics (6) + All blood biomarkers (61) |
| `demo_protein_blood` | Demographics + All protein + All blood |
| `demo_protein_top50` | Demographics + Top 50 protein features (by importance) |
| `demo_blood_top50` | Demographics + Top 50 blood features (or all if <50) |
| `demo_protein_blood_top50` | Demographics + Top 50 combined features |

## Usage

```bash
# Run with default settings
python run_xgb_benchmark.py

# Specify output directory
python run_xgb_benchmark.py --output-dir ./results

# Specify data path
python run_xgb_benchmark.py --data-path /path/to/data
```

Or use the provided shell script:
```bash
bash run_benchmark.sh
```

## Output Files

The script generates several output files:

| File | Description |
|------|-------------|
| `benchmark_results_full_*.csv` | Complete results with all columns |
| `benchmark_results_latest.csv` | Latest results (overwritten each run) |
| `benchmark_pivot_by_features_*.csv` | AUC table pivoted by feature sets |
| `benchmark_pivot_by_time_*.csv` | AUC table pivoted by time points |
| `benchmark_summary_by_features_*.csv` | Summary statistics by feature set |
| `benchmark_summary_by_time_*.csv` | Summary statistics by time point |
| `benchmark_summary_by_diagnosis_*.csv` | Summary statistics by diagnosis |
| `top_features_per_condition.csv` | Top 50 features for each condition |

## XGBoost Parameters

The model uses the following fixed parameters for reproducibility:

```python
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'enable_categorical': True,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.01,
    'verbosity': 0
}
```

## Reproducibility

All random seeds are set explicitly:
- `RANDOM_STATE = 42`
- NumPy seed
- Python random seed
- XGBoost random_state parameter

## Requirements

- Python 3.8+
- pandas
- numpy
- xgboost
- scikit-learn
