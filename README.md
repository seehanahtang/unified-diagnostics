# UK Biobank Cancer Prediction

This repository contains code for cancer prediction and survival analysis using UK Biobank data, including proteomics and genomics features.

## Project Structure

```
ukbiobank/
├── preprocessing/          # Data preprocessing notebooks
├── feature_selection/      # Feature selection scripts and outputs
├── prediction/             # XGBoost cancer prediction pipeline
├── survival_analysis/      # Survival analysis with random survival forests
├── genomics/               # Genomics variant analysis
├── notebooks/              # Exploratory analysis notebooks
└── utils/                  # Utility scripts
```

## Directories

### `preprocessing/`
Data preprocessing notebooks for UK Biobank data:
- `ukb_diagnoses_preprocess.ipynb` - Process diagnosis codes
- `ukb_cancer_preprocess.ipynb` - Cancer-specific data preprocessing
- `breast_cancer_preprocess.ipynb` - Breast cancer data preparation
- `sis_data_preprocess.ipynb` - SIS data preprocessing
- `bgen_parquet_preprocess.ipynb` - Genomic data format conversion
- `protein_to_bed.ipynb` - Protein data to BED format conversion

### `feature_selection/`
Feature selection for cancer prediction models:
- `protein_feature_selection.py` - Protein feature selection
- `ukb_feature_selection.py` - UK Biobank feature selection
- `features/` - Selected features for different cancer types

### `prediction/`
XGBoost-based cancer prediction pipeline:
- `config.py` - Configuration settings
- `data_loader.py` - Data loading utilities
- `models.py` - Model definitions
- `evaluator.py` - Model evaluation metrics
- `predict.py` - Inference script
- `run_prediction.py` - Training script
- `run_prediction_ensemble.py` - Ensemble training

### `survival_analysis/`
Survival analysis using random survival forests:
- `cancer_survival_analysis.py` - Main survival analysis
- `survival_analysis.py` - Core survival analysis functions
- `survival_analysis.ipynb` - Interactive analysis notebook

### `genomics/`
Genomics and variant analysis:
- `train_xgboost_genomics.py` - XGBoost with genomic features
- `train_xgboost_combined.py` - Combined proteomics + genomics model
- `run_xgb_selected_snps.ipynb` - SNP-based analysis
- `hugo_genes_GRCh38.bed` - Gene reference file

### `notebooks/`
Exploratory and experimental notebooks:
- `cancer_prediction.ipynb` - Cancer prediction experiments
- `ukb_gender_exploration.ipynb` - Gender-based analysis
- `biobank_cancer_smote.ipynb` - SMOTE oversampling experiments
- `prediction_genomics.ipynb` - Genomics prediction analysis
- `run_m3h_cancers.ipynb` - M3H model experiments

### `utils/`
Utility scripts:
- `m3h.py` - M3H model implementation
- `tabpfn.py` - TabPFN model runner

## Data

**Note:** Data files are not included in this repository due to UK Biobank data sharing restrictions. You will need to obtain access to UK Biobank data separately.

## Requirements

See `requirements.txt` for dependencies (to be created based on your environment).

## Usage

1. Preprocess data using notebooks in `preprocessing/`
2. Run feature selection with scripts in `feature_selection/`
3. Train prediction models using `prediction/run_prediction.py`
4. Perform survival analysis with `survival_analysis/cancer_survival_analysis.py`

## License

This project is for research purposes. Please ensure compliance with UK Biobank data usage agreements.
