#!/bin/bash
#SBATCH --job-name=cancer_pred
#SBATCH --output=out/diag_pred_all_%j.out
#SBATCH --error=out/diag_pred_all_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Cancer Prediction Pipeline - XGBoost and LightGBM
# Usage: sbatch run_prediction.sbatch

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Create output directory if needed
mkdir -p out


# Load modules 
module load miniforge  
module load cuda/13.0.1  


echo ""
echo "=========================================="
echo "Running XGBoost Model (1-year)"
echo "=========================================="
python run_prediction.py \
    --model xgboost \
    --prediction_horizon 1.0 \
    --threshold-method f1 \
    --gpu

# echo ""
# echo "=========================================="
# echo "Running LightGBM Model (1-year)"
# echo "=========================================="
# python run_prediction.py \
#     --model lightgbm \
#     --prediction_horizon 1.0 \
#     --threshold-method target_recall \
#     --gpu

# echo ""
# echo "=========================================="
# echo "Running XGBoost Model (5-year)"
# echo "=========================================="
# python run_prediction.py \
#     --model xgboost \
#     --threshold-method target_recall \
#     --prediction_horizon 5.0 \
#     --gpu

# echo ""
# echo "=========================================="
# echo "Running LightGBM Model (5-year)"
# echo "=========================================="
# python run_prediction.py \
#     --model lightgbm \
#     --prediction_horizon 5.0 \
#     --threshold-method target_recall \
#     --gpu

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
