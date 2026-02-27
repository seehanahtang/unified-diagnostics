#!/bin/bash
#SBATCH --job-name=disease_pred
#SBATCH --output=out/disease_pred_all_%j.out
#SBATCH --error=out/disease_pred_all_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Diagnosis Prediction Pipeline - XGBoost and LightGBM
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
echo "Running Diagnosis Prediction Pipeline (1-year)"
echo "=========================================="
python diag_prediction.py \
    --prediction_horizon 1.0 \
    --gpu

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
