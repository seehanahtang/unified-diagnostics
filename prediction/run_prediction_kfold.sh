#!/bin/bash
#SBATCH --job-name=diag_kfold
#SBATCH --output=out/diag_kfold_%j.out
#SBATCH --error=out/diag_kfold_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Diagnosis Prediction Pipeline - XGBoost K-Fold Cross-Validation
# Usage: sbatch run_prediction_kfold.sbatch

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


python run_prediction_kfold.py --prediction_horizon 1.0