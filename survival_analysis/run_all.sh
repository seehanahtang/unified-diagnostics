#!/bin/bash
#SBATCH --job-name=survival_analysis
#SBATCH --output=logs/cancers_with_risk_sis_%j.out
#SBATCH --error=logs/cancers_with_risk_sis_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Usage: sbatch run_all.sh

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="


# Load modules 
module load miniforge  
module load cuda/13.0.1  

python survival_analysis-Copy1.py