#!/bin/bash
#SBATCH --job-name=survival_analysis
#SBATCH --output=logs/cancer_sis_only_%j.out
#SBATCH --error=logs/cancer_sis_only_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Usage: sbatch run.sbatch

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="


# Load modules 
module load miniforge  
module load cuda/13.0.1  

python cancer_survival_analysis.py