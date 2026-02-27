#!/bin/bash
#SBATCH --job-name=rsf_disease
#SBATCH --output=logs/rsf_disease_%j.out
#SBATCH --error=logs/rsf_disease_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

# Usage: sbatch run_all.sh

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="


# Load modules 
module load miniforge  
module load cuda/13.0.1  

# python survival_analysis_diag.py
python survival_analysis_diag.py