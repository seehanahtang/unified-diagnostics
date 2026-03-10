#!/bin/bash
#SBATCH --job-name=rsf_disease
#SBATCH --output=logs/rsf_disease_%j.out
#SBATCH --error=logs/rsf_disease_%j.err
#SBATCH --partition=mit_normal  
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="


# Load modules 
module load miniforge  

python run_rsf.py --mode disease