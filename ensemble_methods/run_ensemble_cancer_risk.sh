#!/bin/bash
#SBATCH --job-name=ensemble_cancer_risk
#SBATCH --output=logs/ensemble_cancer_risk_%j.out
#SBATCH --error=logs/ensemble_cancer_risk_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00


OUTPUT_DIR="results/"


echo "=============================================="
echo "Ensemble Methods for Cancer Risk"
echo "=============================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Load modules 
module load miniforge  
module load cuda/13.0.1  

# Run ensemble methods
python3 -u run_ensemble_cancer_risk.py

echo ""
echo "=============================================="
echo "Ensemble methods for cancer risk complete!"
echo "=============================================="

