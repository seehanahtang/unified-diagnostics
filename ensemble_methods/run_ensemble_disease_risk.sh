#!/bin/bash
#SBATCH --job-name=ensemble_disease_risk
#SBATCH --output=logs/ensemble_disease_risk_%j.out
#SBATCH --error=logs/ensemble_disease_risk_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00


OUTPUT_DIR="results/"


echo "=============================================="
echo "Ensemble Methods for Disease Risk"
echo "=============================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Load modules 
module load miniforge  
module load cuda/13.0.1  

# Run ensemble methods
python3 -u run_ensemble_disease_risk.py

echo ""
echo "=============================================="
echo "Ensemble methods for disease risk complete!"
echo "=============================================="

