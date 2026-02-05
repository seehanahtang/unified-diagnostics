#!/bin/bash
#SBATCH --job-name=xgb_excluded
#SBATCH --output=logs/xgb_excluded_%j.out
#SBATCH --error=logs/xgb_excluded_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

OUTPUT_DIR="results/"


echo "=============================================="
echo "XGBoost Excluded Features Experiment"
echo "=============================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Load modules 
module load miniforge  
module load cuda/13.0.1  

# Run benchmark
python run_excluded_features_experiment.py --output-dir "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "Excluded features experiment complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="

