#!/bin/bash
#SBATCH --job-name=xgb_benchmark_disease
#SBATCH --output=logs/xgb_benchmark_disease_%j.out
#SBATCH --error=logs/xgb_benchmark_disease_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00


OUTPUT_DIR="results/"


echo "=============================================="
echo "XGBoost Benchmark for Disease"
echo "=============================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Load modules 
module load miniforge  
module load cuda/13.0.1  

# Run benchmark
python run_xgb_benchmark.py --output-dir "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "XGBoost benchmark for cancer complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
