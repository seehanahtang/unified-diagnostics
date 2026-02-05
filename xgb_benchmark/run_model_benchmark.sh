#!/bin/bash
#SBATCH --job-name=model_benchmark
#SBATCH --output=logs/model_benchmark_%j.out
#SBATCH --error=logs/model_benchmark_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00


OUTPUT_DIR="results/"


echo "=============================================="
echo "Model Benchmark"
echo "=============================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Load modules 
module load miniforge  
module load cuda/13.0.1  

# Run benchmark
python run_model_benchmark.py

echo ""
echo "=============================================="
echo "Model benchmark complete!"
echo "=============================================="
