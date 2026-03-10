#!/bin/bash
#SBATCH --job-name=genomics_summary
#SBATCH --output=genomics_summary_%j.out
#SBATCH --error=genomics_summary_%j.err
#SBATCH --time=010:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --partition=mit_normal

# Load modules
module load miniforge

# Use fewer workers to avoid OOM: each worker loads many parquet files per chromosome.
# 8 workers with 64G gives ~8G per process. For 16 workers use e.g. --mem=128G.
WORKERS=${SLURM_CPUS_PER_TASK:-4}

echo "Running genomics summary: workers=$WORKERS"
python run_genomics_summary.py --workers "$WORKERS"
echo "Exit code: $?"
