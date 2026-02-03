#!/bin/bash
#
# Run XGBoost Diagnosis Benchmark
#
# Usage:
#   bash run_benchmark.sh
#   bash run_benchmark.sh /custom/output/dir
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-$SCRIPT_DIR/results}"

echo "=============================================="
echo "XGBoost Diagnosis Benchmark"
echo "=============================================="
echo "Script directory: $SCRIPT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run benchmark
python "$SCRIPT_DIR/run_xgb_benchmark.py" --output-dir "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "Benchmark complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
