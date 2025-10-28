#!/bin/bash

# Script to run latency predictor explainability analysis
# Usage: ./run_latency_predictor_explainability.sh <final_model_dir>

if [ -z "$1" ]; then
    echo "Usage: $0 <final_model_dir>"
    echo "Example: $0 /mnt/data/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/workload-and-experiment_results/SharingRatio28%/latency_predictor_ttft-trained_on_merged-data_all-iter10-20251011_063005/final_model"
    exit 1
fi

FINAL_MODEL_DIR="$1"

# Check if the model directory exists
if [ ! -d "$FINAL_MODEL_DIR" ]; then
    echo "Error: Model directory does not exist: $FINAL_MODEL_DIR"
    exit 1
fi

# Check if latency_predictor.pth exists
if [ ! -f "$FINAL_MODEL_DIR/latency_predictor.pth" ]; then
    echo "Error: latency_predictor.pth not found in $FINAL_MODEL_DIR"
    exit 1
fi

echo "=========================================="
echo "Latency Predictor Explainability Analysis"
echo "=========================================="
echo "Model directory: $FINAL_MODEL_DIR"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the explainability script
python3 "$SCRIPT_DIR/latency_predictor_explainability.py" \
    --final_model_dir "$FINAL_MODEL_DIR" \
    --samples 32 \
    --seed 42

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Explainability analysis completed successfully!"
    echo "Report saved to: $FINAL_MODEL_DIR/xai_report/xai_report.pdf"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ Explainability analysis failed!"
    echo "=========================================="
    exit 1
fi



























































