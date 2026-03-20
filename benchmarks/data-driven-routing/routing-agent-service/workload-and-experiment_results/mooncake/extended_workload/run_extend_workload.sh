#!/bin/bash
# Generate extended workload from original workload.
# Usage: ./run_extend_workload.sh <target_dir> [multiplier] [seed]
#   e.g. ./run_extend_workload.sh ./conversation-2
#        ./run_extend_workload.sh ./conversation-2 4 42

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <target_dir> [multiplier] [seed]"
    echo "  e.g. $0 ./conversation-2"
    exit 1
fi

TARGET_DIR="$1"
MULTIPLIER="${2:-4}"
SEED="${3:-42}"

INPUT="${TARGET_DIR}/workload-original.jsonl"
OUTPUT="${TARGET_DIR}/workload.jsonl"

python3 "extend_workload.py" "$INPUT" "$OUTPUT" --multiplier "$MULTIPLIER" --seed "$SEED"

python3 "compare_workloads.py" "$INPUT" "$OUTPUT" & python3 "plot_data_distribution.py" "$OUTPUT" & wait