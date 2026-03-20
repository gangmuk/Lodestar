#!/bin/bash
# Plot data distributions for workload and training data.
# Usage: ./run_plot_data_distribution.sh <workload.jsonl>
#   e.g. ./run_plot_data_distribution.sh ./conversation-2/workload.jsonl
#        ./run_plot_data_distribution.sh ./conversation-2/workload-original.jsonl

set -e

if [ -z "$1" ]; then
    # echo "Usage: $0 <workload.jsonl>"
    # echo "  e.g. $0 ./conversation-2/workload.jsonl"
    echo "Usage: $0 <workload_dir>"
    echo "  e.g. $0 ./conversation-2"
    exit 1
fi

# WORKLOAD_FILE="$1"
WORKLOAD_DIR="$1"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# python3 "${SCRIPT_DIR}/plot_data_distribution.py" "$WORKLOAD_FILE"
python3 "${SCRIPT_DIR}/plot_data_distribution.py" "$WORKLOAD_DIR/workload.jsonl" & python3 "${SCRIPT_DIR}/plot_data_distribution.py" "$WORKLOAD_DIR/workload-original.jsonl" & wait
