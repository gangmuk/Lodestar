#!/bin/bash
# Compare original and extended workloads side by side.
# Usage: ./run_compare_workloads.sh <target_dir>
#   e.g. ./run_compare_workloads.sh ./conversation-2

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <target_dir>"
    echo "  e.g. $0 ./conversation-2"
    exit 1
fi

TARGET_DIR="$1"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ORIGINAL="${TARGET_DIR}/workload-original.jsonl"
EXTENDED="${TARGET_DIR}/workload.jsonl"

python3 "${SCRIPT_DIR}/compare_workloads.py" "$ORIGINAL" "$EXTENDED"
