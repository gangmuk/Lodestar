#!/bin/bash

output_dir=workload-toolagent

# Generate workload with TEXT output using English dictionary (recommended)
python mooncake_workload_generator.py \
  --mooncake-trace Mooncake_toolagent_trace.jsonl \
  --target-avg-rps 10 \
  --smoothing-window-seconds 10 \
  --duration-seconds 600 \
  --num-tokens-per-hash-id 100 \
  --max-input-tokens 10000 \
  --output-length-scale 0.5 \
  --timestamp-distribution poisson \
  --output-format text \
  --text-mode dictionary \
  --output-dir $output_dir \
  --seed 42

INPUT="$output_dir/workload.jsonl"
mkdir -p $output_dir/extended
OUTPUT="$output_dir/extended/workload.jsonl"

python3 "extend_workload.py" "$INPUT" "$OUTPUT" --multiplier "4" --seed "42"