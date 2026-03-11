#!/bin/bash

# workload_name="conversation"
# workload_name="toolagent"
workload_name=$1

if [ -z "${workload_name}" ]; then
  echo "Error: Workload name is required [conversation or toolagent or synthetic]"
  echo "Usage: ./mooncake_generate.sh <workload_name>"
  exit 1
fi

# Generate workload with TEXT output using English dictionary (recommended)
python mooncake_workload_generator.py \
  --mooncake-trace Mooncake_${workload_name}_trace.jsonl \
  --target-avg-rps 10 \
  --smoothing-window-seconds 10 \
  --duration-seconds 600 \
  --num-tokens-per-hash-id 200 \
  --output-length-scale 0.5 \
  --timestamp-distribution poisson \
  --output-format text \
  --text-mode dictionary \
  --output-dir workload-${workload_name} \
  --seed 42 \
  --generate-plots
  # --max-input-tokens 10000 \ # can be controlled in client. We shouldn't control it here. The max input length will depend on GPU model, LLM model, workload.