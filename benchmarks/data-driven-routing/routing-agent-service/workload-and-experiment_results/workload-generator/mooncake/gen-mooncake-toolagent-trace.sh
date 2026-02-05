#!/bin/bash

# Generate workload with TEXT output using English dictionary (recommended)
python mooncake_workload_generator.py \
  --mooncake-trace input_trace/Mooncake_toolagent_trace.jsonl \
  --target-avg-rps 10 \
  --smoothing-window-seconds 10 \
  --duration-seconds 600 \
  --num-tokens-per-hash-id 100 \
  --max-input-tokens 10000 \
  --output-length-scale 0.5 \
  --timestamp-distribution poisson \
  --output-format text \
  --text-mode dictionary \
  --output-dir workload-toolagent \
  --seed 42 \
  --generate-plots