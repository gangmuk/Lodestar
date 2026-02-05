#!/bin/bash

python mooncake_workload_generator.py \
  --mooncake-trace input_trace/Mooncake_conversation_trace.jsonl \
  --target-avg-rps 10 \
  --smoothing-window-seconds 10 \
  --duration-seconds 600 \
  --num-tokens-per-hash-id 50 \
  --output-length-scale 0.5 \
  --max-input-tokens 10000 \
  --timestamp-distribution poisson \
  --output-format text \
  --text-mode dictionary \
  --output-dir workload-conversation \
  --seed 42 \
  --generate-plots