#!/bin/bash
workload=$1

if [ -z "${workload}" ]; then
  echo "Error: Workload is required [conversation or toolagent or synthetic]"
  echo "Usage: ./generate-realistic-toolagent-workload.sh <workload>"
  exit 1
fi

rps_scale=2
num_tokens_per_hash_id=200
output_length_scale=1
mooncake_trace=Mooncake_${workload}_trace.jsonl
duration=600
vocab_csv=vocab.csv
output_dir=${workload}_realistic_workload_tokenized-rpsscale_${rps_scale}-numtokens_${num_tokens_per_hash_id}-outputscale_${output_length_scale}-duration_${duration}

python realistic_workload_generator.py --rps-scale ${rps_scale} --num-tokens-per-hash-id ${num_tokens_per_hash_id} --output-length-scale ${output_length_scale} --mooncake-trace ${mooncake_trace} --duration ${duration} --vocab-csv ${vocab_csv} --output-dir ${output_dir}