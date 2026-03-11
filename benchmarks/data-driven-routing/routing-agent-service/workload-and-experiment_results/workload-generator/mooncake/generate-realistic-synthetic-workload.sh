#!/bin/bash

python realistic_workload_generator.py --rps-scale 1 --num-tokens-per-hash-id 100 --output-length-scale 1.0 --mooncake-trace Mooncake_synthetic_trace.jsonl --output-dir synthetic_realistic_workload_tokenized --duration 1200 --vocab-csv vocab.csv