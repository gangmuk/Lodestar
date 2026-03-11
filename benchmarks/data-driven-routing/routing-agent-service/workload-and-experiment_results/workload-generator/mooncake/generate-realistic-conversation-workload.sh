#!/bin/bash

python realistic_workload_generator.py --rps-scale 0.5 --num-tokens-per-hash-id 100 --output-length-scale 1.0 --mooncake-trace Mooncake_conversation_trace.jsonl --output-dir conversation_realistic_workload_tokenized --duration 600 --vocab-csv vocab.csv