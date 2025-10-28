#!/bin/bash

# Optional: Set prefix hit ratio (default: 1.0 means 100% shared prefix)
# You can pass it as the first argument to this script

prefix_ratio_list=(
    0.0
    0.2
    0.4
    0.6
    0.8
    1.0
)

input_length_list=(
    2000
    4000
    8000
    16000
)

for prefix_ratio in "${prefix_ratio_list[@]}"; do
    echo "======================================"
    echo "Prefix Hit Ratio: ${prefix_ratio}"
    echo "======================================"

    for input_length in "${input_length_list[@]}"; do
        echo "**************************************"
        echo "Input length: ${input_length}"
        echo "**************************************"
        python microbenchmark.py ${input_length} ${prefix_ratio} | grep -e "x-timing-ttft-ms" -e "microbenchmark"
    done
done