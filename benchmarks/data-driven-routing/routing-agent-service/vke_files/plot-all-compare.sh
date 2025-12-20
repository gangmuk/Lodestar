#!/bin/bash

set -e

base_dir="/mnt/data/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0"

target_dir_list=(
    "${base_dir}/SharingRatio71%/rps7"
    "${base_dir}/SharingRatio47%/rps6"
    "${base_dir}/SharingRatio28%/rps5"
    "${base_dir}/SharingRatio9%/rps5"
    "${base_dir}/MixedSharingRatio10_30_50_70%/rps6"
)

for target_dir in "${target_dir_list[@]}"; do
    if [ ! -d "${target_dir}" ]; then
        echo "Target directory ${target_dir} does not exist."
        exit 1
    fi
done

for target_dir in "${target_dir_list[@]}"; do
    echo "Processing ${target_dir}"
    python compare_routing_strategies.py "${target_dir}" | grep "Saved" &
done
wait

echo ""
echo "All plots generated successfully!"