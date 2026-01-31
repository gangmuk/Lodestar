#!/bin/bash

set -e

base_dir="/mnt/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/workload-and-experiment_results/NVIDIA-A30/maxTokens_1-maxTokensStd_0/gangmuk-prefix"

target_dir_list=(
    "${base_dir}/SharingRatio71%/rps9-benchmark"
    "${base_dir}/SharingRatio47%/rps8-benchmark"
    "${base_dir}/SharingRatio28%/rps7-benchmark"
    "${base_dir}/SharingRatio9%/rps6-benchmark"
    "${base_dir}/MixedSharingRatio10_30_50_70%/rps9-benchmark"
)

# Verify all directories exist
for target_dir in "${target_dir_list[@]}"; do
    if [ ! -d "${target_dir}" ]; then
        echo "Target directory ${target_dir} does not exist."
        exit 1
    fi
done

echo "=== Step 1: Processing individual workloads ==="
echo ""

# Process each workload (in parallel)
for target_dir in "${target_dir_list[@]}"; do
    echo "Processing ${target_dir}"
    # python compare_routing_strategies.py "${target_dir}" 30 --iteration-from 2 | grep -E "(Saved|metrics CSV)" &
    python compare_routing_strategies.py "${target_dir}" 30 --iteration-from 2 | grep -E "(Saved|metrics CSV)"
done

echo ""
echo "=== Step 2: Merging and plotting all workloads ==="
echo ""

# Merge all metrics and create comparison plots
python merge_and_plot_all_workloads.py "${base_dir}"