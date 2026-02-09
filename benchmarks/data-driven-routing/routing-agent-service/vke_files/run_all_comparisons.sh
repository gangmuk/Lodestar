#!/bin/bash

set -e

base_dir="/mnt/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/workload-and-experiment_results/NVIDIA-A30/use_given_output_length"

target_dir_list=(
    "${base_dir}/gangmuk-prefix/SharingRatio71%/rps8-profiling"
    "${base_dir}/gangmuk-prefix/SharingRatio47%/rps7-profiling"
    "${base_dir}/gangmuk-prefix/SharingRatio28%/rps7-profiling"
    "${base_dir}/gangmuk-prefix/SharingRatio9%/rps7-profiling"
    "${base_dir}/gangmuk-prefix/MixedSharingRatio10_30_50_70%/rps7-profiling"

    "${base_dir}/mooncake/conversation-2/rps5-profiling"
    "${base_dir}/mooncake/toolagent-2/rps10-profiling"

    "${base_dir}/azure/azure_code-access_sequential-sharingmean_0.3-sharingstd_0.2-numreqpergroup_10/rps15-profiling"
    "${base_dir}/azure/azure_conv-access_sequential-sharingmean_0.1-sharingstd_0.2-numreqpergroup_2/rps15-profiling"
    "${base_dir}/azure/azure_conv-access_sequential-sharingmean_0.5-sharingstd_0.2-numreqpergroup_10/rps15-profiling"
)

# Verify all directories exist
for target_dir in "${target_dir_list[@]}"; do
    if [ ! -d "${target_dir}" ]; then
        echo "Target directory ${target_dir} does not exist."
        exit 1
    fi
done

echo "=== Step 1: Processing individual workloads ==="
for target_dir in "${target_dir_list[@]}"; do
    echo "Processing ${target_dir}"
    python compare_routing_strategies.py "${target_dir}" | grep -E "(Saved|metrics CSV)"
    # python compare_routing_strategies.py "${target_dir}" 30 --iteration-from 2 | grep -E "(Saved|metrics CSV)"
done

# Merge all metrics and create comparison plots
echo "=== Step 2: Merging and plotting all workloads ==="
python merge_and_plot_all_workloads.py "${base_dir}"