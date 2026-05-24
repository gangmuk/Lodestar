#!/bin/bash

set -e

# base_dir="/mnt/data/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/workload-and-experiment_results/NVIDIA-A30/use_given_output_length"
# target_dir_list=(
#     "${base_dir}/gangmuk-prefix/SharingRatio71%/rps4-benchmark"
#     # "${base_dir}/gangmuk-prefix/SharingRatio47%/rps5-benchmark"
#     "${base_dir}/gangmuk-prefix/SharingRatio28%/rps5-benchmark"
#     "${base_dir}/gangmuk-prefix/SharingRatio9%/rps5-benchmark"
#     "${base_dir}/gangmuk-prefix/MixedSharingRatio10_30_50_70%/rps5-benchmark"

#     "${base_dir}/mooncake/conversation-2/rps5-benchmark"
#     "${base_dir}/mooncake/toolagent-2/rps7-benchmark"

#     "${base_dir}/azure/azure_code/rps20-benchmark"
#     "${base_dir}/azure/azure_conv-singleturn/rps20-benchmark"
#     "${base_dir}/azure/azure_conv-multiturn/rps20-benchmark"

# )

base_dir="/mnt/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/workload-and-experiment_results/NVIDIA-A30/maxTokens_1-maxTokensStd_0"
target_dir_list=(
    "${base_dir}/gangmuk-prefix/SharingRatio71%/rps10-benchmark"
    # "${base_dir}/gangmuk-prefix/SharingRatio47%/rps9-benchmark"
    "${base_dir}/gangmuk-prefix/SharingRatio28%/rps8-benchmark"
    "${base_dir}/gangmuk-prefix/SharingRatio9%/rps6-benchmark"
    "${base_dir}/gangmuk-prefix/MixedSharingRatio10_30_50_70%/rps9-benchmark"

    # from wisc
    "${base_dir}/gangmuk-prefix/SharingRatio71%/from_wisc/rps10-benchmark"
    "${base_dir}/gangmuk-prefix/SharingRatio47%/from_wisc/rps9-benchmark"
    "${base_dir}/gangmuk-prefix/SharingRatio28%/from_wisc/rps8-benchmark"
    "${base_dir}/gangmuk-prefix/SharingRatio9%/from_wisc/rps6-benchmark"
    "${base_dir}/gangmuk-prefix/MixedSharingRatio10_30_50_70%/from_wisc/rps9-benchmark"

    "${base_dir}/mooncake/conversation-2/rps20-benchmark/from_wisc"
    "${base_dir}/mooncake/conversation-2/rps20-benchmark/clnode"
    "${base_dir}/mooncake/toolagent-2/rps20-benchmark/from_wisc"
    "${base_dir}/mooncake/toolagent-2/rps20-benchmark/clnode"

    "${base_dir}/azure/azure_code/rps20-benchmark"
    "${base_dir}/azure/azure_code/rps20-profiling"
    "${base_dir}/azure/azure_conv-singleturn/rps20-benchmark"
    "${base_dir}/azure/azure_conv-multiturn/rps20-benchmark"
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
    # python compare_routing_strategies.py "${target_dir}" | grep -E "(Saved|metrics CSV)"
    python compare_routing_strategies.py "${target_dir}" &
done

wait

# Merge all metrics and create comparison plots
echo "=== Step 2: Merging and plotting all workloads ==="
python merge_and_plot_all_workloads.py "${target_dir_list[@]}"