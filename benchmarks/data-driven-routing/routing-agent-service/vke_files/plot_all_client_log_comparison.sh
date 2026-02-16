#!/bin/bash

set -e

# base_dir="/mnt/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/workload-and-experiment_results/NVIDIA-A30/maxTokens_1-maxTokensStd_0/gangmuk-prefix/before_latency_optimization"
base_dir="/mnt/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/workload-and-experiment_results/NVIDIA-A30/maxTokens_1-maxTokensStd_0"

target_dir_list=(
    "${base_dir}/gangmuk-prefix/SharingRatio71%/rps4-benchmark/without_bitsandbytes"
    "${base_dir}/gangmuk-prefix/SharingRatio71%/rps6-benchmark/without_bitsandbytes"
    "${base_dir}/gangmuk-prefix/SharingRatio71%/rps8-benchmark/without_bitsandbytes"
    
    "${base_dir}/gangmuk-prefix/SharingRatio47%/rps5-benchmark/without_bitsandbytes"
    "${base_dir}/gangmuk-prefix/SharingRatio47%/rps6-benchmark/without_bitsandbytes"
    "${base_dir}/gangmuk-prefix/SharingRatio47%/rps7-benchmark/without_bitsandbytes"
    "${base_dir}/gangmuk-prefix/SharingRatio47%/rps8-benchmark/without_bitsandbytes"
    "${base_dir}/gangmuk-prefix/SharingRatio47%/rps9-benchmark/without_bitsandbytes"
    "${base_dir}/gangmuk-prefix/SharingRatio47%/rps10-benchmark/without_bitsandbytes"
    
    "${base_dir}/gangmuk-prefix/SharingRatio28%/rps6-benchmark/without_bitsandbytes"
    "${base_dir}/gangmuk-prefix/SharingRatio28%/rps8-benchmark/without_bitsandbytes"
    "${base_dir}/gangmuk-prefix/SharingRatio28%/rps10-benchmark-fail/without_bitsandbytes"
    
    "${base_dir}/gangmuk-prefix/SharingRatio9%/rps6-benchmark/without_bitsandbytes"
    "${base_dir}/gangmuk-prefix/SharingRatio9%/rps8-benchmark/without_bitsandbytes"
    "${base_dir}/gangmuk-prefix/SharingRatio9%/rps10-benchmark-fail/without_bitsandbytes"
    
    "${base_dir}/gangmuk-prefix/MixedSharingRatio10_30_50_70%/rps6-benchmark/without_bitsandbytes"
    "${base_dir}/gangmuk-prefix/MixedSharingRatio10_30_50_70%/rps8-benchmark/without_bitsandbytes"
    # "${base_dir}/gangmuk-prefix/MixedSharingRatio10_30_50_70%/rps10-benchmark/without_bitsandbytes"
    
    ########################################################
    "${base_dir}/mooncake/conversation-2/rps10-benchmark/without_bitsandbytes"
    "${base_dir}/mooncake/conversation-2/rps15-benchmark/without_bitsandbytes"
    "${base_dir}/mooncake/conversation-2/rps20-benchmark/without_bitsandbytes"
    "${base_dir}/mooncake/conversation-2/rps25-benchmark/without_bitsandbytes"

    "${base_dir}/mooncake/toolagent-2/rps10-benchmark/without_bitsandbytes"
    "${base_dir}/mooncake/toolagent-2/rps15-benchmark/without_bitsandbytes"
    "${base_dir}/mooncake/toolagent-2/rps20-benchmark/without_bitsandbytes"
    "${base_dir}/mooncake/toolagent-2/rps25-benchmark/without_bitsandbytes"

    # "${base_dir}/azure/azure_code_poisson/rps25-benchmark"
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
    # python compare_routing_strategies_with_client_log.py "${target_dir}" 30 --iteration-from 2 | grep -E "(Saved|metrics CSV)"
    python compare_routing_strategies_with_client_log.py "${target_dir}" &
done

wait

echo ""
echo "=== Step 2: Merging and plotting all workloads ==="
echo ""

# Merge all metrics and create comparison plots
python merge_and_plot_all_workloads_from_client_log.py "${base_dir}"