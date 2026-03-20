#!/bin/bash

set -e

# base_dir="/mnt/projects/aibrix-gangmuk-fixing/benchmarks/data-driven-routing/routing-agent-service/workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0"
# target_dir_list=(
#     # "${base_dir}/gangmuk-prefix/SharingRatio71%/rps4-benchmark/without_bitsandbytes"
#     # "${base_dir}/gangmuk-prefix/SharingRatio71%/rps6-benchmark/without_bitsandbytes"
#     # "${base_dir}/gangmuk-prefix/SharingRatio71%/rps8-benchmark/without_bitsandbytes"
#     ########################################################
#     # "${base_dir}/gangmuk-prefix/SharingRatio47%/rps5-benchmark/without_bitsandbytes"
#     # "${base_dir}/gangmuk-prefix/SharingRatio47%/rps6-benchmark/without_bitsandbytes"
#     # "${base_dir}/gangmuk-prefix/SharingRatio47%/rps7-benchmark/without_bitsandbytes"
#     # "${base_dir}/gangmuk-prefix/SharingRatio47%/rps8-benchmark/without_bitsandbytes"
#     # "${base_dir}/gangmuk-prefix/SharingRatio47%/rps9-benchmark/without_bitsandbytes"
#     # "${base_dir}/gangmuk-prefix/SharingRatio47%/rps10-benchmark/without_bitsandbytes"
#     ########################################################
#     # "${base_dir}/gangmuk-prefix/SharingRatio28%/rps6-benchmark/without_bitsandbytes"
#     # "${base_dir}/gangmuk-prefix/SharingRatio28%/rps8-benchmark/without_bitsandbytes"
#     # "${base_dir}/gangmuk-prefix/SharingRatio28%/rps10-benchmark-fail/without_bitsandbytes"
#     ########################################################
#     # "${base_dir}/gangmuk-prefix/SharingRatio9%/rps6-benchmark/without_bitsandbytes"
#     # "${base_dir}/gangmuk-prefix/SharingRatio9%/rps8-benchmark/without_bitsandbytes"
#     # "${base_dir}/gangmuk-prefix/SharingRatio9%/rps10-benchmark-fail/without_bitsandbytes"
#     ########################################################
#     # "${base_dir}/gangmuk-prefix/MixedSharingRatio10_30_50_70%/rps6-benchmark/without_bitsandbytes"
#     # "${base_dir}/gangmuk-prefix/MixedSharingRatio10_30_50_70%/rps8-benchmark/without_bitsandbytes"
#     # "${base_dir}/gangmuk-prefix/MixedSharingRatio10_30_50_70%/rps10-benchmark/without_bitsandbytes"
#     ########################################################
#     # "${base_dir}/mooncake/conversation-2/rps10-benchmark/without_bitsandbytes"
#     "${base_dir}/mooncake/conversation-2/rps15-benchmark/without_bitsandbytes"
#     "${base_dir}/mooncake/conversation-2/rps20-benchmark/without_bitsandbytes"
#     "${base_dir}/mooncake/conversation-2/rps25-benchmark/without_bitsandbytes"
#     # "${base_dir}/mooncake/conversation-2/rps30-benchmark/without_bitsandbytes"
#     ########################################################
#     # "${base_dir}/mooncake/toolagent-2/rps10-benchmark/without_bitsandbytes"
#     "${base_dir}/mooncake/toolagent-2/rps15-benchmark/without_bitsandbytes"
#     "${base_dir}/mooncake/toolagent-2/rps20-benchmark/without_bitsandbytes"
#     "${base_dir}/mooncake/toolagent-2/rps25-benchmark/without_bitsandbytes"
#     # "${base_dir}/mooncake/toolagent-2/rps30-benchmark/without_bitsandbytes"
#     ########################################################


#     "${base_dir}/mooncake/conversation_realistic_tokenized_rpsscale_1_numtokens_100/rps-1-benchmark/without_bitsandbytes"
#     "${base_dir}/mooncake/conversation_realistic_tokenized_rpsscale_1_numtokens_200/rps-1-benchmark/without_bitsandbytes"
#     "${base_dir}/mooncake/conversation_realistic_tokenized_rpsscale_1_numtokens_300/rps-1-benchmark/without_bitsandbytes"
#     "${base_dir}/mooncake/toolagent_realistic_tokenized_rpsscale_1_numtokens_100/rps-1-benchmark/without_bitsandbytes"
#     "${base_dir}/mooncake/toolagent_realistic_tokenized_rpsscale_1_numtokens_200/rps-1-benchmark/without_bitsandbytes"
#     "${base_dir}/mooncake/synthetic_realistic_tokenized_rpsscale_1_numtokens_100/rps-1-benchmark/without_bitsandbytes"
#     "${base_dir}/mooncake/synthetic_realistic_tokenized_rpsscale_1_numtokens_200/rps-1-benchmark/without_bitsandbytes"
# )





base_dir="/mnt/projects/aibrix-gangmuk-fixing/benchmarks/data-driven-routing/routing-agent-service/workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/gangmuk-prefix"
target_dir_list=(
    "${base_dir}/SharingRatio71%/rps4-benchmark/without_bitsandbytes"
    "${base_dir}/SharingRatio71%/rps6-benchmark/without_bitsandbytes"
    ########################################################
    "${base_dir}/SharingRatio47%/rps4-benchmark/without_bitsandbytes"
    "${base_dir}/SharingRatio47%/rps6-benchmark/without_bitsandbytes"
    ########################################################
    "${base_dir}/SharingRatio28%/rps4-benchmark/without_bitsandbytes"
    "${base_dir}/SharingRatio28%/rps6-benchmark/without_bitsandbytes"
    ########################################################
    "${base_dir}/SharingRatio9%/rps4-benchmark/without_bitsandbytes"
    "${base_dir}/SharingRatio9%/rps6-benchmark/without_bitsandbytes"
    # ########################################################
    "${base_dir}/MixedSharingRatio10_30_50_70%/rps6-benchmark/without_bitsandbytes"
    "${base_dir}/MixedSharingRatio10_30_50_70%/rps8-benchmark/without_bitsandbytes"
    # ########################################################
)



# base_dir="/mnt/projects/aibrix-gangmuk-fixing/benchmarks/data-driven-routing/routing-agent-service/workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake"
# target_dir_list=(
#     ########################################################
#     "${base_dir}/conversation-2/rps8-benchmark/without_bitsandbytes"
#     "${base_dir}/conversation-2/rps9-benchmark/without_bitsandbytes"
#     "${base_dir}/conversation-2/rps10-benchmark/without_bitsandbytes"
#     "${base_dir}/conversation-2/rps11-benchmark/without_bitsandbytes"
#     "${base_dir}/conversation-2/rps12-benchmark/without_bitsandbytes"
#     ########################################################
#     "${base_dir}/toolagent-2/rps8-benchmark/without_bitsandbytes"
#     "${base_dir}/toolagent-2/rps9-benchmark/without_bitsandbytes"
#     "${base_dir}/toolagent-2/rps10-benchmark/without_bitsandbytes"
#     "${base_dir}/toolagent-2/rps11-benchmark/without_bitsandbytes"
#     "${base_dir}/toolagent-2/rps12-benchmark/without_bitsandbytes"
#     ########################################################
#     "${base_dir}/synthetic_realistic_tokenized_rpsscale_1_numtokens_100_head_1.5x_tail_11.0x_after0.7/rps-1-benchmark/without_bitsandbytes"
#     ########################################################
#     "${base_dir}/conversation_realistic_tokenized_rpsscale_1_numtokens_100/rps-1-benchmark/without_bitsandbytes"
#     ########################################################
#     "${base_dir}/toolagent_realistic_tokenized_rpsscale_1_numtokens_100/rps-1-benchmark/without_bitsandbytes"
#     ########################################################
# )




# base_dir="/mnt/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/workload-and-experiment_results/NVIDIA-A30/qwen25-1-5b-instruct/maxTokens_1-maxTokensStd_0"
# target_dir_list=(
#     ########################################################
#     "${base_dir}/gangmuk-prefix/SharingRatio71%/rps20-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio71%/rps30-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio71%/rps40-benchmark/without_bitsandbytes"
#     # "${base_dir}/gangmuk-prefix/SharingRatio71%/rps50-benchmark/without_bitsandbytes"
#     ########################################################
#     "${base_dir}/gangmuk-prefix/SharingRatio47%/rps20-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio47%/rps30-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio47%/rps40-benchmark/without_bitsandbytes"
#     # "${base_dir}/gangmuk-prefix/SharingRatio47%/rps50-benchmark/without_bitsandbytes"
#     ########################################################
#     "${base_dir}/gangmuk-prefix/SharingRatio28%/rps20-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio28%/rps30-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio28%/rps40-benchmark/without_bitsandbytes"
#     # "${base_dir}/gangmuk-prefix/SharingRatio28%/rps50-benchmark/without_bitsandbytes"
#     ########################################################
#     "${base_dir}/gangmuk-prefix/SharingRatio9%/rps20-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio9%/rps30-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio9%/rps40-benchmark/without_bitsandbytes"
#     # "${base_dir}/gangmuk-prefix/SharingRatio9%/rps50-benchmark/without_bitsandbytes"
#     ########################################################
#     "${base_dir}/gangmuk-prefix/MixedSharingRatio10_30_50_70%/rps20-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/MixedSharingRatio10_30_50_70%/rps30-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/MixedSharingRatio10_30_50_70%/rps40-benchmark/without_bitsandbytes"
#     # "${base_dir}/gangmuk-prefix/MixedSharingRatio10_30_50_70%/rps50-benchmark/without_bitsandbytes"
#     # ########################################################
#     # "${base_dir}/mooncake/conversation-2/rps20-benchmark/without_bitsandbytes"
#     # "${base_dir}/mooncake/conversation-2/rps30-benchmark/without_bitsandbytes"
#     # "${base_dir}/mooncake/conversation-2/rps40-benchmark/without_bitsandbytes"
#     # "${base_dir}/mooncake/conversation-2/rps50-benchmark/without_bitsandbytes"
#     # ########################################################
#     # "${base_dir}/mooncake/toolagent-2/rps20-benchmark/without_bitsandbytes"
#     # "${base_dir}/mooncake/toolagent-2/rps30-benchmark/without_bitsandbytes"
#     # "${base_dir}/mooncake/toolagent-2/rps40-benchmark/without_bitsandbytes"
#     # "${base_dir}/mooncake/toolagent-2/rps50-benchmark/without_bitsandbytes"
#     # ########################################################
# )

# base_dir="/mnt/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/workload-and-experiment_results/NVIDIA-A30/qwen25-1-5b-instruct/use_given_output_length"
# target_dir_list=(
#     ########################################################
#     "${base_dir}/gangmuk-prefix/SharingRatio71%/rps10-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio71%/rps20-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio71%/rps30-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio71%/rps40-benchmark/without_bitsandbytes"
#     ########################################################
#     "${base_dir}/gangmuk-prefix/SharingRatio47%/rps10-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio47%/rps20-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio47%/rps30-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio47%/rps40-benchmark/without_bitsandbytes"
#     ########################################################
#     "${base_dir}/gangmuk-prefix/SharingRatio28%/rps10-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio28%/rps20-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio28%/rps30-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio28%/rps40-benchmark/without_bitsandbytes"
#     ########################################################
#     "${base_dir}/gangmuk-prefix/SharingRatio9%/rps10-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio9%/rps20-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio9%/rps30-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio9%/rps40-benchmark/without_bitsandbytes"
#     ########################################################
#     "${base_dir}/gangmuk-prefix/MixedSharingRatio10_30_50_70%/rps10-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/MixedSharingRatio10_30_50_70%/rps20-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/MixedSharingRatio10_30_50_70%/rps30-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/MixedSharingRatio10_30_50_70%/rps40-benchmark/without_bitsandbytes"
#     ########################################################
#     "${base_dir}/mooncake/conversation-2/rps10-benchmark/without_bitsandbytes"
#     "${base_dir}/mooncake/conversation-2/rps20-benchmark/without_bitsandbytes"
#     "${base_dir}/mooncake/conversation-2/rps30-benchmark/without_bitsandbytes"
#     "${base_dir}/mooncake/conversation-2/rps40-benchmark/without_bitsandbytes"
#     "${base_dir}/mooncake/conversation-2/rps50-benchmark/without_bitsandbytes"
#     ########################################################
#     "${base_dir}/mooncake/toolagent-2/rps10-benchmark/without_bitsandbytes"
#     "${base_dir}/mooncake/toolagent-2/rps20-benchmark/without_bitsandbytes"
#     "${base_dir}/mooncake/toolagent-2/rps30-benchmark/without_bitsandbytes"
#     ########################################################
# )


# base_dir="/mnt/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/workload-and-experiment_results/NVIDIA-A30/qwen3-4b-instruct/use_given_output_length"
# target_dir_list=(
#     ########################################################
#     "${base_dir}/gangmuk-prefix/SharingRatio71%/rps2-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio71%/rps4-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio71%/rps6-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio71%/rps8-benchmark/without_bitsandbytes"
#     # "${base_dir}/gangmuk-prefix/SharingRatio71%/rps10-benchmark/without_bitsandbytes"
#     ########################################################
#     "${base_dir}/gangmuk-prefix/SharingRatio47%/rps2-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio47%/rps4-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio47%/rps6-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio47%/rps8-benchmark/without_bitsandbytes"
#     # "${base_dir}/gangmuk-prefix/SharingRatio47%/rps10-benchmark/without_bitsandbytes"
#     ########################################################
#     "${base_dir}/gangmuk-prefix/SharingRatio28%/rps2-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio28%/rps4-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio28%/rps6-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio28%/rps8-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio28%/rps10-benchmark/without_bitsandbytes"
#     # ########################################################
#     "${base_dir}/gangmuk-prefix/SharingRatio9%/rps2-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio9%/rps4-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio9%/rps6-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio9%/rps8-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/SharingRatio9%/rps10-benchmark/without_bitsandbytes"
#     # ########################################################
#     "${base_dir}/gangmuk-prefix/MixedSharingRatio10_30_50_70%/rps2-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/MixedSharingRatio10_30_50_70%/rps4-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/MixedSharingRatio10_30_50_70%/rps6-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/MixedSharingRatio10_30_50_70%/rps8-benchmark/without_bitsandbytes"
#     "${base_dir}/gangmuk-prefix/MixedSharingRatio10_30_50_70%/rps10-benchmark/without_bitsandbytes"
#     # ########################################################
#     "${base_dir}/mooncake/conversation-2/rps4-benchmark/without_bitsandbytes"
#     "${base_dir}/mooncake/conversation-2/rps8-benchmark/without_bitsandbytes"
#     "${base_dir}/mooncake/conversation-2/rps12-benchmark/without_bitsandbytes"
#     "${base_dir}/mooncake/conversation-2/rps16-benchmark/without_bitsandbytes"
#     # "${base_dir}/mooncake/conversation-2/rps20-benchmark/without_bitsandbytes"
#     # ########################################################
#     "${base_dir}/mooncake/toolagent-2/rps4-benchmark/without_bitsandbytes"
#     "${base_dir}/mooncake/toolagent-2/rps8-benchmark/without_bitsandbytes"
#     "${base_dir}/mooncake/toolagent-2/rps12-benchmark/without_bitsandbytes"
#     "${base_dir}/mooncake/toolagent-2/rps16-benchmark/without_bitsandbytes"
#     # "${base_dir}/mooncake/toolagent-2/rps20-benchmark/without_bitsandbytes"
#     # ########################################################
# )

# Verify all directories exist
for target_dir in "${target_dir_list[@]}"; do
    if [ ! -d "${target_dir}" ]; then
        echo "Target directory ${target_dir} does not exist."
        exit 1
    fi
done

echo "=== Step 1: Processing individual workloads ==="
echo ""

# Write target directories to a temp file for the merge script
target_dirs_file="${base_dir}/target_dirs.txt"
printf "%s\n" "${target_dir_list[@]}" > "${target_dirs_file}"
echo "Wrote ${#target_dir_list[@]} target directories to ${target_dirs_file}"

# Process each workload (in parallel), tracking PIDs to catch failures
pids=()
pid_to_dir=()
for target_dir in "${target_dir_list[@]}"; do
    echo "Processing ${target_dir}"
    # python compare_routing_strategies_with_client_log.py "${target_dir}" --iteration-from 2 --iteration-upto 4 &
    python compare_routing_strategies.py "${target_dir}" --iteration-from 2 &
    pids+=($!)
    pid_to_dir+=("${target_dir}")
    echo "Processed ${target_dir} with iteration from 2"
done

# Wait for all background jobs and check for failures
failed_dirs=()
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "ERROR: Failed to process ${pid_to_dir[$i]}"
        failed_dirs+=("${pid_to_dir[$i]}")
    fi
done

if [ "${#failed_dirs[@]}" -gt 0 ]; then
    echo ""
    echo "Aborting. ${#failed_dirs[@]} workload(s) failed:"
    for d in "${failed_dirs[@]}"; do
        echo "  - ${d}"
    done
    exit 1
fi

echo ""
echo "=== Step 2: Merging and plotting all workloads ==="
echo ""

# Merge all metrics and create comparison plots (using only specified target directories)
# python merge_and_plot_all_workloads_from_client_log.py "${base_dir}" --target-dirs-file "${target_dirs_file}"

python trendline_plot_from_client_log.py "${base_dir}" --target-dirs-file "${target_dirs_file}" --exclude e2e_latency_negative_linear