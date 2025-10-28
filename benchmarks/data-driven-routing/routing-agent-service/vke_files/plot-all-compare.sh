#!/bin/bash

set -e

# target_dir_list=(
#     "../workload-and-experiment_results/SharingRatio71%/hand-picked"
#     "../workload-and-experiment_results/SharingRatio47%/hand-picked"
#     "../workload-and-experiment_results/SharingRatio28%/hand-picked"
#     "../workload-and-experiment_results/SharingRatio9%/hand-picked"
# )


target_dir_list=(
    "../workload-and-experiment_results/for_paper/SharingRatio71%"
    "../workload-and-experiment_results/for_paper/SharingRatio47%"
    "../workload-and-experiment_results/for_paper/SharingRatio28%"
    "../workload-and-experiment_results/for_paper/SharingRatio9%"
    "../workload-and-experiment_results/for_paper/MixedSharingRatio"
)

for target_dir in "${target_dir_list[@]}"; do
    if [ ! -d "${target_dir}" ]; then
        echo "Target directory ${target_dir} does not exist."
        exit 1
    fi
done

for target_dir in "${target_dir_list[@]}"; do
    echo "Processing ${target_dir}..."
    python compare_routing_strategies.py "${target_dir}"
done