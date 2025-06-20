#!/bin/bash

# target_dir=./workload/prefix-sharing-workload/merged-comprehensive-workload/set4-iter1
# target_dir=./workload/prefix-sharing-workload/p4096_s1024_rps20/iter2
target_dir=./workload/prefix-sharing-workload/p4096_s1024_rps10_spp_20_ndp_100

if [ ! -d "${target_dir}" ]; then
    echo "Target directory ${target_dir} does not exist."
    exit 1
fi

# Find all subdirectories in the target directory
for subdir_path in "${target_dir}"/*/; do
    # Check if the glob matched anything (in case there are no subdirectories)
    if [ ! -d "${subdir_path}" ]; then
        continue
    fi
    
    # Extract just the subdirectory name
    subdir=$(basename "${subdir_path}")
    
    log_file="${target_dir}/${subdir}/filtered-aibrix-gateway-plugins.log.csv"
    
    if [ ! -f "${log_file}" ]; then
        echo "Log file ${log_file} does not exist, skipping ${subdir}."
        continue
    fi
    
    echo "Processing ${subdir}..."
    python plot_latency_timeseries.py "${log_file}" 1
done