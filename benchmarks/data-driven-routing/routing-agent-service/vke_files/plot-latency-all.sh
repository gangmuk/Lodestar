#!/bin/bash


python plot_latency_timeseries.py ../workload-and-experiment_results/SharingRatio71%/latency_predictor_ttft-trained_on_merged-data_all-iter10-20251013_074019/filtered-aibrix-gateway-plugins.log.csv & 

python plot_latency_timeseries.py ../workload-and-experiment_results/SharingRatio47%/latency_predictor_ttft-trained_on_merged-data_all-iter10-20251013_085811/filtered-aibrix-gateway-plugins.log.csv & 

python plot_latency_timeseries.py ../workload-and-experiment_results/SharingRatio28%/latency_predictor_ttft-trained_on_merged-data_all-iter10-20251013_095155/latency_metrics.log.txt & 

python plot_latency_timeseries.py ../workload-and-experiment_results/SharingRatio9%/latency_predictor_ttft-trained_on_merged-data_all-iter10-20251013_103902/filtered-aibrix-gateway-plugins.log.csv &

python plot_latency_timeseries.py ../workload-and-experiment_results/MixedSharingRatio10_30_50_70%/latency_predictor_ttft-trained_on_merged-data_all-iter5-20251014_233202/filtered-aibrix-gateway-plugins.log.csv

# # target_dir=./workload/prefix-sharing-workload/merged-comprehensive-workload/set4-iter1
# # target_dir=./workload/prefix-sharing-workload/p4096_s1024_rps20/iter2
# target_dir=./workload/prefix-sharing-workload/p4096_s1024_rps10_spp_20_ndp_100

# if [ ! -d "${target_dir}" ]; then
#     echo "Target directory ${target_dir} does not exist."
#     exit 1
# fi

# # Find all subdirectories in the target directory
# for subdir_path in "${target_dir}"/*/; do
#     # Check if the glob matched anything (in case there are no subdirectories)
#     if [ ! -d "${subdir_path}" ]; then
#         continue
#     fi
    
#     # Extract just the subdirectory name
#     subdir=$(basename "${subdir_path}")
    
#     log_file="${target_dir}/${subdir}/filtered-aibrix-gateway-plugins.log.csv"
    
#     if [ ! -f "${log_file}" ]; then
#         echo "Log file ${log_file} does not exist, skipping ${subdir}."
#         continue
#     fi
    
#     echo "Processing ${subdir}..."
#     python plot_latency_timeseries.py "${log_file}" 1
# done