#!/bin/bash

set -e

# data_file="../training_data/p4096_s1024_rps20/random/data.csv"
# data_file="../training_data/p4096_s1024_rps20/rl/data.csv"
# data_file="../training_data/p4096_s1024_rps20/data.csv"
# data_file="../training_data/mix/sharing71%-random_2-data.csv"

data_file="../training_data/p4096_s1024_rps20/rl+random/data_replaced.csv"

# data_file="../training_data/SharingRatio71%-p2048_s512_rps5_spp_10_ndp50-p4096_s1024_rps8_spp_10_ndp50-p8096_s2048_rps3_spp_10_ndp50-half/data.csv"


## this
# already_processed_csv="../training_data/SharingRatio71%-p2048_s512_rps5_spp_10_ndp50-p4096_s1024_rps8_spp_10_ndp50-p8096_s2048_rps3_spp_10_ndp50-half/normalized_data.csv"

## this
# already_processed_csv="../training_data/SharingRatio71%-p2048_s512_rps5_spp_10_ndp50-p4096_s1024_rps8_spp_10_ndp50-p8096_s2048_rps3_spp_10_ndp50-half/normalized_data-sampled.csv"

## this
already_processed_csv="../training_data/p4096_s1024_rps20/rl+random/normalized_data.csv"

# already_processed_csv="none"


if [ ! -f "${data_file}" ]; then
    echo "Data file does not exist: ${data_file}"
    exit 1
fi

# if already_processed_csv is not None, then training_data_dir is the directory of already_processed_csv
if [ ! -z "${already_processed_csv}" ] && [ "${already_processed_csv}" != "none" ]; then
    training_data_dir=$(dirname "${already_processed_csv}")
else
    training_data_dir=$(dirname "${data_file}")
fi
if [ ! -d "${training_data_dir}" ]; then
    echo "Training data directory does not exist: ${training_data_dir}"
    exit 1
fi

# Define the base final_model_dir
base_final_model_dir="${training_data_dir}/final_model"

# Determine the filename to use for the directory name
training_data_filename=""
if [ -n "${already_processed_csv}" ] && [ "${already_processed_csv}" != "none" ]; then
    # Extract filename from already_processed_csv path
    filename=$(basename -- "${already_processed_csv}")
    training_data_filename="${filename%.*}"
elif [ -n "${data_file}" ]; then
    # Extract filename from data_file path
    filename=$(basename -- "${data_file}")
    training_data_filename="${filename%.*}"
fi

# Modify final_model_dir to include the training data filename
if [ -n "${training_data_filename}" ]; then
    final_model_dir="${base_final_model_dir}-${training_data_filename}"
else
    # Fallback to the original name if no filename is found
    final_model_dir="${base_final_model_dir}"
fi

echo "Updated final_model_dir to: ${final_model_dir}"

if [ -d "${final_model_dir}" ]; then
    echo "Removing existing final model directory: ${final_model_dir}"
    rm -r ${final_model_dir}
fi
mkdir -p "${final_model_dir}"
echo "Created final model directory: ${final_model_dir}"

python_cmd="python3 offline_routing_agent.py ${data_file} --model simpler_contextual_bandit --analyze_behavior --ttft_slo 1000 --avg_tpot_slo 50 --already_processed_csv ${already_processed_csv} --final_model_dir ${final_model_dir}"
${python_cmd} 2>&1 | tee "${final_model_dir}/output.txt"

echo "${python_cmd}" > "${final_model_dir}/python_command.txt"