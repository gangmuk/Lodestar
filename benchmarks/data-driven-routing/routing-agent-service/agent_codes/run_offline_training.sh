#!/bin/bash

set -e

# data_file="../training_data/p4096_s1024_rps20/random/data.csv"
# data_file="../training_data/p4096_s1024_rps20/rl/data.csv"
# data_file="../training_data/p4096_s1024_rps20/data.csv"
# data_file="../training_data/p4096_s1024_rps20/rl+random/data_replaced.csv"
data_file="../training_data/mix/sharing71%-random_2-data.csv"

if [ ! -f "${data_file}" ]; then
    echo "Data file does not exist: ${data_file}"
    exit 1
fi

training_data_dir=$(dirname "${data_file}")
if [ ! -d "${training_data_dir}" ]; then
    echo "Training data directory does not exist: ${training_data_dir}"
    exit 1
fi

final_model_dir="${training_data_dir}/final_model"
if [ -d "${final_model_dir}" ]; then
    echo "Removing existing final model directory: ${final_model_dir}"
    rm -r ${final_model_dir}
fi

mkdir -p "${final_model_dir}"
echo "Created final model directory: ${final_model_dir}"

python_cmd="python3 offline_routing_agent.py ${data_file} --model simpler_contextual_bandit --analyze_behavior --ttft_slo 1000 --avg_tpot_slo 50"
${python_cmd} 2>&1 | tee "${final_model_dir}/output.txt"

echo "${python_cmd}" > "${final_model_dir}/python_command.txt"
echo "* Script complete. Logs copied to ${final_model_dir}"
echo "* model training analysis ${final_model_dir}/comprehenisve_training_metrics.pdf"