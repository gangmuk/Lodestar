#!/bin/bash

set -e

# training_data_dir="../training_data/p4096_s1024_rps10_spp_20_ndp_100/prefix"
# training_data_dir="../training_data/p4096_s1024_rps10_spp_20_ndp_100/random"
# training_data_dir="../training_data/p4096_s1024_rps20/prefix"
training_data_dir="../training_data/p4096_s1024_rps20/rl+random"
# training_data_dir="../training_data/p4096_s1024_rps20/random"
# training_data_dir="../training_data/p4096_s1024_rps20/rl"
# training_data_dir="../training_data/p4096_s1024_rps20"

# data_file="${training_data_dir}/data.csv"
data_file="${training_data_dir}/data_replaced.csv"

echo "training data file: ${data_file}"
sleep 3

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