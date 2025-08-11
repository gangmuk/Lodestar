#!/bin/bash

set -e

training_data_dir="training_data/p4096_s1024_rps20/rl"

final_model_dir="${training_data_dir}/final_model"
if [ -d "${final_model_dir}" ]; then
    echo "Removing existing final model directory: ${final_model_dir}"
    rm -r ${final_model_dir}
fi

mkdir -p "${final_model_dir}"
echo "Created final model directory: ${final_model_dir}"

python_cmd="python3 offline_routing_agent.py ${training_data_dir}/data.csv --model simpler_contextual_bandit --analyze_behavior --ttft_slo 1000 --avg_tpot_slo 50"
# ${python_cmd} &> offline_training_output.txt
${python_cmd}

cp offline_training_output.txt "${final_model_dir}/output.txt"
echo "Copied output to ${final_model_dir}/output.txt"

echo "${python_cmd}" > "${final_model_dir}/python_command.txt"
echo "* Script complete. Logs copied to ${final_model_dir}"
echo "* Output log copied to ${final_model_dir}/output.txt"