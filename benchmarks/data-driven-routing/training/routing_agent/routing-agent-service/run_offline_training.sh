#!/bin/bash


training_data_dir="training_data/p4096_s1024_rps10_spp_20_ndp_100/prefix"
final_model_dir="${training_data_dir}/final_model"
rm -r ${final_model_dir}
echo "Removed existing final model directory: ${final_model_dir}"

# Define the command without backgrounding it initially
python_cmd="python3 offline_routing_agent.py ${training_data_dir}/data.csv --model simpler_contextual_bandit --analyze_behavior --ttft_slo 1000 --avg_tpot_slo 50"

# Execute the command, redirecting output
${python_cmd} &> output.log
# ${python_cmd}

# Check the exit status of the previous command
if [ $? -eq 0 ]; then
    echo "Python script executed successfully."
else
    echo "Error: Python script failed."
    exit 1 # Exit the bash script if the python script failed
fi

# Echo the command that was run to a file
echo "${python_cmd} &> output.log" > "${final_model_dir}/python_command.txt"

# Copy the generated log file after the python script has finished
cp output.log "${final_model_dir}/output.log"

echo "Script complete. Logs copied to ${final_model_dir}"