#!/bin/bash

# filename: run_offline_training.sh

set -e


# workload_dataset="SharingRatio9%-p200_s1800_rps8_spp_20_ndp80-p400_s3600_rps8_spp_20_ndp80-p800_s7200_rps3_spp_20_ndp80-half"
# workload_dataset="SharingRatio28%-p600_s1400_rps8_spp_20_ndp80-p1200_s2800_rps8_spp_20_ndp80-p2400_s5600_rps3_spp_20_ndp80-half"
# workload_dataset="SharingRatio47%-p1024_s1024_rps8_spp_20_ndp80-p2048_s2048_rps8_spp_20_ndp80-p4096_s4096_rps3_spp_20_ndp80-half"
# workload_dataset="SharingRatio71%-p2048_s512_rps5_spp_10_ndp50-p4096_s1024_rps8_spp_10_ndp50-p8096_s2048_rps3_spp_10_ndp50-half"

# workload_dataset="SharingRatio9%"
# workload_dataset="SharingRatio28%"
# workload_dataset="SharingRatio47%"
# workload_dataset="SharingRatio71%"
workload_dataset="p4096_s1024_rps20"

routing_policy_for_data_file="all" # "all", "prefix", "rl", "random", "rl+random"

# data_file="../training_data/${workload_dataset}/${routing_policy_for_data_file}/data.csv"
data_file="../training_data/${workload_dataset}/${routing_policy_for_data_file}/data_replaced.csv"

use_sampled_data=false # true, false
analyze_behavior=true # true, false
ttft_slo=1000
avg_tpot_slo=50
ttft_reward_weight=2.0 # ttft_reward_weight*ttft_rewards + max(0, (1-ttft_reward_weight))*tpot_rewards
REWARD_FUNCTION="linear_simple" # "linear_simple", "linear_simple_extended", "piecewise_linear_steeper_gradient", "latency_optimized"
offline_learning_rate=0.001

# Step 1: Process raw data to structured CSV
if [ ! -f "${data_file}" ]; then
    echo "❌ Data file not found: ${data_file}"
    exit 1
fi

echo "✓ Found data file: ${data_file}"

# Generate processed CSV filename automatically
data_dir=$(dirname "${data_file}")
data_basename=$(basename -- "${data_file}")
data_name="${data_basename%.*}"  # Remove .csv extension
processed_csv="${data_dir}/${data_name}-processed.csv"

echo "📊 STEP 1: Processing data to structured CSV"
echo "==========================================="
process_cmd="python3 data_processor.py --input_file ${data_file} --output_file ${processed_csv} --ttft_slo ${ttft_slo} --avg_tpot_slo ${avg_tpot_slo} --REWARD_FUNCTION ${REWARD_FUNCTION} --ttft_reward_weight ${ttft_reward_weight}"
echo "Command: ${process_cmd}"
${process_cmd}

if [ ! -f "${processed_csv}" ]; then
    echo "❌ Failed to create processed CSV: ${processed_csv}"
    exit 1
fi


python3 dataset_analyzer.py --processed_csv ${processed_csv} --reward-function ${REWARD_FUNCTION} --ttft-slo ${ttft_slo} --avg-tpot-slo ${avg_tpot_slo} --ttft-reward-weight ${ttft_reward_weight} --save-sampled-dataset

sampled_processed_csv="${processed_csv%.*}-sampled.csv"
echo "Sampled processed CSV: ${sampled_processed_csv}"

if [ ${use_sampled_data} = "true" ]; then
    processed_csv="${sampled_processed_csv}"
fi

# Step 2: Setup model directory
training_data_dir=$(dirname "${processed_csv}")
training_data_filename=$(basename -- "${processed_csv}")
training_data_filename="${training_data_filename%.*}"

base_final_model_dir="${training_data_dir}/final_model"
final_model_dir="${base_final_model_dir}-${training_data_filename}-${REWARD_FUNCTION}-lr_${offline_learning_rate}-ttft_weight_${ttft_reward_weight}-ttftslo_${ttft_slo}-avgtpotslo_${avg_tpot_slo}"

echo "📁 SETTING UP MODEL DIRECTORY"
echo "=============================="
echo "Training data directory: ${training_data_dir}"
echo "Final model directory: ${final_model_dir}"

if [ -d "${final_model_dir}" ]; then
    echo "⚠ Removing existing model directory: ${final_model_dir}"
    rm -rf "${final_model_dir}"
fi
mkdir -p "${final_model_dir}"
echo "✅ Created model directory: ${final_model_dir}"

# Step 3: Run training with the new streamlined pipeline
# Build command arguments
analyze_flag=""
if [ "${analyze_behavior}" = "true" ]; then
    analyze_flag="--analyze_behavior"
fi

python_cmd="python3 offline_routing_agent.py ${processed_csv} ${analyze_flag} --ttft_slo ${ttft_slo} --avg_tpot_slo ${avg_tpot_slo} --final_model_dir ${final_model_dir} --ttft_reward_weight ${ttft_reward_weight} --offline_learning_rate ${offline_learning_rate} --reward_function ${REWARD_FUNCTION}"


${python_cmd} 2>&1 | tee "${final_model_dir}/output.txt"
echo "${python_cmd}" > "${final_model_dir}/python_command.txt"

echo final_model_dir: ${final_model_dir}

echo "final_model_dir: ${final_model_dir}" > ${final_model_dir}/full_path.txt
echo "data_file: ${data_file}" >> ${final_model_dir}/full_path.txt
echo "processed_csv: ${processed_csv}" >> ${final_model_dir}/full_path.txt

python csv_training_analyzer.py ${final_model_dir}/training_metrics.csv

echo "Model saved to: ${final_model_dir}"