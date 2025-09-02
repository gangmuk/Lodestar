#!/bin/bash

set -e

echo "LLM Routing Agent - Streamlined Data Pipeline"
echo "=============================================="

# data_file="../training_data/p4096_s1024_rps20/rl+random/data_replaced.csv"
data_file="../training_data/SharingRatio71%-p2048_s512_rps5_spp_10_ndp50-p4096_s1024_rps8_spp_10_ndp50-p8096_s2048_rps3_spp_10_ndp50-half/data.csv"

use_sampled_data=true

# Configuration: Processing and training parameters
ttft_slo=1000
avg_tpot_slo=50
model_type="simpler_contextual_bandit"
analyze_behavior=true
REWARD_FUNCTION="linear_simple"

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

echo ""
echo "📊 STEP 1: Processing data to structured CSV"
echo "==========================================="
process_cmd="python3 data_processor.py --input_file ${data_file} --ttft_slo ${ttft_slo} --avg_tpot_slo ${avg_tpot_slo} --REWARD_FUNCTION ${REWARD_FUNCTION}"
echo "Command: ${process_cmd}"
${process_cmd}

if [ ! -f "${processed_csv}" ]; then
    echo "❌ Failed to create processed CSV: ${processed_csv}"
    exit 1
fi


python3 dataset_analyzer.py --processed_csv ${processed_csv} --reward-function ${REWARD_FUNCTION} --save-sampled-dataset

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
final_model_dir="${base_final_model_dir}-${training_data_filename}"

echo ""
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

python_cmd="python3 offline_routing_agent.py ${processed_csv} --model ${model_type} ${analyze_flag} --ttft_slo ${ttft_slo} --avg_tpot_slo ${avg_tpot_slo} --final_model_dir ${final_model_dir}"

echo "Command: ${python_cmd}"
echo ""

# Run training and capture output
${python_cmd} 2>&1 | tee "${final_model_dir}/output.txt"

# Save command for reference
echo "${python_cmd}" > "${final_model_dir}/python_command.txt"

echo "Model saved to: ${final_model_dir}"
echo "Training log: ${final_model_dir}/output.txt"