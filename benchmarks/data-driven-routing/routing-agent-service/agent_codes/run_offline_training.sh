#!/bin/bash

set -e

data_dir=$1
normalization_mode=${2:-zscore}  # "zscore" (default, data-dependent) or "fixed_range" (domain-knowledge bounds)
latency_metric=ttft # "ttft", "avg_tpot", "e2e_latency" (applies to all model types including contextual_bandit)
REWARD_FUNCTION=negative_linear  # "negative_linear", "quantile_advantage", "quantile_based"

sampling_ratio=0.5
# sampling_ratio=1.0
max_training_data_size=100000
ttft_threshold=15000
# training_epochs=0
training_epochs=50


model_type="contextual_bandit_perpodmodel_checkpoint" # "contextual_bandit_perpodmodel_advanced", "contextual_bandit_perpodmodel_policygradient", "latency_predictor"

datetime=$(date +%Y%m%d_%H%M%S)

# Build final_model_dir path
final_model_dir="${data_dir}/final_model"
final_model_dir="${final_model_dir}-${model_type}"
if [ "${model_type}" == "latency_predictor" ]; then
    final_model_dir="${final_model_dir}_${latency_metric}"
elif [[ "${model_type}" == *"contextual_bandit"* ]]; then
    final_model_dir="${final_model_dir}_${latency_metric}_${REWARD_FUNCTION}"
fi



final_model_dir=${final_model_dir}-${datetime}
# final_model_dir=${final_model_dir}


if [ -d "${final_model_dir}" ]; then
    rm -rf "${final_model_dir}"
fi
echo "Final model directory: ${final_model_dir}"
mkdir -p "${final_model_dir}"

# All data files go under final_model_dir
data_file="${final_model_dir}/data.csv"
processed_csv="${final_model_dir}/data-processed.csv"

./merge-filtered-gateway-log.sh "${data_dir}" "${data_file}" contextual_bandit final_model
if [ ! -f "${data_file}" ]; then
    echo "❌ Data file not found: ${data_file}"
    exit 1
fi
echo "✓ Found data file: ${data_file}"



analyze_dataset=0
analyze_behavior=0
buffer_size=100000

hidden_dim=128
batch_size=256

learning_rate=0.0003
# learning_rate=0.0001
lr_scheduler_gamma=0.95
lr_scheduler_type="exponential" # "exponential", "constant", "gradient_adaptive"

excluded_pod_features="cpu_kv_cache,inflight_requests" # "inflight_requests,cpu_kv_cache" 'decode_tokens', 'gpu_kv_cache', 'inflight_requests', 'kv_hit_ratio', 'prefill_tokens', 'running_requests', 'waiting_requests', 'inflight_prefill_requests', 'inflight_decode_requests'
excluded_request_features="output_tokens,total_tokens" # "none", "input_tokens", "output_tokens", "total_tokens"
include_gpu_features=0
no_normalize_features="none" # "kv_hit_ratio", "none"
ttft_slo=1000
avg_tpot_slo=50
ttft_reward_weight=1.0 # ttft_reward_weight*ttft_rewards + max(0, (1-ttft_reward_weight))*tpot_rewards (should be 0-1)

##########################################################

data_processor_log="${final_model_dir}/data_processor.log.txt"
dataset_analyzer_log="${final_model_dir}/dataset_analyzer.log.txt"
offline_routing_agent_log="${final_model_dir}/offline_routing_agent.log.txt"

hyper_json="${final_model_dir}/model_config.json"
echo "📄 STEP 0: Writing hyperparameters JSON"
python3 write_hyperparameters.py \
--output ${hyper_json} \
--ttft_slo ${ttft_slo} \
--avg_tpot_slo ${avg_tpot_slo} \
--hidden_dim ${hidden_dim} \
--ttft_reward_weight ${ttft_reward_weight} \
--reward_function ${REWARD_FUNCTION} \
--learning_rate ${learning_rate} \
--excluded_pod_features "${excluded_pod_features}" \
--excluded_request_features "${excluded_request_features}" \
--no_normalize_features "${no_normalize_features}" \
--lr_scheduler_type ${lr_scheduler_type} \
--training_epochs ${training_epochs} \
--batch_size ${batch_size} \
--lr_scheduler_gamma ${lr_scheduler_gamma} \
--model_type ${model_type} \
--latency_metric ${latency_metric} \
--include_gpu_features ${include_gpu_features} \
--buffer_size ${buffer_size} \
--ttft_threshold ${ttft_threshold}

echo "📄 STEP 1: start data_processor"
start_time=$(date +%s)
data_processor_cmd="python3 data_processor.py --input_file ${data_file} --output_file ${processed_csv} --sampling_ratio ${sampling_ratio} --hyperparameters_file_path ${hyper_json} --ttft_threshold ${ttft_threshold} --max_training_data_size ${max_training_data_size}"
# python3 data_processor.py --input_file ${data_file} --output_file ${processed_csv} --sampling_ratio ${sampling_ratio} --hyperparameters_file_path ${hyper_json} --ttft_threshold ${ttft_threshold} 2>&1 | tee ${data_processor_log}

echo "data_processor_cmd: ${data_processor_cmd}"
echo "${data_processor_cmd}" > "${final_model_dir}/data_processor_command.txt"
${data_processor_cmd} 2>&1 | tee ${data_processor_log}
end_time=$(date +%s)
echo "finished data_processor in $((end_time - start_time)) seconds"

if [ ! -f "${processed_csv}" ]; then
    echo "❌ Failed to create processed CSV: ${processed_csv}"
    exit 1
fi

if [ "${analyze_dataset}" = "1" ] || [ "${analyze_dataset}" = "true" ]; then
    echo "📄 STEP 2: start dataset_analyzer - this can take a while on large CSVs"
    python3 dataset_analyzer.py --processed_csv ${processed_csv} --reward-function ${REWARD_FUNCTION} --ttft-slo ${ttft_slo} --avg-tpot-slo ${avg_tpot_slo} --ttft-reward-weight ${ttft_reward_weight} 2>&1 | tee ${dataset_analyzer_log}
fi

training_data_dir=$(dirname "${processed_csv}")
training_data_filename=$(basename -- "${processed_csv}")
training_data_filename="${training_data_filename%.*}"

# Note: final_model_dir already computed above to host model_config.json

echo "📁 SETTING UP MODEL DIRECTORY"
echo "=============================="
echo "Training data directory: ${training_data_dir}"
echo "Final model directory: ${final_model_dir}"

echo "✅ Using model directory: ${final_model_dir}"

# Step 3: Run training with the new streamlined pipeline
python_cmd="python3 offline_routing_agent.py ${processed_csv} --analyze_behavior ${analyze_behavior} --final_model_dir ${final_model_dir} --hyperparameter_file_path ${hyper_json} --normalization_mode ${normalization_mode}"
echo "python_cmd: ${python_cmd}"
echo "${python_cmd}" > "${final_model_dir}/python_command.txt"
${python_cmd} 2>&1 | tee ${offline_routing_agent_log}

cat ${data_processor_log} >> ${offline_routing_agent_log}
if [ "${analyze_dataset}" = "1" ] || [ "${analyze_dataset}" = "true" ]; then
    cat ${dataset_analyzer_log} >> ${offline_routing_agent_log}
fi
cat ${offline_routing_agent_log} >> ${final_model_dir}/output.txt

echo "final_model_dir: ${final_model_dir}" > ${final_model_dir}/full_path.txt
echo "data_file: ${data_file}" >> ${final_model_dir}/full_path.txt
echo "processed_csv: ${processed_csv}" >> ${final_model_dir}/full_path.txt
# if [ "${model_type}" == "contextual_bandit" ]; then
#     python csv_training_analyzer.py ${final_model_dir}/training_metrics.csv
# fi