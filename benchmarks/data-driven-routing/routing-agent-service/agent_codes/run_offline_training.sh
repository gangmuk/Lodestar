#!/bin/bash

set -e

lr_scheduler_type="constant" # "exponential", "constant", "gradient_adaptive"
hidden_dim=128 # 64, 128, 256, 128 was very very slightly better than 64
batch_size=256
training_epochs=50
sampling_ratio=0.3

# excluded_pod_features="waiting_requests,cpu_kv_cache,running_requests" # still working
# excluded_pod_features="kv_hit_ratio,gpu_kv_cache,running_requests,waiting_requests,inflight_requests" # kinda bad
# excluded_pod_features="gpu_kv_cache,running_requests,waiting_requests,inflight_requests" # this is also kinda bad
# excluded_pod_features="kv_hit_ratio,gpu_kv_cache,running_requests,inflight_requests" #
# excluded_pod_features="gpu_kv_cache,running_requests,inflight_requests" #
excluded_pod_features="none" #
# kv_hit_ratio is not really helpful.... at least in 28% workload
# excluded_request_features="input_tokens,total_tokens" # very bad
excluded_request_features="output_tokens"
# excluded_pod_features="kv_hit_ratio,gpu_kv_cache,inflight_requests" # slightly better 2
# excluded_pod_features="gpu_kv_cache,inflight_requests" # same

include_gpu_features=0

## For excluded_pod_features, you need to use the same name in preprocess.py
# 'kv_hit_ratio': f"{pod_id}-kv_hit_ratio"]
# 'inflight_requests': f"{pod_id}-inflight_requests"
# 'gpu_kv_cache': f"{pod_id}-gpu_kv_cache"]
# 'cpu_kv_cache': f"{pod_id}-cpu_kv_cache"]
# 'running_requests': f"{pod_id}-running_requests"]
# 'waiting_requests': f"{pod_id}-waiting_requests"]
# 'prefill_tokens': f"{pod_id}-prefill_tokens"]
# 'decode_tokens': f"{pod_id}-decode_tokens"]
# 'GPU': f"{pod_id}-GPU"] 

no_normalize_features="none" # "kv_hit_ratio", "none"

# model_type="latency_predictor"
model_type="contextual_bandit"

latency_metric="ttft" # "ttft", "avg_tpot", "e2e_latency" (for latency_predictor)
analyze_behavior=false # true, false
analyze_dataset=false # true, false
reward_decay_factor=0.91
ttft_slo=1000
avg_tpot_slo=50
ttft_reward_weight=2.0 # ttft_reward_weight*ttft_rewards + max(0, (1-ttft_reward_weight))*tpot_rewards
REWARD_FUNCTION="latency_optimized" # "inverse_latency", "linear_simple", "linear_simple_extended", "piecewise_linear_steeper_gradient", "latency_optimized"
offline_learning_rate=0.001
time_stamp=$(date +%Y%m%d_%H%M%S)
lr_scheduler_gamma=0.95

# data_file=../workload-and-experiment_results/NVIDIA-A10/data.csv
data_file=$1
data_dir=$(dirname "${data_file}")
if [ ! -f "${data_file}" ]; then
    echo "❌ Data file not found: ${data_file}"
    exit 1
fi
echo "✓ Found data file: ${data_file}"

# Generate processed CSV filename automatically
data_basename=$(basename -- "${data_file}")
data_name="${data_basename%.*}"  # Remove .csv extension
processed_csv="${data_dir}/${data_name}-processed.csv"

## final_mode_dir naming
final_model_dir="${data_dir}/final_model"
# if [ "${model_type}" == "contextual_bandit" ]; then
#     final_model_dir="${final_model_dir}-${data_name}-processed-${REWARD_FUNCTION}-lr_${offline_learning_rate}-ttft_weight_${ttft_reward_weight}-ttftslo_${ttft_slo}-avgtpotslo_${avg_tpot_slo}"
#     if [ "${excluded_pod_features}" != "" ]; then
#         final_model_dir="${final_model_dir}-without_${excluded_pod_features}"
#     fi
#     final_model_dir="${final_model_dir}-hidden_dim_${hidden_dim}"

#     if [ "${lr_scheduler_type}" == "gradient_adaptive" ]; then
#         final_model_dir="${final_model_dir}-lrs_grad_adapt"
#     elif [ "${lr_scheduler_type}" == "exponential" ]; then
#         final_model_dir="${final_model_dir}-lrs_exp"
#     elif [ "${lr_scheduler_type}" == "plateau" ]; then
#         final_model_dir="${final_model_dir}-lrs_plateau"
#     else
#         echo "❌ Unknown LR scheduler type: ${lr_scheduler_type}"
#         exit 1
#     fi
# fi
final_model_dir="${final_model_dir}-${model_type}"
if [ "${model_type}" == "latency_predictor" ]; then
    final_model_dir="${final_model_dir}_${latency_metric}"
fi
final_model_dir=${final_model_dir}-${time_stamp}
if [ -d "${final_model_dir}" ]; then
    rm -rf "${final_model_dir}"
fi
echo "Final model directory: ${final_model_dir}"
mkdir -p "${final_model_dir}"
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
--offline_learning_rate ${offline_learning_rate} \
--excluded_pod_features "${excluded_pod_features}" \
--excluded_request_features "${excluded_request_features}" \
--no_normalize_features "${no_normalize_features}" \
--lr_scheduler_type ${lr_scheduler_type} \
--training_epochs ${training_epochs} \
--batch_size ${batch_size} \
--lr_scheduler_gamma ${lr_scheduler_gamma} \
--reward_decay_factor ${reward_decay_factor} \
--model_type ${model_type} \
--latency_metric ${latency_metric} \
--include_gpu_features ${include_gpu_features}

echo "📄 STEP 1: start data_processor"
python3 data_processor.py --input_file ${data_file} --output_file ${processed_csv} --sampling_ratio ${sampling_ratio} --hyperparameters_file_path ${hyper_json} --ttft_threshold 10000 2>&1 | tee ${data_processor_log}
echo "finished data_processor"

if [ ! -f "${processed_csv}" ]; then
    echo "❌ Failed to create processed CSV: ${processed_csv}"
    exit 1
fi

if [ "${analyze_dataset}" = "true" ]; then
    python3 dataset_analyzer.py --processed_csv ${processed_csv} --reward-function ${REWARD_FUNCTION} --ttft-slo ${ttft_slo} --avg-tpot-slo ${avg_tpot_slo} --ttft-reward-weight ${ttft_reward_weight} --save-sampled-dataset 2>&1 | tee ${dataset_analyzer_log}
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
# Build command arguments
analyze_flag=""
if [ "${analyze_behavior}" = "true" ]; then
    analyze_flag="--analyze_behavior"
fi

python_cmd="python3 offline_routing_agent.py ${processed_csv} ${analyze_flag} --final_model_dir ${final_model_dir} --hyperparameter_file_path ${hyper_json}"
echo "${python_cmd}" > "${final_model_dir}/python_command.txt"
${python_cmd} 2>&1 | tee ${offline_routing_agent_log}

cat ${data_processor_log} >> ${offline_routing_agent_log}
if [ "${analyze_dataset}" = "true" ]; then
    cat ${dataset_analyzer_log} >> ${offline_routing_agent_log}
fi
cat ${offline_routing_agent_log} >> ${final_model_dir}/output.txt

echo "final_model_dir: ${final_model_dir}" > ${final_model_dir}/full_path.txt
echo "data_file: ${data_file}" >> ${final_model_dir}/full_path.txt
echo "processed_csv: ${processed_csv}" >> ${final_model_dir}/full_path.txt
# if [ "${model_type}" == "contextual_bandit" ]; then
#     python csv_training_analyzer.py ${final_model_dir}/training_metrics.csv
# fi
echo "Model saved to: ${final_model_dir}"