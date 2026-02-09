#!/bin/bash

# Script to run the client in a K8s pod using kubectl exec

set -e

# Configuration
CLIENT_SERVICE_POD_NAME=client-service
CLIENT_SERVICE_CONTAINER_NAME=client
k8s_cluster="vke"
# k8s_cluster="aws"
# target_gpu="NVIDIA-L40S"
# target_gpu="NVIDIA-A10"
# Define experiment configurations
# Format: "routing_policy|workload_name|target_gpu|rps|total_num_episodes"

target_gpu="NVIDIA-A30"
workload_mode="profiling" # benchmark, profiling
experiment_configs=(
    # # ## SharingRatio71%, total number of requests: 1500
    # # ## benchmark mode: A30, rps >= 9
    # # # "latency_predictor|gangmuk-prefix|SharingRatio71%|${target_gpu}|10|3"
    # # "contextual_bandit_perpodmodel_checkpoint_negative_linear-before_latency_optimization|gangmuk-prefix|SharingRatio71%|${target_gpu}|10|3"
    # "contextual_bandit_perpodmodel_checkpoint_negative_linear|gangmuk-prefix|SharingRatio71%|${target_gpu}|10|3"
    "prefix_cache_1|gangmuk-prefix|SharingRatio71%|${target_gpu}|8|2"
    "least_request|gangmuk-prefix|SharingRatio71%|${target_gpu}|8|2"
    "random|gangmuk-prefix|SharingRatio71%|${target_gpu}|8|2"


    # # # ## SharingRatio47%, total number of requests: 2000
    # # # ## benchmark mode: A30, rps >= 7
    # # # # "latency_predictor|gangmuk-prefix|SharingRatio47%|${target_gpu}|6|3"
    # "contextual_bandit_perpodmodel_checkpoint_negative_linear|gangmuk-prefix|SharingRatio47%|${target_gpu}|9|3"
    "prefix_cache_1|gangmuk-prefix|SharingRatio47%|${target_gpu}|7|2"
    "least_request|gangmuk-prefix|SharingRatio47%|${target_gpu}|7|2"
    "random|gangmuk-prefix|SharingRatio47%|${target_gpu}|7|2"


    # # # ## SharingRatio28%, total number of requests: 2000
    # # # ## benchmark mode: A30, rps >= 5
    # # # # "latency_predictor|gangmuk-prefix|SharingRatio28%|${target_gpu}|8|3"
    # "contextual_bandit_perpodmodel_checkpoint_negative_linear|gangmuk-prefix|SharingRatio28%|${target_gpu}|8|3"
    "prefix_cache_1|gangmuk-prefix|SharingRatio28%|${target_gpu}|7|2"
    "least_request|gangmuk-prefix|SharingRatio28%|${target_gpu}|7|2"
    "random|gangmuk-prefix|SharingRatio28%|${target_gpu}|7|2"


    # # # ## SharingRatio9%, total number of requests: 2000
    # # # ## benchmark mode: A30, rps >= 6
    # # # # "latency_predictor|gangmuk-prefix|SharingRatio9%|${target_gpu}|6|3"
    # "contextual_bandit_perpodmodel_checkpoint_negative_linear|gangmuk-prefix|SharingRatio9%|${target_gpu}|6|3"
    "prefix_cache_1|gangmuk-prefix|SharingRatio9%|${target_gpu}|7|2"
    "least_request|gangmuk-prefix|SharingRatio9%|${target_gpu}|7|2"
    "random|gangmuk-prefix|SharingRatio9%|${target_gpu}|7|2"

    
    # # # # ## MixedSharingRatio10_30_50_70, total number of requests: 4000
    # # # ## benchmark mode: A30, rps >= 8
    # # # # "latency_predictor|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|9|1"
    # "contextual_bandit_perpodmodel_checkpoint_negative_linear|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|9|2"
    "prefix_cache_1|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|7|1"
    "least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|7|1"
    "random|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|7|1"

    # # # "latency_predictor|mooncake|conversation-2|${target_gpu}|20|3"
    # "contextual_bandit_perpodmodel_checkpoint_negative_linear|mooncake|conversation-2|${target_gpu}|20|3"
    "prefix_cache_1|mooncake|conversation-2|${target_gpu}|5|2"
    "random|mooncake|conversation-2|${target_gpu}|5|2"
    "least_request|mooncake|conversation-2|${target_gpu}|5|2"

    # # ## Mooncake toolagent: 2713
    # # ## benchmark mode: A30, rps >= 10
    # # # "latency_predictor|mooncake|toolagent-2|${target_gpu}|20|3"
    # "contextual_bandit_perpodmodel_checkpoint_negative_linear|mooncake|toolagent-2|${target_gpu}|20|3"
    "prefix_cache_1|mooncake|toolagent-2|${target_gpu}|10|2"
    "random|mooncake|toolagent-2|${target_gpu}|10|2"
    "least_request|mooncake|toolagent-2|${target_gpu}|10|2"

    # # ## Azure code: 2608
    # # ## benchmark mode: A30, rps >= 16
    # # # "latency_predictor|azure|azure_code_poisson|${target_gpu}|25|3"
    # "contextual_bandit_perpodmodel_checkpoint_negative_linear|azure|azure_code_poisson|${target_gpu}|25|3"
    "prefix_cache_1|azure|azure_code_poisson|${target_gpu}|25|2"
    "least_request|azure|azure_code_poisson|${target_gpu}|25|2"
    "random|azure|azure_code_poisson|${target_gpu}|25|2"

    # azure code: long input, short output, medium prefix sharing ratio
    # "contextual_bandit_perpodmodel_checkpoint_negative_linear|azure|azure_code-access_sequential-sharingmean_0.3-sharingstd_0.2-numreqpergroup_10|${target_gpu}|20|1"
    "prefix_cache_1|azure|azure_code-access_sequential-sharingmean_0.3-sharingstd_0.2-numreqpergroup_10|${target_gpu}|15|1"
    "least_request|azure|azure_code-access_sequential-sharingmean_0.3-sharingstd_0.2-numreqpergroup_10|${target_gpu}|15|1"
    "random|azure|azure_code-access_sequential-sharingmean_0.3-sharingstd_0.2-numreqpergroup_10|${target_gpu}|15|1"

    # azure single-turn chat: short input, long output, small prefix sharing ratio
    # "contextual_bandit_perpodmodel_checkpoint_negative_linear|azure|azure_conv-access_sequential-sharingmean_0.1-sharingstd_0.2-numreqpergroup_2|${target_gpu}|20|1"
    "prefix_cache_1|azure|azure_conv-access_sequential-sharingmean_0.1-sharingstd_0.2-numreqpergroup_2|${target_gpu}|15|1"
    "least_request|azure|azure_conv-access_sequential-sharingmean_0.1-sharingstd_0.2-numreqpergroup_2|${target_gpu}|15|1"
    "random|azure|azure_conv-access_sequential-sharingmean_0.1-sharingstd_0.2-numreqpergroup_2|${target_gpu}|15|1"

    # azure multi-turn chat: short input, long output, large prefix sharing ratio
    # "contextual_bandit_perpodmodel_checkpoint_negative_linear|azure|azure_conv-access_sequential-sharingmean_0.5-sharingstd_0.2-numreqpergroup_10|${target_gpu}|20|1"
    "prefix_cache_1|azure|azure_conv-access_sequential-sharingmean_0.5-sharingstd_0.2-numreqpergroup_10|${target_gpu}|15|1"
    "least_request|azure|azure_conv-access_sequential-sharingmean_0.5-sharingstd_0.2-numreqpergroup_10|${target_gpu}|15|1"
    "random|azure|azure_conv-access_sequential-sharingmean_0.5-sharingstd_0.2-numreqpergroup_10|${target_gpu}|15|1"
    
)

ship_model=0
ship_code=0
ship_data=0

force_exact_output_tokens=1
override_workload_output_length=0
max_tokens=1
max_tokens_std=0
max_input_tokens=10000
input_tokens_std=100

if [ "${override_workload_output_length}" == "1" ]; then
    output_wrk_name="maxTokens_${max_tokens}-maxTokensStd_${max_tokens_std}"
else
    output_wrk_name="use_given_output_length"
fi

ENABLE_ONLINE_LEARNING=1
INCLUDE_GPU_FEATURES=0
LOAD_PRETRAINED_MODEL=1
MAX_TOTAL_DATA=1000000
# MAX_TOTAL_DATA=50000
MIN_NUM_TRAINING_DATA=10000
MIN_NUM_UPDATE_DATA=2000
EXPLORATION_ENABLED=0
EXPLORATION_RATE=0.1
prompt_type="chat" # chat, token-ids
ENABLE_FLUSH=1
FLUSH_PERIOD=10
MIN_NUM_LOG_MESSAGES_TO_FLUSH=100

# Configuration for the client
api_key="sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi"
POD_LABEL_SELECTOR="model.aibrix.ai/name=llama-3-8b-instruct"
llm_model="llama-3-8b-instruct"

ipaddr=$(kubectl get svc -n envoy-gateway-system envoy-aibrix-system-aibrix-eg-903790dc -o jsonpath='{.spec.clusterIP}')
port=80
echo "ipaddr of aibrix-system-aibrix-eg-903790dc svc: ${ipaddr}, port: ${port}"
if [ -z "${ipaddr}" ]; then
    echo "Error: ipaddr is empty"
    exit 1
fi

num_experiments=${#experiment_configs[@]}
echo "========================================="
echo "Will run ${num_experiments} experiment(s):"
for i in $(seq 0 $((num_experiments-1))); do
    IFS='|' read -r routing workload gpu rps episodes <<< "${experiment_configs[$i]}"
    echo "  $((i+1)). routing=${routing}, workload=${workload}, gpu=${gpu}, rps=${rps}, episodes=${episodes}"
done
echo "========================================="

for experiment_idx in $(seq 0 $((num_experiments-1))); do
    experiment_start_time=$(date +%s)

    IFS='|' read -r routing_policy workload_category workload_name target_gpu rps total_num_episodes <<< "${experiment_configs[$experiment_idx]}"

    echo ""
    echo "========================================="
    echo "Starting Experiment $((experiment_idx+1))/${num_experiments}"
    echo "========================================="
    echo "  Routing Policy:    ${routing_policy}"
    echo "  Workload Category: ${workload_category}"
    echo "  Workload:          ${workload_name}"
    echo "  Target GPU:        ${target_gpu}"
    echo "  RPS:               ${rps}"
    echo "  Total Episodes:    ${total_num_episodes}"
    echo "========================================="
    delimiter="+"
    if [ "${routing_policy}" == "preble" ]; then
        config="preble${delimiter}${routing_policy}"
    else
        config="rl-online-router${delimiter}${routing_policy}"
    fi
    routing="${config%%${delimiter}*}"
    subAlgorithm="${config#*${delimiter}}"
    if [[ "${routing_policy}" == *"contextual_bandit"* ]]; then
        source_final_model_dir="../workload-and-experiment_results/${target_gpu}/${output_wrk_name}/${workload_category}/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear"
        echo "✓ Using contextual bandit model: ${source_final_model_dir}"
    elif [[ "${routing_policy}" == *"latency_predictor"* ]]; then
        source_final_model_dir="../workload-and-experiment_results/${target_gpu}/${output_wrk_name}/${workload_category}/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear"
        echo "✓ Using contextual bandit model: ${source_final_model_dir}"
    else
        source_final_model_dir="../workload-and-experiment_results/${target_gpu}/maxTokens_1-maxTokensStd_0/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear"
    fi

    if [ "${ship_model}" == "1" ]; then
        if [ ! -d "${source_final_model_dir}" ]; then
        echo "Error: Final model directory does not exist: ${source_final_model_dir}"
            echo "Exiting... 4"
            exit 1
        fi

        if [ ! -f "${source_final_model_dir}/model_config.json" ]; then
            echo "Error: model_config.json does not exist: ${source_final_model_dir}/model_config.json"
            echo "Exiting... 5"
            exit 1
        fi

        # Check for model weights based on routing policy
        if [[ "${routing_policy}" == *"contextual_bandit"* ]]; then
            if [ ! -f "${source_final_model_dir}/reward_net.pth" ]; then
                echo "Error: reward_net.pth does not exist: ${source_final_model_dir}/reward_net.pth"
                echo "Exiting... 6"
                exit 1
            fi
        elif [[ "${routing_policy}" == *"latency_predictor"* ]]; then
            if [ ! -f "${source_final_model_dir}/latency_predictor.pth" ]; then
                echo "Error: latency_predictor.pth does not exist: ${source_final_model_dir}/latency_predictor.pth"
                echo "Exiting... 6"
                exit 1
            fi
        fi

        if [ ! -f "${source_final_model_dir}/feature_normalization_statistics.csv" ]; then
            echo "Error: feature_normalization_statistics.csv does not exist: ${source_final_model_dir}/feature_normalization_statistics.csv"
            echo "Exiting... 7"
            exit 1
        fi
        if [ ! -f "${source_final_model_dir}/feature_distribution_statistics.csv" ]; then
            echo "Error: feature_distribution_statistics.csv does not exist: ${source_final_model_dir}/feature_distribution_statistics.csv"
            echo "Exiting... 8"
            exit 1
        fi
    fi


    if [ "${prompt_type}" == "chat" ]; then
        # Always use the base workload.jsonl as input to async-client.py.
        # In profiling mode, async-client.py will internally generate a profiling
        # schedule (and optionally dump workload_profiling.jsonl) at runtime.
        workload_file_name="workload.jsonl"
        workload_path_in_pod="/app/workload/${workload_category}/${workload_name}/${workload_file_name}"
        workload_path_in_host="../workload-and-experiment_results/${workload_category}/${workload_name}/${workload_file_name}"
    elif [ "${prompt_type}" == "token-ids" ]; then
        # Same idea for token-ids mode: async-client.py handles profiling generation.
        workload_file_name="workload_token.jsonl"
        workload_path_in_pod="/app/workload/${workload_category}/${workload_name}/${workload_file_name}"
        workload_path_in_host="../workload-and-experiment_results/${workload_category}/${workload_name}/${workload_file_name}"
    else
        echo "Error: Unknown prompt type: ${prompt_type}"
        echo "Exiting... 8"
        exit 1
    fi
    # if [ ! -f "${workload_path_in_host}" ]; then
    #     echo "Error: Workload file does not exist: ${workload_path_in_host}"
    #     echo "Exiting... 9"
    #     exit 1
    # fi
    output_dir="/app/output/${workload_category}/${workload_name}/${subAlgorithm}-$(date +%Y%m%d_%H%M%S)"
    output_jsonl_path="${output_dir}/output.jsonl"

    # Create local experiment result output directory
    timestamp=$(date +%Y%m%d_%H%M%S)
    experiment_result_output_dir="../workload-and-experiment_results/${target_gpu}/${output_wrk_name}/${workload_category}/${workload_name}/rps${rps}-${workload_mode}/${subAlgorithm}-iter${total_num_episodes}"
    if [[ "${routing_policy}" == *"contextual_bandit"* ]] || [[ "${routing_policy}" == *"latency_predictor"* ]]; then
        experiment_result_output_dir="${experiment_result_output_dir}-onlinelearning_${ENABLE_ONLINE_LEARNING}"
    fi
    experiment_result_output_dir="${experiment_result_output_dir}-${timestamp}"
    echo "* experiment_result_output_dir: ${experiment_result_output_dir}"
    if [ ! -d "${experiment_result_output_dir}" ]; then
        mkdir -p "${experiment_result_output_dir}"
    fi

    echo "========================================="
    echo "* workload_category: ${workload_category}"
    echo "* workload_name: ${workload_name}"
    echo "* target_gpu: ${target_gpu}"
    echo "* rps: ${rps}"
    echo "* total_num_episodes: ${total_num_episodes}"
    echo "* experiment_result_output_dir: ${experiment_result_output_dir}"
    echo "========================================="

    # if [ "${routing_policy}" == "scalable_rl_agent" ]; then
    #     # Note: For scalable_rl_agent, total_num_episodes is recalculated based on training parameters
    #     # This overrides the value from total_num_episodes_list
    #     training_epochs=10
    #     num_requests_per_episode=500
    #     num_iterations=20
    #     num_episodes_per_iteration=5
    #     n_eval_episodes=5
    #     total_num_episode_for_training=$((num_iterations * num_episodes_per_iteration))
    #     total_num_episode_for_evaluation=$((num_iterations * n_eval_episodes))
    #     total_num_episodes=$((total_num_episode_for_training + total_num_episode_for_evaluation))
    #     batch_size=256
    #     echo "*******************************************"
    #     echo "num_iterations: ${num_iterations}"
    #     echo "num_episodes_per_iteration: ${num_episodes_per_iteration}"
    #     echo "n_eval_episodes: ${n_eval_episodes}"
    #     echo "total_num_episode_for_training: ${total_num_episode_for_training}"
    #     echo "total_num_episode_for_evaluation: ${total_num_episode_for_evaluation}"
    #     echo "total_num_episodes (recalculated): ${total_num_episodes}"
    #     echo "*******************************************"
    #     python update_model_config.py \
    #         --source_final_model_dir "../training_data/scalable_rl_agent/final_model" \
    #         --num_requests_per_episode ${num_requests_per_episode} \
    #         --num_iterations ${num_iterations} \
    #         --num_episodes_per_iteration ${num_episodes_per_iteration} \
    #         --n_eval_episodes ${n_eval_episodes} \
    #         --training_epochs ${training_epochs} \
    #         --batch_size ${batch_size}
    # fi

    echo "Starting to update k8s env for routing-agent-service"
    python3 update_k8s_env.py \
        --deployment routing-agent-service \
        --namespace default \
        --container routing-agent \
        --env EXPLORATION_ENABLED=${EXPLORATION_ENABLED} \
        --env MIN_NUM_TRAINING_DATA=${MIN_NUM_TRAINING_DATA} \
        --env MIN_NUM_UPDATE_DATA=${MIN_NUM_UPDATE_DATA} \
        --env ENABLE_ONLINE_LEARNING=${ENABLE_ONLINE_LEARNING} \
        --env POD_LABEL_SELECTOR=${POD_LABEL_SELECTOR} \
        --env EXPLORATION_RATE=${EXPLORATION_RATE} \
        --env TARGET_GPU_MODEL=${target_gpu} \
        --env MAX_TOTAL_DATA=${MAX_TOTAL_DATA} \
        --env INCLUDE_GPU_FEATURES=${INCLUDE_GPU_FEATURES} \
        --env LOAD_PRETRAINED_MODEL=${LOAD_PRETRAINED_MODEL} \
        --env ROUTING_STRATEGY=${routing_policy} \
        --env OUTPUT_WRK_NAME=${output_wrk_name} \
        --env WORKLOAD_CATEGORY=${workload_category} \
        --env WORKLOAD_NAME=${workload_name}
    echo "Finished updating k8s env for routing-agent-service"
    
    # echo "Starting to update k8s env for aibrix-gateway-plugins"
    # python3 update_k8s_env.py \
    #     --deployment aibrix-gateway-plugins \
    #     --namespace aibrix-system \
    #     --container gateway-plugin \
    #     --env ENABLE_FLUSH=${ENABLE_FLUSH} \
    #     --env FLUSH_PERIOD=${FLUSH_PERIOD} \
    #     --env MIN_NUM_LOG_MESSAGES_TO_FLUSH=${MIN_NUM_LOG_MESSAGES_TO_FLUSH}
    # sleep 2

    kubectl rollout restart deployment client-service --namespace default
    kubectl rollout restart deployment aibrix-gateway-plugins --namespace aibrix-system
    kubectl rollout restart deployment routing-agent-service --namespace default

    sleep 2
    python3 check_ready.py --deployment llama-3-8b-instruct --namespace default
    python3 check_ready.py --deployment aibrix-gateway-plugins --namespace aibrix-system
    python3 check_ready.py --deployment routing-agent-service --namespace default
    python3 check_ready.py --deployment client-service --namespace default
    sleep 2

    ###############
    ## code ship ##
    ###############

    ship_start_time=$(date +%s)
    if [ "${ship_code}" == "1" ] || [ "${ship_model}" == "1" ]; then
        python ship_all.py --ship_code ${ship_code} --ship_model ${ship_model} --source_final_model_dir ${source_final_model_dir} --ship_data ${ship_data} --k8s_cluster ${k8s_cluster}
    fi
    
    if [ "${routing_policy}" == "scalable_rl_agent" ]; then
        scalable_rl_agent_init_model_dir="../training_data/scalable_rl_agent/init_model"
        python kubectl_cp_from_host_to_pod.py ${scalable_rl_agent_init_model_dir} /app/final_model routing-agent-service default
    fi

    python kubectl_cp_from_host_to_pod.py async-client.py /app client-service default

    ship_end_time=$(date +%s)
    ship_took=$((ship_end_time - ship_start_time))
    echo "* ship_all took: ${ship_took}s"

    echo "========================================="
    echo "Running Client in K8s Pod"
    echo "========================================="
    echo "Pod:                 ${CLIENT_SERVICE_POD_NAME}"
    echo "Container:           ${CLIENT_SERVICE_CONTAINER_NAME}"
    echo "Routing Strategy:    ${routing}"
    echo "Sub-Algorithm:       ${subAlgorithm}"
    echo "Workload Category:   ${workload_category}"
    echo "Workload Name:       ${workload_name}"
    echo "Online Learning:     ${ENABLE_ONLINE_LEARNING}"
    echo "total_num_episodes:  ${total_num_episodes}"
    echo "Max Tokens:          ${max_tokens}"
    echo "Max Tokens Std:      ${max_tokens_std}"
    echo "Override Workload Output Length: ${override_workload_output_length}"
    echo "RPS:                 ${rps}"
    echo "========================================="

    # Find the actual pod name (in case of deployment with generated suffix)
    ACTUAL_POD=$(kubectl get pods -l app=${CLIENT_SERVICE_POD_NAME} -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [ -z "$ACTUAL_POD" ]; then
        echo "Error: No pod found with label app=${CLIENT_SERVICE_POD_NAME}"
        echo "Trying to use pod name directly: ${CLIENT_SERVICE_POD_NAME}"
        ACTUAL_POD=${CLIENT_SERVICE_POD_NAME}
    fi

    echo "Using pod: ${ACTUAL_POD}"

    # Wait for pod to be ready
    echo "Waiting for pod to be ready..."
    kubectl wait --for=condition=ready pod/${ACTUAL_POD} --timeout=60s || {
        echo "Error: Pod did not become ready within 60 seconds"
        kubectl describe pod ${ACTUAL_POD}
        echo "Exiting... 8"
        exit 1
    }

    # Create output directory in pod
    echo "Creating output directory in pod..."
    kubectl exec ${ACTUAL_POD} -c ${CLIENT_SERVICE_CONTAINER_NAME} -- mkdir -p ${output_dir}

    # Check if workload file exists
    echo "Checking if workload file exists..."
    kubectl exec ${ACTUAL_POD} -c ${CLIENT_SERVICE_CONTAINER_NAME} -- test -f ${workload_path_in_pod} || {
        echo "Error: Workload file ${workload_path_in_pod} not found in pod"
        echo "Available workloads:"
        kubectl exec ${ACTUAL_POD} -c ${CLIENT_SERVICE_CONTAINER_NAME} -- find /app/workload -name "*.jsonl"
        echo "Exiting... 9"
        exit 1
    }

    echo "Starting log collection..."
    
    # periodic dump
    collect_logs_continuously() {
        local namespace=$1
        local pod_pattern=$2
        local output_file=$3
        local interval=30  # Collect logs every 5 seconds
        
        local since_time=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        
        while true; do
            pod_name=$(kubectl get pods -n ${namespace} | grep ${pod_pattern} | awk '{print $1}' | head -n 1)
            if [ -n "$pod_name" ]; then
                # Get logs since last collection time
                kubectl logs --since-time="${since_time}" -n ${namespace} ${pod_name} >> ${output_file} 2>&1
                # Update timestamp for next iteration
                since_time=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
            fi
            sleep ${interval}
        done
    }
    
    ## follow mode
    # collect_logs_continuously() {
    #     local namespace=$1
    #     local pod_pattern=$2
    #     local output_file=$3
        
    #     while true; do
    #         pod_name=$(kubectl get pods -n ${namespace} | grep ${pod_pattern} | awk '{print $1}' | head -n 1)
            
    #         if [ -n "$pod_name" ]; then
    #             echo "[$(date)] Following logs from ${pod_name}" >> ${output_file}
    #             # Follow logs until pod dies/restarts, then loop restarts
    #             kubectl logs -f -n ${namespace} ${pod_name} >> ${output_file} 2>&1 || true
    #         fi
            
    #         sleep 5  # Brief pause before reattaching
    #     done
    # }
    
    # Function to copy checkpoints periodically
    copy_checkpoints_periodically() {
        local output_dir=$1
        local interval=300  # Copy checkpoints every 5 minutes (300 seconds)
        
        # Wait a bit before first copy to let training start
        sleep 60
        
        while true; do
            checkpoint_ts=$(date +%Y%m%d_%H%M%S)
            echo "[$(date)] Copying checkpoints at ${checkpoint_ts}..."
            python kubectl_cp_from_pod_to_host.py --src /app/final_model/checkpoints --dst "${output_dir}/checkpoints_${checkpoint_ts}" --deployment routing-agent-service --namespace default 2>&1 | grep -v "tar: Removing leading"
            echo "[$(date)] Checkpoint copy completed: checkpoints_${checkpoint_ts}"
            sleep ${interval}
        done
    }
    
    # Start log collection in background
    collect_logs_continuously "aibrix-system" "aibrix-gateway-plugins" "${experiment_result_output_dir}/all-aibrix-gateway-plugins.log.txt" &
    pid_1=$!
    collect_logs_continuously "default" "routing-agent-service" "${experiment_result_output_dir}/all-routing-agent-service.log.txt" &
    pid_2=$!
    
    # # Start periodic checkpoint copying for scalable_rl_agent
    # if [ "${subAlgorithm}" == "scalable_rl_agent" ]; then
    #     echo "Starting periodic checkpoint copying (every 5 minutes)..."
    #     copy_checkpoints_periodically "${experiment_result_output_dir}" &
    #     pid_checkpoint=$!
    # fi

    echo "Starting client in pod..."
    echo "Output will be saved to: ${output_dir}"

    # ./curl-aws.sh ${llm_model} ${subAlgorithm}
    # sleep 0.5
    # ./curl-aws.sh ${llm_model} ${subAlgorithm}
    # sleep 0.5
    # ./curl-aws.sh ${llm_model} ${subAlgorithm}

    # Run the client using kubectl exec
        # python3 /app/async-client.py \
    # Run client inside the pod. async-client.py will interpret --workload_mode:
    # - benchmark: use original timing or RPS-based scheduling
    # - profiling: build a profiling schedule from --rps, dump it, and send requests accordingly
    if [ "${workload_category}" == "mooncake" ]; then
        input_token_length_scaling=2.0
    elif [ "${workload_category}" == "azure" ]; then
        input_token_length_scaling=3.0
    else
        input_token_length_scaling=1.0
    fi

    kubectl exec ${ACTUAL_POD} -c ${CLIENT_SERVICE_CONTAINER_NAME} -- \
        python3 /app/async-client.py \
            --workload_path ${workload_path_in_pod} \
            --model ${llm_model} \
            --endpoint http://${ipaddr}:${port} \
            --api_key ${api_key} \
            --output_file_path ${output_jsonl_path} \
            --routing_strategy ${routing} \
            --subAlgorithm ${subAlgorithm} \
            --max_tokens ${max_tokens} \
            --max_tokens_std ${max_tokens_std} \
            --override_workload_output_length ${override_workload_output_length} \
            --force_exact_output_tokens ${force_exact_output_tokens} \
            --output_dir ${output_dir} \
            --prompt_type ${prompt_type} \
            --rps ${rps} \
            --poisson_arrivals \
            --shuffle_requests \
            --iterations ${total_num_episodes} \
            --streaming \
            --max_input_tokens ${max_input_tokens} \
            --input_tokens_std ${input_tokens_std} \
            --input_token_length_scaling ${input_token_length_scaling} \
            --output_token_length_scaling 1.0 \
            --iteration_overlap_ratio 0.0 \
            --iteration_ramp_duration 10.0 \
            --iteration_ramp_start_fraction 0.1 \
            --workload_mode ${workload_mode} \
            2>&1 | tee ${experiment_result_output_dir}/client.log.txt

    sleep 5
    # kubectl rollout restart deployment llama-3-8b-instruct

    # Process logs
    cat ${experiment_result_output_dir}/all-aibrix-gateway-plugins.log.txt | grep "**@latency_metrics" | grep -v "infer:" > ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins.log.csv
    echo "* processed gateway log: ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins.log.csv-processed.csv"
    echo "* all gateway log: ${experiment_result_output_dir}/all-aibrix-gateway-plugins.log.txt"
    echo "* filtered gateway log: ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins.log.csv"
    echo "* routing agent log: ${experiment_result_output_dir}/all-routing-agent-service.log.txt"
    echo "* client log: ${experiment_result_output_dir}/client.log.txt"
    if [ "${subAlgorithm}" == "scalable_rl_agent" ]; then
        echo "* checkpoints (periodic snapshots): ${experiment_result_output_dir}/checkpoints_*/"
        echo "* final checkpoint: ${experiment_result_output_dir}/checkpoints_${checkpoint_ts}/"
    fi

    python plot_latency_timeseries.py ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins.log.csv
    python3 plot_latency_analysis_with_client_log.py ${experiment_result_output_dir}/client.log.txt

    
    # Copy final model
    # if [ "${routing_policy}" == "contextual_bandit" ] || [ "${routing_policy}" == "latency_predictor" ]; then
    # if routing_policy has contextual_bandit or latency_predictor in its name, then copy the final model
    if [[ "${routing_policy}" == *"contextual_bandit"* ]] || [[ "${routing_policy}" == *"latency_predictor"* ]]; then
        kubectl_cp_start_time=$(date +%s)
        echo "Copying final_model from pod..."
        model_dir_in_pod="/app/${target_gpu}/${output_wrk_name}/${workload_category}/final_model-${routing_policy}"
        echo "* experiment_result_output_dir: ${experiment_result_output_dir}"
        echo "* model_dir_in_pod: ${model_dir_in_pod}"
        python kubectl_cp_from_pod_to_host.py --src ${model_dir_in_pod} --dst "${experiment_result_output_dir}/final_model/${target_gpu}" --deployment routing-agent-service --namespace default --skip-files "tensor_dataset.pt" "metadata.pkl"
        
        kubectl_cp_end_time=$(date +%s)
        echo "* copying final_model took: $((kubectl_cp_end_time - kubectl_cp_start_time))s"
        hyperparameters_file_path=$(find "${experiment_result_output_dir}/final_model/${target_gpu}" -name "model_config.json" | head -1)
        if [ -z "$hyperparameters_file_path" ]; then
            echo "ERROR: Could not find model_config.json in ${experiment_result_output_dir}/final_model/${target_gpu}"
            # exit 1
        fi
    else
        echo "Skipping final model copying for ${routing_policy}. It does not use any learned model."
    fi
    if [ -z "$hyperparameters_file_path" ]; then
        echo "* Using hyperparameters file: $hyperparameters_file_path"
        python ../agent_codes/data_processor.py --input_file ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins.log.csv --output_file ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins-processed.log.csv --hyperparameters_file_path "$hyperparameters_file_path"
    else
        python ../agent_codes/data_processor.py --input_file ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins.log.csv --output_file ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins-processed.log.csv
    fi


    # python kubectl_cp_from_pod_to_host.py /tmp/latency_metrics.log "${experiment_result_output_dir}/latency_metrics.log.txt" gateway-plugins aibrix-system

    # # Copy checkpoints (for scalable_rl_agent)
    # if [ "${subAlgorithm}" == "scalable_rl_agent" ]; then
    #     checkpoint_ts=$(date +%Y%m%d_%H%M%S)
    #     echo "Copying scalable RL agent checkpoints..."
    #     python kubectl_cp_from_pod_to_host.py /app/final_model/checkpoints "${experiment_result_output_dir}/checkpoints_${checkpoint_ts}" routing-agent-service default || echo "⚠️  No checkpoints found (agent may not have trained enough steps)"
    # fi

    
    # Kill background processes
    kill $pid_1 2>/dev/null || true
    kill $pid_2 2>/dev/null || true
    if [ "${subAlgorithm}" == "scalable_rl_agent" ] && [ -n "${pid_checkpoint}" ]; then
        kill $pid_checkpoint 2>/dev/null || true
    fi

    
    experiment_end_time=$(date +%s)
    echo ""
    echo "========================================="
    echo "Experiment $((experiment_idx+1))/${num_experiments} completed!"
    echo "* experiment took: $((experiment_end_time - experiment_start_time))s"
    echo "* workload:            ${workload_name}"
    echo "* workload category:   ${workload_category}"
    echo "* routing strategy:    ${routing}"
    echo "* sub-algorithm:       ${subAlgorithm}"
    echo "* max tokens:          ${max_tokens}"
    echo "* max tokens std:      ${max_tokens_std}"
    echo "* override workload output length: ${override_workload_output_length}"
    echo "* rps:                 ${rps}"
    echo "========================================="

    # kubectl rollout restart deployment client-service
    # kubectl rollout restart deployment routing-agent-service
    # kubectl rollout restart deployment aibrix-gateway-plugins -n aibrix-system
done
