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

target_gpu="NVIDIA-A10"

experiment_configs=(
    # # ## SharingRatio71%, total number of requests: 1500
    # "contextual_bandit_quantile_based_perpodmodel_policygradient|SharingRatio71%|${target_gpu}|7|4"
    # "contextual_bandit_quantile_based_perpodmodel_advanced|SharingRatio71%|${target_gpu}|7|4"
    # "contextual_bandit_quantile_based_perpodmodel|SharingRatio71%|${target_gpu}|7|4"
    # "contextual_bandit_negative_linear_perpodmodel|SharingRatio71%|${target_gpu}|7|4"
    # "prefix_cache_1|SharingRatio71%|${target_gpu}|7|4"
    # "least_latency|SharingRatio71%|${target_gpu}|7|4"
    # "least_request|SharingRatio71%|${target_gpu}|7|4"
    # "least_kv_cache|SharingRatio71%|${target_gpu}|7|2"
    # "random|SharingRatio71%|${target_gpu}|7|2"

    # # # # # ## SharingRatio47%, total number of requests: 2000
    # "contextual_bandit_quantile_based_perpodmodel_policygradient|SharingRatio47%|${target_gpu}|6|4"
    # "contextual_bandit_quantile_based_perpodmodel_advanced|SharingRatio47%|${target_gpu}|6|4"
    # "contextual_bandit_quantile_based_perpodmodel|SharingRatio47%|${target_gpu}|6|4"
    # "latency_predictor|SharingRatio47%|${target_gpu}|6|5"
    # "prefix_cache_1|SharingRatio47%|${target_gpu}|6|4"
    # "least_latency|SharingRatio47%|${target_gpu}|6|1"
    # "least_request|SharingRatio47%|${target_gpu}|6|4"
    # "least_kv_cache|SharingRatio47%|${target_gpu}|6|1"
    # "random|SharingRatio47%|${target_gpu}|6|1"


    # # # # # ## SharingRatio28%, total number of requests: 2000
    # "contextual_bandit_quantile_based_perpodmodel_policygradient|SharingRatio28%|${target_gpu}|5|4"
    # "contextual_bandit_quantile_based_perpodmodel_advanced|SharingRatio28%|${target_gpu}|5|4"
    # "contextual_bandit_quantile_based_perpodmodel|SharingRatio28%|${target_gpu}|5|4"
    # # "latency_predictor|SharingRatio28%|${target_gpu}|5|5"
    # # "prefix_cache_1|SharingRatio28%|${target_gpu}|5|1"
    # # "least_latency|SharingRatio28%|${target_gpu}|5|1"
    # "least_request|SharingRatio28%|${target_gpu}|5|4"
    # # "least_kv_cache|SharingRatio28%|${target_gpu}|5|1"
    # # "random|SharingRatio28%|${target_gpu}|5|1"


    # # # # # ## SharingRatio9%, total number of requests: 2000
    # "contextual_bandit_quantile_based_perpodmodel|SharingRatio9%|${target_gpu}|5|4"
    # "contextual_bandit_quantile_based_perpodmodel_advanced|SharingRatio9%|${target_gpu}|5|4"
    # # "latency_predictor|SharingRatio9%|${target_gpu}|5|5"
    # # "prefix_cache_1|SharingRatio9%|${target_gpu}|5|1"
    # # "least_latency|SharingRatio9%|${target_gpu}|5|1"
    # "least_request|SharingRatio9%|${target_gpu}|5|4"
    # # "least_kv_cache|SharingRatio9%|${target_gpu}|5|1"
    # # "random|SharingRatio9%|${target_gpu}|5|1"

    # # # ## MixedSharingRatio10_30_50_70, total number of requests: 4000
    # "contextual_bandit_quantile_based_perpodmodel|MixedSharingRatio10_30_50_70%|${target_gpu}|6|4"
    # "contextual_bandit_quantile_based_perpodmodel_advanced|MixedSharingRatio10_30_50_70%|${target_gpu}|6|4"
    # # "latency_predictor|MixedSharingRatio10_30_50_70%|${target_gpu}|6|3"
    # # "prefix_cache_1|MixedSharingRatio10_30_50_70%|${target_gpu}|6|1"
    # # "least_latency|MixedSharingRatio10_30_50_70%|${target_gpu}|6|1"
    # "least_request|MixedSharingRatio10_30_50_70%|${target_gpu}|6|4"
    # # "least_kv_cache|MixedSharingRatio10_30_50_70%|${target_gpu}|6|1"
    # # "random|MixedSharingRatio10_30_50_70%|${target_gpu}|6|1"

)

INCLUDE_GPU_FEATURES=0
LOAD_PRETRAINED_MODEL=1

ENABLE_ONLINE_LEARNING=1
MAX_TOTAL_DATA=40000
# MIN_NUM_TRAINING_DATA=2000
# MIN_NUM_UPDATE_DATA=1000

MIN_NUM_TRAINING_DATA=1000
MIN_NUM_UPDATE_DATA=1000

EXPLORATION_ENABLED=0
EXPLORATION_RATE=0.1

max_input_tokens=8000
override_workload_output_length=1
max_tokens=1
max_tokens_std=0
force_exact_output_tokens=1

ship_model=0
ship_code=1
ship_offline_training_data=0

ENABLE_FLUSH=1
FLUSH_PERIOD=10
MIN_NUM_LOG_MESSAGES_TO_FLUSH=100

architecture="PrefillOnly" # PrefillOnly, Aggregated


# Configuration for the client
api_key="sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi"
# POD_LABEL_SELECTOR="llama2-7b"
# POD_LABEL_SELECTOR="tinyllama"
POD_LABEL_SELECTOR="model.aibrix.ai/name=llama-3-8b-instruct"
llm_model="llama-3-8b-instruct"
# POD_LABEL_SELECTOR="model.aibrix.ai/name=llama3-1-8b"
# llm_model="llama3-1-8b"


# async-client.py will run inside the client-service pod. So we don't need to port forward. ClusterIP of envoy-aibrix-system-aibrix-eg-903790dc svc will work!
# but for curl (e.g., ./curl-aws.sh), we need to port forward. It is done inside ./curl-aws.sh.
ipaddr=$(kubectl get svc -n envoy-gateway-system envoy-aibrix-system-aibrix-eg-903790dc -o jsonpath='{.spec.clusterIP}')
port=80
echo "ipaddr of aibrix-system-aibrix-eg-903790dc svc: ${ipaddr}, port: ${port}"
if [ -z "${ipaddr}" ]; then
    echo "Error: ipaddr is empty"
    exit 1
fi

# Display experiment configurations
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
    IFS='|' read -r routing_policy workload_name target_gpu rps total_num_episodes <<< "${experiment_configs[$experiment_idx]}"
    echo ""
    echo "========================================="
    echo "Starting Experiment $((experiment_idx+1))/${num_experiments}"
    echo "========================================="
    echo "  Routing Policy:    ${routing_policy}"
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
    if [ "${routing_policy}" == "scalable_rl_agent" ]; then
        final_model_dir="../training_data/scalable_rl_agent/final_model"
    elif [ "${routing_policy}" == "contextual_bandit" ]; then
        # Use the contextual bandit model from training
        # When shipping: model is in Docker at /app/final_model/{GPU}/contextual_bandit/
        # When developing: point to host path for ship_all.py to copy
        if [ "${target_gpu}" == "NVIDIA-A10" ]; then
            # Point to the inverse_latency trained model (will be shipped to pod)
            final_model_dir="../workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0/final_model-contextual_bandit-20251208_164438"
            echo "✓ Using contextual bandit model: ${final_model_dir}"
        elif [ "${target_gpu}" == "GPU-L3c" ] || [ "${target_gpu}" == "NVIDIA-L40" ] || [ "${target_gpu}" == "NVIDIA-L40S" ]; then
            echo "Error: No contextual bandit model trained for ${target_gpu} yet"
            echo "Exiting... 3"
            exit 1
        else
            echo "Error: Unknown target GPU model for contextual_bandit: ${target_gpu}"
            echo "Exiting... 3"
            exit 1
        fi
    else
        if [ "${target_gpu}" == "GPU-L3c" ] || [ "${target_gpu}" == "NVIDIA-L40" ] || [ "${target_gpu}" == "NVIDIA-L40S" ]; then
            # final_model_dir="../training_data/L20-7/merged-data/all-with-mixed/final_model-latency_predictor_ttft"

            final_model_dir="../workload-and-experiment_results/GPU-L3c/SharingRatio71%/rps8/final_model-latency_predictor_ttft-20251119_192610"

        elif [ "${target_gpu}" == "NVIDIA-A30" ] || [ "${target_gpu}" == "NVIDIA-L4" ] || [ "${target_gpu}" == "NVIDIA-A10" ]; then
            final_model_dir="../training_data/A30-8/final_model-latency_predictor_ttft-20251028_183743"
        elif [ "${target_gpu}" == "hetero" ]; then
            # final_model_dir="../training_data/hetero/used-in-paper/final_model-latency_predictor_ttft-20251029_034844"
            final_model_dir="../training_data/hetero/final_model-latency_predictor_ttft-20251101_213101-epoch60"
        else
            echo "Error: Unknown target GPU model: ${target_gpu}"
            echo "Exiting... 3"
            exit 1
        fi
    fi


    if [ "${ship_model}" == "1" ]; then
        if [ ! -d "${final_model_dir}" ]; then
        echo "Error: Final model directory does not exist: ${final_model_dir}"
            echo "Exiting... 4"
            exit 1
        fi

        if [ ! -f "${final_model_dir}/model_config.json" ]; then
            echo "Error: model_config.json does not exist: ${final_model_dir}/model_config.json"
            echo "Exiting... 5"
            exit 1
        fi

        # Check for model weights based on routing policy
        if [ "${routing_policy}" == "contextual_bandit" ]; then
            if [ ! -f "${final_model_dir}/reward_net.pth" ]; then
                echo "Error: reward_net.pth does not exist: ${final_model_dir}/reward_net.pth"
                echo "Exiting... 6"
                exit 1
            fi
        elif [ "${routing_policy}" == "latency_predictor" ]; then
            if [ ! -f "${final_model_dir}/latency_predictor.pth" ]; then
                echo "Error: latency_predictor.pth does not exist: ${final_model_dir}/latency_predictor.pth"
                echo "Exiting... 6"
                exit 1
            fi
        fi

        if [ ! -f "${final_model_dir}/feature_normalization_statistics.csv" ]; then
            echo "Error: feature_normalization_statistics.csv does not exist: ${final_model_dir}/feature_normalization_statistics.csv"
            echo "Exiting... 7"
            exit 1
        fi
    fi


    workload_path_in_pod="/app/workload/${workload_name}/workload.jsonl"
    workload_path_in_host="../workload-and-experiment_results/${workload_name}/workload.jsonl"
    output_dir="/app/output/${workload_name}-${subAlgorithm}-$(date +%Y%m%d_%H%M%S)"
    output_jsonl_path="${output_dir}/output.jsonl"

    # Create local experiment result output directory
    timestamp=$(date +%Y%m%d_%H%M%S)
    experiment_result_output_dir="../workload-and-experiment_results/${target_gpu}/maxTokens_${max_tokens}-maxTokensStd_${max_tokens_std}/${workload_name}/rps${rps}/${subAlgorithm}"
    if [ "${subAlgorithm}" == "rl_naive" ]; then
        trained_model_data_name=$(echo "$final_model_dir" | awk -F'training_data/' '{print $2}' | cut -d'/' -f1)
        used_data_name=$(echo "$final_model_dir" | awk -F'training_data/' '{print $2}' | cut -d'/' -f2)
        hyperparameter_name=$(echo "$final_model_dir" | awk -F'processed-' '{print $2}')
        hyperparameter_name="${hyperparameter_name}-explr_${EXPLORATION_ENABLED}"
        postfix="onlinelearning_${ENABLE_ONLINE_LEARNING}-trained_on_${trained_model_data_name}_${used_data_name}-${hyperparameter_name}-total_num_episodes${total_num_episodes}"
        experiment_result_output_dir="${experiment_result_output_dir}-${postfix}"
    elif [ "${subAlgorithm}" == "latency_predictor" ]; then
        trained_model_data_name=$(echo "$final_model_dir" | awk -F'training_data/' '{print $2}' | cut -d'/' -f1)
        prediction_metric=$(echo "$final_model_dir" | awk -F'latency_predictor_' '{print $2}')
        used_data_name=$(echo "$final_model_dir" | awk -F'training_data/' '{print $2}' | cut -d'/' -f2)
        # postfix="trained_on_${trained_model_data_name}_${used_data_name}"
        # postfix="trained_on_${used_data_name}"
        # experiment_result_output_dir="${experiment_result_output_dir}_${prediction_metric}"
        experiment_result_output_dir="${experiment_result_output_dir}_ttft"
    fi
    experiment_result_output_dir="${experiment_result_output_dir}-iter${total_num_episodes}-${timestamp}"
    echo "* experiment_result_output_dir: ${experiment_result_output_dir}"
    if [ ! -d "${experiment_result_output_dir}" ]; then
        mkdir -p "${experiment_result_output_dir}"
    fi

    echo "========================================="
    echo "* workload_name: ${workload_name}"
    echo "* target_gpu: ${target_gpu}"
    echo "* rps: ${rps}"
    echo "* total_num_episodes: ${total_num_episodes}"
    echo "* final_model_dir: ${final_model_dir}"
    echo "* experiment_result_output_dir: ${experiment_result_output_dir}"
    echo "========================================="

    if [ "${routing_policy}" == "scalable_rl_agent" ]; then
        # Note: For scalable_rl_agent, total_num_episodes is recalculated based on training parameters
        # This overrides the value from total_num_episodes_list
        training_epochs=10
        num_requests_per_episode=500
        num_iterations=20
        num_episodes_per_iteration=5
        n_eval_episodes=5
        total_num_episode_for_training=$((num_iterations * num_episodes_per_iteration))
        total_num_episode_for_evaluation=$((num_iterations * n_eval_episodes))
        total_num_episodes=$((total_num_episode_for_training + total_num_episode_for_evaluation))
        batch_size=256
        echo "*******************************************"
        echo "num_iterations: ${num_iterations}"
        echo "num_episodes_per_iteration: ${num_episodes_per_iteration}"
        echo "n_eval_episodes: ${n_eval_episodes}"
        echo "total_num_episode_for_training: ${total_num_episode_for_training}"
        echo "total_num_episode_for_evaluation: ${total_num_episode_for_evaluation}"
        echo "total_num_episodes (recalculated): ${total_num_episodes}"
        echo "*******************************************"
        python update_model_config.py \
            --final_model_dir "../training_data/scalable_rl_agent/final_model" \
            --num_requests_per_episode ${num_requests_per_episode} \
            --num_iterations ${num_iterations} \
            --num_episodes_per_iteration ${num_episodes_per_iteration} \
            --n_eval_episodes ${n_eval_episodes} \
            --training_epochs ${training_epochs} \
            --batch_size ${batch_size}
    fi

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
        --env WORKLOAD=${workload_name} \
        --env ROUTING_STRATEGY=${routing_policy}
    
    echo "Starting to update k8s env for aibrix-gateway-plugins"
    python3 update_k8s_env.py \
        --deployment aibrix-gateway-plugins \
        --namespace aibrix-system \
        --container gateway-plugin \
        --env ENABLE_FLUSH=${ENABLE_FLUSH} \
        --env FLUSH_PERIOD=${FLUSH_PERIOD} \
        --env MIN_NUM_LOG_MESSAGES_TO_FLUSH=${MIN_NUM_LOG_MESSAGES_TO_FLUSH}
    sleep 2

    kubectl rollout restart deployment client-service --namespace default

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
    python ship_all.py --ship_code ${ship_code} --ship_model ${ship_model} --final_model_dir ${final_model_dir} --k8s_cluster ${k8s_cluster} --ship_offline_training_data ${ship_offline_training_data}
    
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
    echo "Workload:            ${workload_name}"
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
    
    # Function to collect logs with timestamp tracking to avoid duplicates
    collect_logs_continuously() {
        local namespace=$1
        local pod_pattern=$2
        local output_file=$3
        local interval=30  # Collect logs every 30 seconds
        
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
    
    # Function to copy checkpoints periodically
    copy_checkpoints_periodically() {
        local output_dir=$1
        local interval=300  # Copy checkpoints every 5 minutes (300 seconds)
        
        # Wait a bit before first copy to let training start
        sleep 60
        
        while true; do
            checkpoint_ts=$(date +%Y%m%d_%H%M%S)
            echo "[$(date)] Copying checkpoints at ${checkpoint_ts}..."
            python kubectl_cp_from_pod_to_host.py /app/final_model/checkpoints "${output_dir}/checkpoints_${checkpoint_ts}" routing-agent-service default 2>&1 | grep -v "tar: Removing leading"
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
            --prompt_type chat \
            --rps ${rps} \
            --poisson_arrivals \
            --shuffle_requests \
            --iterations ${total_num_episodes} \
            --streaming \
            --max_input_tokens ${max_input_tokens} \
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

    
    # Copy final model
    if [ "${routing_policy}" == "contextual_bandit" ] || [ "${routing_policy}" == "latency_predictor" ]; then
        kubectl_cp_start_time=$(date +%s)
        echo "Copying final_model from pod..."
        # python kubectl_cp_from_pod_to_host.py /app/final_model/${target_gpu} "${experiment_result_output_dir}/final_model/${target_gpu}" routing-agent-service default
        model_dir_in_pod="/app/${target_gpu}/${architecture}/final_model/${routing_policy}"
        echo "* experiment_result_output_dir: ${experiment_result_output_dir}"
        echo "* model_dir_in_pod: ${model_dir_in_pod}"
        python kubectl_cp_from_pod_to_host.py ${model_dir_in_pod} "${experiment_result_output_dir}/final_model/${target_gpu}" routing-agent-service default --skip-files "tensor_dataset.pt"
        kubectl_cp_end_time=$(date +%s)
        echo "* copying final_model took: $((kubectl_cp_end_time - kubectl_cp_start_time))s"
        hyperparameters_file_path=$(find "${experiment_result_output_dir}/final_model/${target_gpu}" -name "model_config.json" | head -1)
        if [ -z "$hyperparameters_file_path" ]; then
            echo "ERROR: Could not find model_config.json in ${experiment_result_output_dir}/final_model/${target_gpu}"
            exit 1
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
