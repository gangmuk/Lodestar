#!/bin/bash

# Script to run the client in a K8s pod using kubectl exec

set -e

# Configuration
NAMESPACE=${NAMESPACE:-default}
POD_NAME=${POD_NAME:-client-service}
CONTAINER_NAME=${CONTAINER_NAME:-client}
k8s_cluster="vke"

routing_policy="scalable_rl_agent"
# routing_policy="latency_predictor"

# Parameters are now hardcoded above
# To change them, edit the values at the top of this script

# Build subAlgorithm from routing_policy
delimiter="+"
config="rl-online-router${delimiter}${routing_policy}"
routing="${config%%${delimiter}*}"
subAlgorithm="${config#*${delimiter}}"


iterations=5
EXPLORATION_ENABLED="0"
ENABLE_ONLINE_LEARNING="0"
MIN_NUM_TRAINING_DATA="100"
ENABLE_FLUSH="0"
FLUSH_PERIOD="10"
MIN_NUM_LOG_MESSAGES_TO_FLUSH="100"
max_tokens=1000

# Configuration for the client
api_key="sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi"
# POD_LABEL_SELECTOR="llama2-7b"
# POD_LABEL_SELECTOR="tinyllama"
POD_LABEL_SELECTOR="model.aibrix.ai/name=llama-3-8b-instruct"
llm_model="llama-3-8b-instruct"
# POD_LABEL_SELECTOR="model.aibrix.ai/name=llama3-1-8b"
# llm_model="llama3-1-8b"


## External IP of client-service svc
ipaddr="115.190.180.7" # vke cluster
## CLUSTER-IP of client-service svc
# ipaddr="10.102.24.174" # local cluster
port=80

# workload_name="ten_request"

workload_name_list=(
    "ten_request"
    # "hundred_request"
    # "MixedSharingRatio10_30_50_70%"
    # "SharingRatio71%"
    # "SharingRatio47%"
    # "SharingRatio28%"
    # "SharingRatio9%"
)


for workload_name in "${workload_name_list[@]}"; do
    python3 update_k8s_env.py \
        --deployment routing-agent-service \
        --namespace default \
        --container routing-agent \
        --env EXPLORATION_ENABLED=${EXPLORATION_ENABLED} \
        --env MIN_NUM_TRAINING_DATA=${MIN_NUM_TRAINING_DATA} \
        --env ENABLE_ONLINE_LEARNING=${ENABLE_ONLINE_LEARNING} \
        --env POD_LABEL_SELECTOR=${POD_LABEL_SELECTOR}

    python3 update_k8s_env.py \
        --deployment aibrix-gateway-plugins \
        --namespace aibrix-system \
        --container gateway-plugin \
        --env ENABLE_FLUSH=${ENABLE_FLUSH} \
        --env FLUSH_PERIOD=${FLUSH_PERIOD} \
        --env MIN_NUM_LOG_MESSAGES_TO_FLUSH=${MIN_NUM_LOG_MESSAGES_TO_FLUSH} \
        --env useRealRequest=1
        # --env LATENCY_METRICS_LOG_PATH=/path/to/your/metrics.log

    ship_model=1
    ship_code=1
    final_model_dir="../training_data/scalable_rl_agent/final_model"
    # final_model_dir="../training_data/merged-data/all/final_model-latency_predictor_ttft"
    # final_model_dir="../training_data/merged-data/all/final_model-latency_predictor_ttft-withoutprefilltoken"

    if [ "${ship_model}" == "1" ] && [ ! -d "${final_model_dir}" ]; then
        echo "Error: Final model directory does not exist: ${final_model_dir}"
        echo "Exiting..."
        exit 1
    fi
    ship_start_time=$(date +%s)
    python ship_all.py --ship_code ${ship_code} --ship_model ${ship_model} --final_model_dir ${final_model_dir} --k8s_cluster ${k8s_cluster}
    # python ship_all_copy.py --ship_code ${ship_code} --ship_model ${ship_model} --final_model_dir ${final_model_dir} --k8s_cluster ${k8s_cluster}
    ship_end_time=$(date +%s)
    ship_took=$((ship_end_time - ship_start_time))
    echo "* ship_all took: ${ship_took}s"
    # python ship_all.py --ship_code 1 --ship_model 1 --final_model_dir "../training_data/merged-data/all/final_model-latency_predictor_ttft" --k8s_cluster vke
    sleep 5

    echo "========================================="
    echo "Running Client in K8s Pod"
    echo "========================================="
    echo "Namespace:           ${NAMESPACE}"
    echo "Pod:                 ${POD_NAME}"
    echo "Container:           ${CONTAINER_NAME}"
    echo "Routing Strategy:    ${routing}"
    echo "Sub-Algorithm:       ${subAlgorithm}"
    echo "Workload:            ${workload_name}"
    echo "Online Learning:     ${ENABLE_ONLINE_LEARNING}"
    echo "Iterations:          ${iterations}"
    echo "Max Tokens:          ${max_tokens}"
    echo "========================================="

    # Find the actual pod name (in case of deployment with generated suffix)
    ACTUAL_POD=$(kubectl get pods -n ${NAMESPACE} -l app=${POD_NAME} -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [ -z "$ACTUAL_POD" ]; then
        echo "Error: No pod found with label app=${POD_NAME} in namespace ${NAMESPACE}"
        echo "Trying to use pod name directly: ${POD_NAME}"
        ACTUAL_POD=${POD_NAME}
    fi

    echo "Using pod: ${ACTUAL_POD}"

    # Wait for pod to be ready
    echo "Waiting for pod to be ready..."
    kubectl wait --for=condition=ready pod/${ACTUAL_POD} -n ${NAMESPACE} --timeout=60s || {
        echo "Error: Pod did not become ready within 60 seconds"
        kubectl describe pod ${ACTUAL_POD} -n ${NAMESPACE}
        exit 1
    }

    workload_path="/app/workload/${workload_name}/workload.jsonl"
    output_dir="/app/output/${workload_name}-${subAlgorithm}-$(date +%Y%m%d_%H%M%S)"
    output_jsonl_path="${output_dir}/output.jsonl"

    # Create output directory in pod
    echo "Creating output directory in pod..."
    kubectl exec -n ${NAMESPACE} ${ACTUAL_POD} -c ${CONTAINER_NAME} -- mkdir -p ${output_dir}

    # Check if workload file exists
    echo "Checking if workload file exists..."
    kubectl exec -n ${NAMESPACE} ${ACTUAL_POD} -c ${CONTAINER_NAME} -- test -f ${workload_path} || {
        echo "Error: Workload file ${workload_path} not found in pod"
        echo "Available workloads:"
        kubectl exec -n ${NAMESPACE} ${ACTUAL_POD} -c ${CONTAINER_NAME} -- find /app/workload -name "*.jsonl"
        exit 1
    }

    # Create local experiment result output directory
    timestamp=$(date +%Y%m%d_%H%M%S)
    experiment_result_output_dir="../workload-and-experiment_results/${workload_name}/${subAlgorithm}"
    if [ "${subAlgorithm}" == "rl_agent" ]; then
        postfix="iter${iterations}"
        experiment_result_output_dir="${experiment_result_output_dir}-${postfix}"
    elif [ "${subAlgorithm}" == "rl_naive" ]; then
        trained_model_data_name=$(echo "$final_model_dir" | awk -F'training_data/' '{print $2}' | cut -d'/' -f1)
        used_data_name=$(echo "$final_model_dir" | awk -F'training_data/' '{print $2}' | cut -d'/' -f2)
        hyperparameter_name=$(echo "$final_model_dir" | awk -F'processed-' '{print $2}')
        hyperparameter_name="${hyperparameter_name}-explr_${EXPLORATION_ENABLED}"
        postfix="onlinelearning_${ENABLE_ONLINE_LEARNING}-trained_on_${trained_model_data_name}_${used_data_name}-${hyperparameter_name}-iter${iterations}"
        experiment_result_output_dir="${experiment_result_output_dir}-${postfix}"
    elif [ "${subAlgorithm}" == "latency_predictor" ]; then
        trained_model_data_name=$(echo "$final_model_dir" | awk -F'training_data/' '{print $2}' | cut -d'/' -f1)
        prediction_metric=$(echo "$final_model_dir" | awk -F'latency_predictor_' '{print $2}')
        used_data_name=$(echo "$final_model_dir" | awk -F'training_data/' '{print $2}' | cut -d'/' -f2)
        postfix="trained_on_${trained_model_data_name}_${used_data_name}-iter${iterations}"
        experiment_result_output_dir="${experiment_result_output_dir}_${prediction_metric}-${postfix}"
    fi

    experiment_result_output_dir="${experiment_result_output_dir}-${timestamp}"
    echo "* experiment_result_output_dir: ${experiment_result_output_dir}"
    if [ ! -d "${experiment_result_output_dir}" ]; then
        mkdir -p "${experiment_result_output_dir}"
    fi

    echo "Starting log collection..."
    kubectl logs -f -n aibrix-system $(kubectl get pods -n aibrix-system | grep aibrix-gateway-plugins | awk '{print $1}') > ${experiment_result_output_dir}/all-aibrix-gateway-plugins.log.txt &
    pid_1=$!
    kubectl logs -f -n default $(kubectl get pods -n default | grep routing-agent-service | awk '{print $1}') > ${experiment_result_output_dir}/all-routing-agent-service.log.txt &
    pid_2=$!

    echo "Starting client in pod..."
    echo "Output will be saved to: ${output_dir}"

    # Run the client using kubectl exec
    kubectl exec -n ${NAMESPACE} ${ACTUAL_POD} -c ${CONTAINER_NAME} -- \
        python3 /app/async-client.py \
            --workload_path ${workload_path} \
            --model ${llm_model} \
            --endpoint http://${ipaddr}:${port} \
            --api_key ${api_key} \
            --output_file_path ${output_jsonl_path} \
            --routing_strategy ${routing} \
            --subAlgorithm ${subAlgorithm} \
            --max_tokens ${max_tokens} \
            --output_dir ${output_dir} \
            --iterations ${iterations} \
            --streaming \
            2>&1 | tee ${experiment_result_output_dir}/client.log.txt

    echo ""
    echo "========================================="
    echo "Client execution completed!"
    echo "========================================="
    echo ""

    # Copy final model
    echo "Copying final_model from pod..."
    python kubectl_cp_from_pod_to_host.py /app/final_model "${experiment_result_output_dir}/final_model" routing-agent-service default

    python kubectl_cp_from_pod_to_host.py /tmp/latency_metrics.log "${experiment_result_output_dir}/latency_metrics.log.txt" gateway-plugins aibrix-system

    # Copy checkpoints (for scalable_rl_agent)
    if [ "${subAlgorithm}" == "scalable_rl_agent" ]; then
        echo "Copying scalable RL agent checkpoints..."
        python kubectl_cp_from_pod_to_host.py /app/final_model/checkpoints "${experiment_result_output_dir}/checkpoints" routing-agent-service default || echo "⚠️  No checkpoints found (agent may not have trained enough steps)"
    fi

    # Process logs
    cat ${experiment_result_output_dir}/all-aibrix-gateway-plugins.log.txt | grep "**@latency_metrics" | grep -v "infer:" > ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins.log.csv
    echo "* all gateway log: ${experiment_result_output_dir}/all-aibrix-gateway-plugins.log.txt"
    echo "* filtered gateway log: ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins.log.csv"
    echo "* routing agent log: ${experiment_result_output_dir}/all-routing-agent-service.log.txt"
    echo "* client log: ${experiment_result_output_dir}/client.log.txt"
    if [ "${subAlgorithm}" == "scalable_rl_agent" ]; then
        echo "* checkpoints: ${experiment_result_output_dir}/checkpoints/"
    fi
    # python plot_latency_timeseries.py ${experiment_result_output_dir}/latency_metrics.log.txt
    kill $pid_1
    kill $pid_2

    kubectl rollout restart deployment client-service
done