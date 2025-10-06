#!/bin/bash

# Script to run the client in a K8s pod using kubectl exec

set -e

# Configuration
NAMESPACE=${NAMESPACE:-default}
POD_NAME=${POD_NAME:-client-service}
CONTAINER_NAME=${CONTAINER_NAME:-client}

# Client parameters (hardcoded)
routing_policy="scalable_rl_agent"
workload_name="ten_request"
ENABLE_ONLINE_LEARNING="false"
iterations=1
max_tokens=1000

# Parameters are now hardcoded above
# To change them, edit the values at the top of this script

# Build subAlgorithm from routing_policy
delimiter="+"
config="rl-online-router${delimiter}${routing_policy}"
routing="${config%%${delimiter}*}"
subAlgorithm="${config#*${delimiter}}"

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

# Configuration for the client
api_key="sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi"
# model="llama2-7b"
# model="tinyllama"
model="llama-3-8b-instruct"
port=80

# CLUSTER-IP of client-service svc
# ipaddr="10.102.24.174" # local cluster
ipaddr="192.168.76.255" # vke cluster

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
experiment_result_output_dir="../workload-and-experiment_results/${workload_name}/${subAlgorithm}-$(date +%Y%m%d_%H%M%S)"
mkdir -p "${experiment_result_output_dir}"

###################
## Start logging ##
###################
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
        --model ${model} \
        --endpoint http://${ipaddr}:${port} \
        --api_key ${api_key} \
        --output_file_path ${output_jsonl_path} \
        --routing_strategy ${routing} \
        --subAlgorithm ${subAlgorithm} \
        --max_tokens ${max_tokens} \
        --output_dir ${output_dir} \
        --iterations ${iterations} \
        --streaming

echo ""
echo "========================================="
echo "Client execution completed!"
echo "========================================="
echo ""

# Copy final model
echo "Copying final_model from pod..."
python kubectl_cp_from_pod_to_host.py /app/final_model "${experiment_result_output_dir}/final_model" routing-agent-service default

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
if [ "${subAlgorithm}" == "scalable_rl_agent" ]; then
    echo "* checkpoints: ${experiment_result_output_dir}/checkpoints/"
fi
# python plot_latency_timeseries.py ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins.log.csv
kill $pid_1
kill $pid_2



