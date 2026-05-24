#!/bin/bash

set -e

routing_policy=$1
workload_dir=$2
ENABLE_ONLINE_LEARNING=$3
iterations=$4
max_tokens=$5
full_path_in_vke_node=$6

if [ -z "$workload_dir" ]; then
    echo "Usage: $0 <routing_policy> <workload_dir>"
    echo "exiting..."
    exit 1
fi

if [ -z "$routing_policy" ]; then
    echo "Usage: $0 <routing_policy>"
    echo "Example: $0 random"
    exit 1
fi


delimiter="+"
config="rl-online-router${delimiter}${routing_policy}"
routing="${config%%${delimiter}*}"
subAlgorithm="${config#*${delimiter}}"
echo "routing: ${routing}"
echo "subAlgorithm: ${subAlgorithm}"


api_key="sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi"
output_jsonl_path="./output.jsonl"
model="llama-3-8b-instruct"

## Inside vke cluster
port=80
ipaddr=101.126.41.102 # external-ip of envoy-aibrix-system-aibrix-eg-903790dc svc in envoy-gateway-system namespace

workload_path="${workload_dir}/workload.jsonl"
if [ ! -f "${workload_path}" ]; then
    echo "Workload file ${workload_path} does not exist. Exiting."
    exit 1
fi

python3 check_ready.py && sleep 3

if [ ! -d "${full_path_in_vke_node}" ]; then
    mkdir -p "${full_path_in_vke_node}"
fi


kubectl logs -f -n aibrix-system $(kubectl get pods -n aibrix-system | grep aibrix-gateway-plugins | awk '{print $1}') > ${full_path_in_vke_node}/all-aibrix-gateway-plugins.log.txt &
pid_1=$!
kubectl logs -f -n default $(kubectl get pods -n default | grep routing-agent-service | awk '{print $1}') > ${full_path_in_vke_node}/all-routing-agent-service.log.txt &
pid_2=$!

# echo "========================================"
# echo "* full_path_in_vke_node: ${full_path_in_vke_node}"
# echo "* workload_path: ${workload_path}"
# echo "* client_log_file_name: ${client_log_file_name}"
# echo "* all gateway log: ${full_path_in_vke_node}/all-aibrix-gateway-plugins.log.txt"
# echo "* routing agent service log: ${full_path_in_vke_node}/all-routing-agent-service.log.txt"
sleep 10
python3 async-client.py \
        --workload_path ${workload_path} \
        --model ${model} \
        --endpoint http://${ipaddr}:${port} \
        --api_key ${api_key} \
        --output_file_path ${output_jsonl_path} \
        --routing_strategy ${routing} \
        --subAlgorithm ${subAlgorithm} \
        --max_tokens ${max_tokens} \
        --output_dir ${full_path_in_vke_node} \
        --iterations ${iterations} \
        --streaming &> ${full_path_in_vke_node}/"client.log.txt"

python kubectl_cp_from_pod_to_host.py /app/final_model "${full_path_in_vke_node}/final_model" routing-agent-service default
# python kubectl_cp_from_pod_to_host.py /app/llm_router.log "${full_path_in_vke_node}/llm_router.log" routing-agent-service default
cat ${full_path_in_vke_node}/all-aibrix-gateway-plugins.log.txt | grep "**@latency_metrics" | grep -v "infer:" > ${full_path_in_vke_node}/filtered-aibrix-gateway-plugins.log.csv
echo "* filtered gateway log: ${full_path_in_vke_node}/filtered-aibrix-gateway-plugins.log.csv"
python plot_latency_timeseries.py ${full_path_in_vke_node}/filtered-aibrix-gateway-plugins.log.csv
kill $pid_1
kill $pid_2