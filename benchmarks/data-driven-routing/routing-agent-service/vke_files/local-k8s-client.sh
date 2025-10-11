#!/bin/bash

set -e

routing_policy=$1
workload_dir=$2
ENABLE_ONLINE_LEARNING=$3
iterations=$4
max_tokens=$5
experiment_result_output_dir=$6

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
model="llama2-7b"

# ipaddr=115.190.180.7 # external-ip of vke cluster's envoy-aibrix-system-aibrix-eg-903790dc svc in envoy-gateway-system namespace
ipaddr=10.102.24.174
port=80

workload_path="${workload_dir}/workload.jsonl"
if [ ! -f "${workload_path}" ]; then
    echo "Workload file ${workload_path} does not exist. Exiting."
    exit 1
fi

python3 async-client.py \
        --workload_path ${workload_path} \
        --model ${model} \
        --endpoint http://${ipaddr}:${port} \
        --api_key ${api_key} \
        --output_file_path ${output_jsonl_path} \
        --routing_strategy ${routing} \
        --subAlgorithm ${subAlgorithm} \
        --max_tokens ${max_tokens} \
        --output_dir ${experiment_result_output_dir} \
        --iterations ${iterations} \
        --streaming
        # --streaming &> ${experiment_result_output_dir}/"client.log.txt"
