#!/bin/bash

# input_workload_path="./workload/i_like_apple_request.jsonl"
# input_workload_path="./workload/one_request.jsonl"
# input_workload_path="./workload/ten_request"
# input_workload_path="./workload/5s.jsonl"
# input_workload_path="./workload/5min-later-part-init.jsonl"
# input_workload_path="./workload/prefix-sharing-workload/prefixsharingworkload-p1024_s128_rps5.jsonl"
# input_workload_path="./workload/prefix-sharing-workload/p1024_s128_rps5-p2048_s128_rps5-p4096_s128_rps5.jsonl"
# input_workload_path="./workload/prefix-sharing-workload/p1024_s128_rps7-p2048_s128_rps7-p4096_s128_rps7-p8096_s128_rps7.jsonl"
# input_workload_path="./workload/prefix-sharing-workload/p1024_s128_rps10-p2048_s128_rps10-p4096_s128_rps10-p8096_s128_rps10.jsonl"
# input_workload_path="./workload/prefix-sharing-workload/p2048_s512_rps10-p4096_s1024_rps10-p8096_s2048_rps10-p16192_s4096_rps10.jsonl"

max_tokens=100
api_key="sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi" # set your api key
output_jsonl_path="./output.jsonl"
model="llama-3-8b-instruct"
port=80
ipaddr=10.0.3.21 # external-ip of envoy-aibrix-system-aibrix-eg-903790dc svc in envoy-gateway-system namespace
# port=8888
# ipaddr=localhost

iterations=2

input_workload_dirs=(
    # "workload/one_request"

    # "workload/prefix-sharing-workload/p4096_s1024_rps5"
    # "workload/prefix-sharing-workload/p4096_s1024_rps10"
    "workload/prefix-sharing-workload/p4096_s1024_rps15"
    # "workload/prefix-sharing-workload/p4096_s1024_rps20"

    # "workload/prefix-sharing-workload/p1024_s128_rps10-p2048_s128_rps10-p4096_s128_rps10-p8096_s128_rps10"
    # "workload/prefix-sharing-workload/p2048_s512_rps10-p4096_s1024_rps10-p8096_s2048_rps5"
    
    
    # "workload/prefix-sharing-workload/p2048_s512_rps10-p4096_s1024_rps10-p8096_s2048_rps10"
    # "workload/prefix-sharing-workload/p2048_s512_rps10-p4096_s1024_rps10-p8096_s2048_rps5-p16192_s4096_rps3-pp"

    # "workload/prefix-sharing-workload/merged-comprehensive-workload"

    # workload/prefix-sharing-workload/comprehensive_set/random/basic-load-patterns
    # workload/prefix-sharing-workload/comprehensive_set/random/chatbot-simulation
    # workload/prefix-sharing-workload/comprehensive_set/random/input-size-impact
    # workload/prefix-sharing-workload/comprehensive_set/random/output-size-impact
    # workload/prefix-sharing-workload/comprehensive_set/random/production-like-mixed-load
    # workload/prefix-sharing-workload/comprehensive_set/random/burst-patterns
    # workload/prefix-sharing-workload/comprehensive_set/random/content-generation
    # workload/prefix-sharing-workload/comprehensive_set/random/large-context-testing
    # workload/prefix-sharing-workload/comprehensive_set/random/prefix-sharing-efficiency
    # workload/prefix-sharing-workload/comprehensive_set/random/quick-qa
)

delimiter="+"
routing_configs=(
    "rl-online-router${delimiter}none"
    "rl-online-router${delimiter}prefix-cache"
    "rl-online-router${delimiter}random"
    # # "latency-prediction-based${delimiter}none"
    # "prefix-cache-and-load${delimiter}none"
    # # "flexible-prefix-cache${delimiter}prefix-cache"
    # # "least-latency${delimiter}none"
    # "flexible-prefix-cache${delimiter}random"
)
TTFT_SLO=400
AVG_TPOT_SLO=40
idxs=(5555)
for workload_dir in "${input_workload_dirs[@]}"; do
    for idx in "${idxs[@]}"; do
        for config in "${routing_configs[@]}"; do
            routing="${config%%${delimiter}*}"
            subAlgorithm="${config#*${delimiter}}"
            output_dir="${workload_dir}/${config}"
            # if output_dir exist, rename
            if [ -d "${output_dir}" ]; then
                timestamp=$(date +%Y%m%d-%H%M%S)
                output_dir="${output_dir}-${timestamp}"
                echo "output_dir already exists, renaming to ${output_dir}"
            fi

            if [ ! -d "${output_dir}" ]; then
                mkdir -p "${output_dir}"
            fi
            python update_slo_env.py --ttft-slo ${TTFT_SLO} --avg-tpot-slo ${AVG_TPOT_SLO}
            workload_path="${workload_dir}/workload.jsonl"
            client_log_file_name=${output_dir}/"client.log.txt"
            start_time=$(date +%s)
            kubectl rollout restart deployment aibrix-gateway-plugins -n aibrix-system
            # kubectl rollout restart deployment latency-predictor-service -n default
            kubectl rollout restart deployment routing-agent-service -n default
            python3 check_ready.py
            sleep 3
            
            kubectl logs -f -n aibrix-system $(kubectl get pods -n aibrix-system | grep aibrix-gateway-plugins | awk '{print $1}') > ${output_dir}/all-aibrix-gateway-plugins.log.txt &
            pid_1=$!

            # kubectl logs -f -n default $(kubectl get pods -n default | grep latency-predictor-service | awk '{print $1}') > ${output_dir}/all-latency-predictor-service.log.txt &
            # pid_2=$!

            kubectl logs -f -n default $(kubectl get pods -n default | grep routing-agent-service | awk '{print $1}') > ${output_dir}/all-routing-agent-service.log.txt &
            pid_3=$!

            echo "========================================"

            python kubectl_cp_from_host_to_pod.py hyperparameters.txt /app/hyperparameters.txt routing-agent-service default

            echo "* output_dir: ${output_dir}"
            echo "* workload_path: ${workload_path}"
            echo "* client_log_file_name: ${client_log_file_name}"
            echo "* all gateway log: ${output_dir}/all-aibrix-gateway-plugins.log.txt"
            echo "* routing agent service log: ${output_dir}/all-routing-agent-service.log.txt"
            python3 async-client.py \
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
                    --streaming &> ${client_log_file_name}
                    # --streaming 2>&1 | tee ${client_log_file_name}
            duration=$(( $(date +%s) - start_time ))

            python kubectl_cp_from_pod_to_host.py /app/final_model "${output_dir}/final_model" routing-agent-service default
            python kubectl_cp_from_pod_to_host.py /app/final_model "${output_dir}/final_model" routing-agent-service default
            python kubectl_cp_from_pod_to_host.py /app/llm_router.log "${output_dir}/llm_router.log" routing-agent-service default
            python kubectl_cp_from_pod_to_host.py /app/global_tensor_dataset.pt "${output_dir}/global_tensor_dataset.pt" routing-agent-service default



            cat ${output_dir}/all-aibrix-gateway-plugins.log.txt | grep "**@latency_metrics" > ${output_dir}/filtered-aibrix-gateway-plugins.log.csv
            
            python plot_latency_timeseries.py ${output_dir}/filtered-aibrix-gateway-plugins.log.csv

            echo "* filtered gateway log: ${output_dir}/filtered-aibrix-gateway-plugins.log.csv"
            # echo "* latency predictor service log: ${output_dir}/all-latency-predictor-service.log.txt"
            sleep 5

            kill $pid_1
            # kill $pid_2
            kill $pid_3
            echo "* Total time taken for the experiment: ${duration} seconds"
            echo "========================================"
        done
    done
done