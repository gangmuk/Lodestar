#!/bin/bash

kill_and_execute_routing_agent_service() {
    pod_name="$(kubectl get pods -l app=routing-agent-service -o jsonpath='{.items[0].metadata.name}')"
    kubectl exec "${pod_name}" -- pkill -f routing_agent_service.py
    echo "sleep 5 seconds..."
    sleep 5
    kubectl exec "${pod_name}" -- sh -c "nohup python routing_agent_service.py > /dev/null 2>&1 &"
    echo "sleep 5 seconds..."
    sleep 5
}

max_tokens=100
api_key="sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi" # set your api key
output_jsonl_path="./output.jsonl"
model="llama-3-8b-instruct"
port=80
ipaddr=10.0.3.21 # external-ip of envoy-aibrix-system-aibrix-eg-903790dc svc in envoy-gateway-system namespace
# port=8888
# ipaddr=localhost

iterations=1

input_workload_dirs=(
    # "workload/three_requests"

    ## test purpose
    # "workload/prefix-sharing-workload/SharingRatio64%-p4096_s1024_rps5_spp_5_ndp5"

    # "workload/prefix-sharing-workload/p4096_s1024_rps20"

    "workload/prefix-sharing-workload/p4096_s1024_rps10_spp_20_ndp_100"
    
    # "workload/prefix-sharing-workload/p4096_s1024_rps10_spp_10_ndp200"
    # "workload/prefix-sharing-workload/p4096_s1024_rps5_spp_10_ndp200"
    # "workload/prefix-sharing-workload/p4096_s1024_rps7_spp_10_ndp200"
    # "workload/prefix-sharing-workload/p4096_s1024_rps15"
    # "workload/prefix-sharing-workload/p8096_s2048_rps10"
    # "workload/prefix-sharing-workload/p8096_s2048_rps15"

    # "workload/prefix-sharing-workload/SharingRatio75%-p4096_s1024_rps10_spp_20_ndp20"
    # "workload/prefix-sharing-workload/SharingRatio71%-p2048_s512_rps5_spp_10_ndp50-p4096_s1024_rps8_spp_10_ndp50-p8096_s2048_rps3_spp_10_ndp50"
    # "workload/prefix-sharing-workload/SharingRatio47%-p1024_s1024_rps8_spp_20_ndp80-p2048_s2048_rps8_spp_20_ndp80-p4096_s4096_rps3_spp_20_ndp80"
    # "workload/prefix-sharing-workload/SharingRatio47%-p1024_s1024_rps8_spp_20_ndp80-p2048_s2048_rps8_spp_20_ndp80-p4096_s4096_rps2_spp_20_ndp80"
    # "workload/prefix-sharing-workload/SharingRatio28%-p600_s1400_rps8_spp_20_ndp80-p1200_s2800_rps8_spp_20_ndp80-p2400_s5600_rps3_spp_20_ndp80"
    # "workload/prefix-sharing-workload/SharingRatio9%-p200_s1800_rps8_spp_20_ndp80-p400_s3600_rps8_spp_20_ndp80-p800_s7200_rps3_spp_20_ndp80"

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
    # "rl-online-router${delimiter}prefix-cache"
    # "rl-online-router${delimiter}random"
    # # "latency-prediction-based${delimiter}none"
    # "prefix-cache-and-load${delimiter}none"
    # # "flexible-prefix-cache${delimiter}prefix-cache"
    # # "least-latency${delimiter}none"
    # "flexible-prefix-cache${delimiter}random"
)
TTFT_SLO=1000
AVG_TPOT_SLO=50
idxs=(5555)
# for online_learning in true false; do
for online_learning in true; do
# for online_learning in false; do
    for workload_dir in "${input_workload_dirs[@]}"; do
        for idx in "${idxs[@]}"; do
            for config in "${routing_configs[@]}"; do
                routing="${config%%${delimiter}*}"
                subAlgorithm="${config#*${delimiter}}"
                if [ "${online_learning}" = "true" ] && [ "${subAlgorithm}" != "none" ]; then
                    echo "Skipping online learning for ${subAlgorithm} as it is not supported."
                    continue
                fi
                python3 update_k8s_env.py --env TTFT_SLO=${TTFT_SLO} --env AVG_TPOT_SLO=${AVG_TPOT_SLO} --env MODEL=simpler_contextual_bandit --env ENABLE_ONLINE_LEARNING=${online_learning} --deployment routing-agent-service --namespace default
                start_time=$(date +%s)
                
                kubectl rollout restart deployment aibrix-gateway-plugins -n aibrix-system
                kubectl rollout restart deployment routing-agent-service -n default
                sleep 3 && python3 check_ready.py && sleep 3

                if [ "${subAlgorithm}" == "none" ]; then
                    output_dir="${workload_dir}/${config}-onlinelearning_${online_learning}"
                else
                    output_dir="${workload_dir}/${config}"
                fi
                # if output_dir exist, rename
                if [ -d "${output_dir}" ]; then
                    timestamp=$(date +%Y%m%d-%H%M%S)
                    output_dir="${output_dir}-${timestamp}"
                    echo "output_dir already exists, renaming to ${output_dir}"
                fi

                if [ ! -d "${output_dir}" ]; then
                    mkdir -p "${output_dir}"
                fi
                workload_path="${workload_dir}/workload.jsonl"
                client_log_file_name=${output_dir}/"client.log.txt"

                kubectl logs -f -n aibrix-system $(kubectl get pods -n aibrix-system | grep aibrix-gateway-plugins | awk '{print $1}') > ${output_dir}/all-aibrix-gateway-plugins.log.txt &
                pid_1=$!
                kubectl logs -f -n default $(kubectl get pods -n default | grep routing-agent-service | awk '{print $1}') > ${output_dir}/all-routing-agent-service.log.txt &
                pid_2=$!

                echo "========================================"
                echo "* output_dir: ${output_dir}"
                echo "* workload_path: ${workload_path}"
                echo "* client_log_file_name: ${client_log_file_name}"
                echo "* all gateway log: ${output_dir}/all-aibrix-gateway-plugins.log.txt"
                echo "* routing agent service log: ${output_dir}/all-routing-agent-service.log.txt"
                sleep 3
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
                python kubectl_cp_from_pod_to_host.py /app/llm_router.log "${output_dir}/llm_router.log" routing-agent-service default
                cat ${output_dir}/all-aibrix-gateway-plugins.log.txt | grep "**@latency_metrics" > ${output_dir}/filtered-aibrix-gateway-plugins.log.csv
                echo "* filtered gateway log: ${output_dir}/filtered-aibrix-gateway-plugins.log.csv"
                python plot_latency_timeseries.py ${output_dir}/filtered-aibrix-gateway-plugins.log.csv
                kill $pid_1
                kill $pid_2
            done
        done
    done
done