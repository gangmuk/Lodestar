#!/bin/bash

set -e

kill_and_execute_routing_agent_service() {
    pod_name="$(kubectl get pods -l app=routing-agent-service -o jsonpath='{.items[0].metadata.name}')"
    kubectl exec "${pod_name}" -- pkill -f routing_agent_service.py
    echo "sleep 5 seconds..."
    # sleep 5
    kubectl exec "${pod_name}" -- sh -c "nohup python routing_agent_service.py > /dev/null 2>&1 &"
    echo "sleep 5 seconds..."
    sleep 5
}

max_tokens=100
api_key="sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi" # set your api key
output_jsonl_path="./output.jsonl"
model="llama-3-8b-instruct"

## Inside vke cluster
# port=80
# ipaddr=10.0.3.21 # external-ip of envoy-aibrix-system-aibrix-eg-903790dc svc in envoy-gateway-system namespace

## Outside vke cluster
ipaddr=localhost
port=8888

iterations=1

input_workload_dirs=(
    # "workload/ten_requests"
    # "workload/p4096_s1024_rps10_spp_20_ndp_100-shortversion"
    "workload/p4096_s1024_rps10_spp_20_ndp_100-half"
    # "workload/p4096_s1024_rps10_spp_20_ndp_100"
    # "workload/p4096_s1024_rps10_spp_10_ndp200"
    # "workload/p4096_s1024_rps5_spp_10_ndp200"
    # "workload/p4096_s1024_rps7_spp_10_ndp200"
    # "workload/p4096_s1024_rps15"
    # "workload/p8096_s2048_rps10"
    # "workload/p8096_s2048_rps15"

    # "workload/SharingRatio71%-p2048_s512_rps5_spp_10_ndp50-p4096_s1024_rps8_spp_10_ndp50-p8096_s2048_rps3_spp_10_ndp50"
    # "workload/SharingRatio47%-p1024_s1024_rps8_spp_20_ndp80-p2048_s2048_rps8_spp_20_ndp80-p4096_s4096_rps3_spp_20_ndp80"
    # "workload/SharingRatio47%-p1024_s1024_rps8_spp_20_ndp80-p2048_s2048_rps8_spp_20_ndp80-p4096_s4096_rps2_spp_20_ndp80"
    # "workload/SharingRatio28%-p600_s1400_rps8_spp_20_ndp80-p1200_s2800_rps8_spp_20_ndp80-p2400_s5600_rps3_spp_20_ndp80"
    # "workload/SharingRatio9%-p200_s1800_rps8_spp_20_ndp80-p400_s3600_rps8_spp_20_ndp80-p800_s7200_rps3_spp_20_ndp80"
)

delimiter="+"

# final_model_dir="./final_model-seems_almost_random"
# final_model_dir="./final_model-rl_dataset-all_normalized"
# final_model_dir="./final_model-working-model"
final_model_dir="./final_model-new"

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
for online_learning in false; do
# for online_learning in true; do
    for workload_dir in "${input_workload_dirs[@]}"; do
        workload_path="${workload_dir}/workload.jsonl"
        if [ ! -f "${workload_path}" ]; then
            echo "Workload file ${workload_path} does not exist. Exiting."
            exit 1
        fi
        for idx in "${idxs[@]}"; do
            for config in "${routing_configs[@]}"; do
                routing="${config%%${delimiter}*}"
                subAlgorithm="${config#*${delimiter}}"
                if [ "${online_learning}" = "true" ] && [ "${subAlgorithm}" != "none" ]; then
                    echo "Skipping online learning for ${subAlgorithm} as it is not supported."
                    continue
                fi


                # ##################################################################################
                python3 update_k8s_env.py --env TTFT_SLO=${TTFT_SLO} --env AVG_TPOT_SLO=${AVG_TPOT_SLO} --env MODEL=simpler_contextual_bandit --env ENABLE_ONLINE_LEARNING=${online_learning} --deployment routing-agent-service --namespace default
                start_time=$(date +%s)
                
                kubectl rollout restart deployment aibrix-gateway-plugins -n aibrix-system
                kubectl rollout restart deployment routing-agent-service -n default

                if [ ! -d "${final_model_dir}" ]; then
                    echo "Final model directory does not exist: ${final_model_dir}"
                    exit 1
                fi
                python ship_all.py 0 ${final_model_dir}
                ##################################################################################


                sleep 3 && python3 check_ready.py && sleep 3

                ###################################################33
                if [ "${subAlgorithm}" == "none" ]; then
                    output_dir="${workload_dir}/${config}-onlinelearning_${online_learning}"
                else
                    output_dir="${workload_dir}/${config}"
                fi
                ###################################################33
                # output_dir="./output"
                ###################################################33
                
                if [ -d "${output_dir}" ]; then
                    timestamp=$(date +%Y%m%d-%H%M%S)
                    output_dir="${output_dir}-${timestamp}"
                    echo "output_dir already exists, renaming to ${output_dir}"
                fi

                if [ ! -d "${output_dir}" ]; then
                    mkdir -p "${output_dir}"
                fi

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