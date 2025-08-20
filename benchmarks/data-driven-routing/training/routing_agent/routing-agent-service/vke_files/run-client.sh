#!/bin/bash

set -e

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
    # "workload/p4096_s1024_rps10_spp_20_ndp_100-half"
    

    # "workload/p4096_s1024_rps5_spp_10_ndp200"
    # "workload/p4096_s1024_rps7_spp_10_ndp200"
    # "workload/p4096_s1024_rps15"
    # "workload/p8096_s2048_rps10"
    # "workload/p8096_s2048_rps15"

    # "workload/SharingRatio71%-p2048_s512_rps5_spp_10_ndp50-p4096_s1024_rps8_spp_10_ndp50-p8096_s2048_rps3_spp_10_ndp50"
    "workload/SharingRatio71%-p2048_s512_rps5_spp_10_ndp50-p4096_s1024_rps8_spp_10_ndp50-p8096_s2048_rps3_spp_10_ndp50-half"

    # "workload/SharingRatio47%-p1024_s1024_rps8_spp_20_ndp80-p2048_s2048_rps8_spp_20_ndp80-p4096_s4096_rps3_spp_20_ndp80"
    # "workload/SharingRatio47%-p1024_s1024_rps8_spp_20_ndp80-p2048_s2048_rps8_spp_20_ndp80-p4096_s4096_rps3_spp_20_ndp80-half"

    # "workload/SharingRatio28%-p600_s1400_rps8_spp_20_ndp80-p1200_s2800_rps8_spp_20_ndp80-p2400_s5600_rps3_spp_20_ndp80"
    # "workload/SharingRatio28%-p600_s1400_rps8_spp_20_ndp80-p1200_s2800_rps8_spp_20_ndp80-p2400_s5600_rps3_spp_20_ndp80-half"

    # "workload/SharingRatio9%-p200_s1800_rps8_spp_20_ndp80-p400_s3600_rps8_spp_20_ndp80-p800_s7200_rps3_spp_20_ndp80"
    # "workload/SharingRatio9%-p200_s1800_rps8_spp_20_ndp80-p400_s3600_rps8_spp_20_ndp80-p800_s7200_rps3_spp_20_ndp80-half"

    # "workload/p4096_s1024_rps10_spp_20_ndp_100"
    # "workload/p4096_s1024_rps10_spp_10_ndp200"
)

delimiter="+"

# final_model_dir="./final_model-seems_almost_random"
# final_model_dir="./final_model-rl_dataset-all_normalized"
# final_model_dir="./final_model-working-model"
# final_model_dir="./final_model-new"
final_model_dir="../training_data/p4096_s1024_rps20/rl+random/final_model"
if [ ! -d "${final_model_dir}" ]; then
    echo "Final model directory does not exist: ${final_model_dir}"
    exit 1
fi

routing_configs=(
    "rl-online-router${delimiter}none"
    # "rl-online-router${delimiter}prefix-cache-1"
    # "rl-online-router${delimiter}prefix-cache-2"
    # "rl-online-router${delimiter}random"

    # # "latency-prediction-based${delimiter}none"
    # "prefix-cache-and-load${delimiter}none"
    # # "flexible-prefix-cache${delimiter}prefix-cache"
    # # "least-latency${delimiter}none"
    # "flexible-prefix-cache${delimiter}random"
)

TTFT_SLO=1000
AVG_TPOT_SLO=50
MIN_NUM_LOG_MESSAGES_TO_FLUSH=100
ENABLE_FLUSH=0
flushPeriod=10 # seconds
ONLINE_NORMALIZATION_DURING_FLUSH=0

# for ENABLE_ONLINE_LEARNING in true false; do
for ENABLE_ONLINE_LEARNING in false; do
# for ENABLE_ONLINE_LEARNING in true; do
    for workload_dir in "${input_workload_dirs[@]}"; do
        workload_path="${workload_dir}/workload.jsonl"
        if [ ! -f "${workload_path}" ]; then
            echo "Workload file ${workload_path} does not exist. Exiting."
            exit 1
        fi
        for config in "${routing_configs[@]}"; do
            routing="${config%%${delimiter}*}"
            subAlgorithm="${config#*${delimiter}}"
            if [ "${ENABLE_ONLINE_LEARNING}" = "true" ] && [ "${subAlgorithm}" != "none" ]; then
                echo "Skipping online learning for ${subAlgorithm} as it is not supported."
                continue
            fi

            ## Env var for routing-agent-service deployment
            python3 update_k8s_env.py \
                    --env TTFT_SLO=${TTFT_SLO} \
                    --env AVG_TPOT_SLO=${AVG_TPOT_SLO} \
                    --env MODEL=simpler_contextual_bandit \
                    --env ENABLE_ONLINE_LEARNING=${ENABLE_ONLINE_LEARNING} \
                    --env ONLINE_NORMALIZATION_DURING_FLUSH=${ONLINE_NORMALIZATION_DURING_FLUSH} \
                    --deployment routing-agent-service \
                    --namespace default \
                    --container routing-agent

            ## Env var for aibrix-gateway-plugins deployment
            python3 update_k8s_env.py \
                    --env ENABLE_FLUSH=${ENABLE_FLUSH} \
                    --env FLUSH_PERIOD=${flushPeriod} \
                    --env MIN_NUM_LOG_MESSAGES_TO_FLUSH=${MIN_NUM_LOG_MESSAGES_TO_FLUSH} \
                    --deployment aibrix-gateway-plugins \
                    --namespace aibrix-system \
                    --container gateway-plugin

            start_time=$(date +%s)
            
            kubectl rollout restart deployment aibrix-gateway-plugins -n aibrix-system
            kubectl rollout restart deployment routing-agent-service -n default
            python ship_all.py 0 ${final_model_dir}
            sleep 10
            #################################################
            ## Send load from vke node ##
            #################################################
            echo "SSH into vke to run client script!!!!"
            ssh root@180.184.82.203 "cd /mnt/vdb/data-driven-routing/client && bash run-client-only.sh ${subAlgorithm} ${workload_dir}" 

            #################################################
            ## Send load from local machine ##
            #################################################
            # sleep 3 && python3 check_ready.py && sleep 3
            # if [ "${subAlgorithm}" == "none" ]; then
            #     output_dir="${workload_dir}/${config}-onlinelearning_${ENABLE_ONLINE_LEARNING}"
            # else
            #     output_dir="${workload_dir}/${config}"
            # fi
            # if [ -d "${output_dir}" ]; then
            #     timestamp=$(date +%Y%m%d-%H%M%S)
            #     output_dir="${output_dir}-${timestamp}"
            #     echo "output_dir already exists, renaming to ${output_dir}"
            # fi
            # if [ ! -d "${output_dir}" ]; then
            #     mkdir -p "${output_dir}"
            # fi

            # client_log_file_name=${output_dir}/"client.log.txt"
            # kubectl logs -f -n aibrix-system $(kubectl get pods -n aibrix-system | grep aibrix-gateway-plugins | awk '{print $1}') > ${output_dir}/all-aibrix-gateway-plugins.log.txt &
            # pid_1=$!
            # kubectl logs -f -n default $(kubectl get pods -n default | grep routing-agent-service | awk '{print $1}') > ${output_dir}/all-routing-agent-service.log.txt &
            # pid_2=$!

            # echo "========================================"
            # echo "* output_dir: ${output_dir}"
            # echo "* workload_path: ${workload_path}"
            # echo "* client_log_file_name: ${client_log_file_name}"
            # echo "* all gateway log: ${output_dir}/all-aibrix-gateway-plugins.log.txt"
            # echo "* routing agent service log: ${output_dir}/all-routing-agent-service.log.txt"
            # sleep 3
            # python3 async-client.py \
            #         --workload_path ${workload_path} \
            #         --model ${model} \
            #         --endpoint http://${ipaddr}:${port} \
            #         --api_key ${api_key} \
            #         --output_file_path ${output_jsonl_path} \
            #         --routing_strategy ${routing} \
            #         --subAlgorithm ${subAlgorithm} \
            #         --max_tokens ${max_tokens} \
            #         --output_dir ${output_dir} \
            #         --iterations ${iterations} \
            #         --streaming &> ${client_log_file_name}
            #         # --streaming 2>&1 | tee ${client_log_file_name}
            # duration=$(( $(date +%s) - start_time ))
            # python kubectl_cp_from_pod_to_host.py /app/final_model "${output_dir}/final_model" routing-agent-service default
            # python kubectl_cp_from_pod_to_host.py /app/llm_router.log "${output_dir}/llm_router.log" routing-agent-service default
            # cat ${output_dir}/all-aibrix-gateway-plugins.log.txt | grep "**@latency_metrics" > ${output_dir}/filtered-aibrix-gateway-plugins.log.csv
            # echo "* filtered gateway log: ${output_dir}/filtered-aibrix-gateway-plugins.log.csv"
            # python plot_latency_timeseries.py ${output_dir}/filtered-aibrix-gateway-plugins.log.csv
            # kill $pid_1
            # kill $pid_2
            #################################################
            #################################################
            #################################################

        done
    done
done