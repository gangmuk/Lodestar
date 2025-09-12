#!/bin/bash

set -e

# api_key="sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi" # set your api key
# model="llama-3-8b-instruct"
# output_jsonl_path="./output.jsonl"

## Inside vke cluster
# port=80
# ipaddr=10.0.3.21 # external-ip of envoy-aibrix-system-aibrix-eg-903790dc svc in envoy-gateway-system namespace
# ## Outside vke cluster
# ipaddr=localhost
# port=8888


#### benchmark related parameters
iterations=1
max_tokens=5000

#### online learning related parameters
## routing_agent_service.py
EXPLORATION_ENABLED=0
ENABLE_ONLINE_LEARNING=false
MIN_NUM_TRAINING_DATA=1000

if [ "${ENABLE_ONLINE_LEARNING}" = "true" ]; then
    ENABLE_FLUSH=1
else
    ENABLE_FLUSH=0
fi
MIN_NUM_LOG_MESSAGES_TO_FLUSH=100
FLUSH_PERIOD=10 # seconds

#### model loading to the container
do_you_want_to_ship=true
ship_final_model_only=1

# final_model_dir=None

final_model_dir_list=(
# "../training_data/p4096_s1024_rps20/rl+random/final_model_backup" # not working again...
# "../training_data/p4096_s1024_rps20/rl+random/final_model" # not working again...
# "../training_data/p4096_s1024_rps20/rl+random/final_model-data_replaced-processed-linear_simple-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50" # this is sort of working

# "../training_data/p4096_s1024_rps20/all/final_model-data_replaced-processed-linear_simple-lr_0.001-ttft_weight_0.5-ttftslo_1000-avgtpotslo_50"
# "../training_data/p4096_s1024_rps20/all/final_model-data_replaced-processed-linear_simple-lr_0.001-ttft_weight_1.0-ttftslo_1000-avgtpotslo_50"
# "../training_data/p4096_s1024_rps20/all/final_model-data_replaced-processed-linear_simple_extended-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50"

# "../training_data/p4096_s1024_rps20/all/final_model-data_replaced-processed-linear_simple-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_none"
"../training_data/p4096_s1024_rps20/all/final_model-data_replaced-processed-linear_simple-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_prefill"

# "../training_data/SharingRatio71%/all/final_model-data_replaced-processed-linear_simple-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_none"
"../training_data/SharingRatio71%/all/final_model-data_replaced-processed-linear_simple-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_prefill"
)


for final_model_dir in "${final_model_dir_list[@]}"; do
    if [ "${do_you_want_to_ship}" == "true" ] && [ ! -d "${final_model_dir}" ]; then
        echo "Final model directory does not exist: ${final_model_dir}"
        exit 1
    fi
done

delimiter="+"
routing_configs=(
    "rl-online-router${delimiter}none"
    # "rl-online-router${delimiter}prefix-cache-1"
    # "rl-online-router${delimiter}prefix-cache-2"
    # "rl-online-router${delimiter}random"
)

input_workload_dirs=(
    # "workload/ten_requests"
    # "workload/p4096_s1024_rps10_spp_20_ndp_100-shortversion"
    # "workload/p4096_s1024_rps10_spp_20_ndp_100-half"
    

    # "workload/p4096_s1024_rps5_spp_10_ndp200"
    # "workload/p4096_s1024_rps7_spp_10_ndp200"
    # "workload/p4096_s1024_rps15"
    # "workload/p8096_s2048_rps10"
    # "workload/p8096_s2048_rps15"

    ## # not enough load to stress
    # "workload/SharingRatio77%-p512_s64_rps20_spp_20_ndp20-p1024_s128_rps10_spp_20_ndp20-p2048_s512_rps5_spp_20_ndp20-p4096_s1024_rps5_spp_20_ndp20" 

    ## not enough load to stress
    # "workload/SharingRatio77%-p512_s64_rps20_spp_20_ndp20-p1024_s128_rps10_spp_20_ndp20-p2048_s512_rps10_spp_20_ndp20-p4096_s1024_rps10_spp_20_ndp20"

    ## the system breaks after the first half of the experiment
    # "workload/SharingRatio78%-p512_s64_rps20_spp_30_ndp20-p1024_s128_rps20_spp_30_ndp20-p2048_s512_rps20_spp_30_ndp20-p4096_s1024_rps20_spp_30_ndp20"

    ## the first half cut of the above. It works.
    # "workload/SharingRatio78%-p512_s64_rps20_spp_30_ndp20-p1024_s128_rps20_spp_30_ndp20-p2048_s512_rps20_spp_30_ndp20-p4096_s1024_rps20_spp_30_ndp20-half"
    
    ## sort of working, but there is no big difference between rl and prefix-cache-1
    # "workload/SharingRatio80%-p512_s64_rps20_spp_20_ndp10-p1024_s128_rps20_spp_20_ndp10-p2048_s512_rps20_spp_20_ndp10-p4096_s1024_rps20_spp_20_ndp10-p8192_s1024_rps5_spp_20_ndp10"

    # "workload/SharingRatio71%-p2048_s512_rps5_spp_10_ndp50-p4096_s1024_rps8_spp_10_ndp50-p8096_s2048_rps3_spp_10_ndp50"

    ## challenging one!
    # "workload/SharingRatio71%-p2048_s512_rps5_spp_10_ndp50-p4096_s1024_rps8_spp_10_ndp50-p8096_s2048_rps3_spp_10_ndp50-half"

    # "workload/SharingRatio47%-p1024_s1024_rps8_spp_20_ndp80-p2048_s2048_rps8_spp_20_ndp80-p4096_s4096_rps3_spp_20_ndp80"
    # "workload/SharingRatio47%-p1024_s1024_rps8_spp_20_ndp80-p2048_s2048_rps8_spp_20_ndp80-p4096_s4096_rps3_spp_20_ndp80-half"

    # "workload/SharingRatio28%-p600_s1400_rps8_spp_20_ndp80-p1200_s2800_rps8_spp_20_ndp80-p2400_s5600_rps3_spp_20_ndp80"
    # "workload/SharingRatio28%-p600_s1400_rps8_spp_20_ndp80-p1200_s2800_rps8_spp_20_ndp80-p2400_s5600_rps3_spp_20_ndp80-half"

    # "workload/SharingRatio9%-p200_s1800_rps8_spp_20_ndp80-p400_s3600_rps8_spp_20_ndp80-p800_s7200_rps3_spp_20_ndp80"

    ## works well.
    "workload/SharingRatio9%-p200_s1800_rps8_spp_20_ndp80-p400_s3600_rps8_spp_20_ndp80-p800_s7200_rps3_spp_20_ndp80-half"

    # "workload/p4096_s1024_rps10_spp_20_ndp_100"
    # "workload/p4096_s1024_rps10_spp_10_ndp200"
)

for workload_dir in "${input_workload_dirs[@]}"; do
    for final_model_dir in "${final_model_dir_list[@]}"; do
        # workload_path="${workload_dir}/workload.jsonl"
        for config in "${routing_configs[@]}"; do
            routing="${config%%${delimiter}*}"
            subAlgorithm="${config#*${delimiter}}"
            echo "subAlgorithm: ${subAlgorithm}"
            if [ "${ENABLE_ONLINE_LEARNING}" = "true" ] && [ "${subAlgorithm}" != "none" ]; then
                echo "Skipping online learning for ${subAlgorithm} as it is not supported."
                continue
            fi

            part1=$(echo "$final_model_dir" | awk -F'training_data/' '{print $2}' | cut -d'/' -f1)
            part2=$(echo "$final_model_dir" | awk -F'processed-' '{print $2}')
            new_var="$part1-$part2"

            if [ "${subAlgorithm}" == "none" ]; then
                output_dir_in_vke_node="${config}-onlinelearning_${ENABLE_ONLINE_LEARNING}-${new_var}"
            else
                output_dir_in_vke_node="${config}-${new_var}"
            fi
            timestamp=$(date +%Y%m%d_%H%M%S)
            output_dir_in_vke_node="${output_dir_in_vke_node}-${timestamp}"
            echo "output_dir_in_vke_node: ${output_dir_in_vke_node}"
            full_path_in_vke_node="${workload_dir}/${output_dir_in_vke_node}"

            kubectl rollout restart deployment aibrix-gateway-plugins -n aibrix-system
            kubectl rollout restart deployment routing-agent-service -n default

            # ## Env var for routing-agent-service deployment
            python3 update_k8s_env.py \
                    --deployment routing-agent-service \
                    --namespace default \
                    --container routing-agent \
                    --env EXPLORATION_ENABLED=${EXPLORATION_ENABLED} \
                    --env MIN_NUM_TRAINING_DATA=${MIN_NUM_TRAINING_DATA} \
                    --env ENABLE_ONLINE_LEARNING=${ENABLE_ONLINE_LEARNING}
                    # --env TTFT_SLO=${TTFT_SLO} \
                    # --env AVG_TPOT_SLO=${AVG_TPOT_SLO} \
                    # --env REWARD_FUNCTION=${REWARD_FUNCTION} \

            ## Env var for aibrix-gateway-plugins deployment
            python3 update_k8s_env.py \
                    --deployment aibrix-gateway-plugins \
                    --namespace aibrix-system \
                    --container gateway-plugin \
                    --env ENABLE_FLUSH=${ENABLE_FLUSH} \
                    --env FLUSH_PERIOD=${FLUSH_PERIOD} \
                    --env MIN_NUM_LOG_MESSAGES_TO_FLUSH=${MIN_NUM_LOG_MESSAGES_TO_FLUSH} \
                    --env useRealRequest=1

            if [ "${do_you_want_to_ship}" == "true" ]; then
                python ship_all.py --ship_final_model_only ${ship_final_model_only} --final_model_dir ${final_model_dir}
            fi
            sleep 5

            #################################################
            ## Send load from vke node ##
            #################################################
            echo "SSH into vke to run client script!!!!"
            ssh root@180.184.82.203 "cd /mnt/vdb/data-driven-routing/client && bash run-client-only.sh ${subAlgorithm} ${workload_dir} ${ENABLE_ONLINE_LEARNING} ${iterations} ${max_tokens} ${full_path_in_vke_node}"

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