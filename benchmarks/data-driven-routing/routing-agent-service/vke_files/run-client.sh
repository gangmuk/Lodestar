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
ship_model=0
ship_code=1

# final_model_dir=None

final_model_dir_list=(

####### contextual bandit

# "../training_data/p4096_s1024_rps20/rl+random/final_model_backup" # not working again...
# "../training_data/p4096_s1024_rps20/rl+random/final_model" # not working again...
# "../training_data/p4096_s1024_rps20/rl+random/final_model-data_replaced-processed-linear_simple-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50" # this is sort of working

# "../training_data/p4096_s1024_rps20/all/final_model-data_replaced-processed-linear_simple-lr_0.001-ttft_weight_0.5-ttftslo_1000-avgtpotslo_50"
# "../training_data/p4096_s1024_rps20/all/final_model-data_replaced-processed-linear_simple-lr_0.001-ttft_weight_1.0-ttftslo_1000-avgtpotslo_50"
# "../training_data/p4096_s1024_rps20/all/final_model-data_replaced-processed-linear_simple_extended-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50"

# "../training_data/p4096_s1024_rps20/all/final_model-data_replaced-processed-linear_simple-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_prefill"
# "../training_data/p4096_s1024_rps20/all/final_model-data_replaced-processed-linear_simple-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_prefill-hidden_dim_128"


#### one of the best
"../training_data/p4096_s1024_rps20/all/final_model-data_replaced-processed-linear_simple-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_none-hidden_dim_64-lrs_grad_adapt"

# "../training_data/p4096_s1024_rps20/all/final_model-data_replaced-processed-piecewise_linear_steeper_gradient-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_none-hidden_dim_128"

# "../training_data/p4096_s1024_rps20/all/final_model-data_replaced-processed-linear_simple_extended-lr_0.001-ttft_weight_0.5-ttftslo_1000-avgtpotslo_50-without_none-hidden_dim_128"

# "../training_data/p4096_s1024_rps20/all/final_model-data_replaced-processed-linear_simple_extended-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_prefill_tokens-hidden_dim_128-lrs_exp"


####
# "../training_data/p4096_s1024_rps20/all/final_model-data_replaced-processed-linear_simple_extended-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_none-hidden_dim_128-lrs_exp"

# "../training_data/p4096_s1024_rps20/all/linear_simple_extended-lr_0.001-ttft_weight_0.1-ttftslo_1000-avgtpotslo_50-without_none-hidden_dim_128"

# "../training_data/merged-data/all/linear_simple_extended-lr_0.001-ttft_weight_0.1-ttftslo_1000-avgtpotslo_50-without_none-hidden_dim_128"

# "../training_data/merged-data/all/final_model-data_replaced-processed-piecewise_linear_steeper_gradient-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_prefill_tokens-hidden_dim_128"

# "../training_data/merged-data/all/final_model-data_replaced-processed-linear_simple-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_none-hidden_dim_128"

####    
# "../training_data/merged-data/all/final_model-data_replaced-processed-linear_simple_extended-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_none-hidden_dim_128"

#### kinda worked for online learning with iterations=10
# "../training_data/merged-data/all/final_model-data_replaced-processed-piecewise_linear_steeper_gradient-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_none-hidden_dim_128"

# "../training_data/merged-data/all/final_model-data_replaced-processed-piecewise_linear_steeper_gradient-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_none-hidden_dim_128"

# "../training_data/SharingRatio71%/all/final_model-data_replaced-processed-linear_simple-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_none"
# "../training_data/SharingRatio71%/all/final_model-data_replaced-processed-linear_simple-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_prefill"

## overfit test (2025-09-21)
# "../training_data/SharingRatio71%/prefix/final_model-data-processed-linear_simple-lr_0.
# 001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_prefill_tokens"

##(2025-09-21)
# "../training_data/SharingRatio71%/all/final_model-data-processed-linear_simple_extended-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_prefill_tokens-hidden_dim_64-lrs_exp"


## this does not work... don't know why
# "../training_data/SharingRatio9%/all/final_model-data_replaced-processed-linear_simple_extended-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_none-hidden_dim_128-lrs_exp"

# "../training_data/SharingRatio9%/all/final_model-data_replaced-processed-linear_simple-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_none-hidden_dim_128-lrs_grad_adapt"

## actually not the worst but not best
# "../training_data/SharingRatio9%/all/final_model-data_replaced-processed-linear_simple-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_prefill_tokens"

# "../training_data/SharingRatio9%/all/final_model-data_replaced-processed-linear_simple-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_none-hidden_dim_128"



####### latency predictor
# "../training_data/merged-data/all/final_model-data_replaced-processed-linear_simple_extended-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_prefill_tokens-hidden_dim_64-lrs_exp-latency_predictor"

# "../training_data/merged-data/all/final_model-latency_predictor_e2e_latency"
# "../training_data/merged-data/all/final_model-latency_predictor_avg_tpot"
"../training_data/merged-data/all/final_model-latency_predictor_ttft"
)


for final_model_dir in "${final_model_dir_list[@]}"; do
    if [ "${ship_model}" == "1" ] && [ ! -d "${final_model_dir}" ]; then
        echo "Error: Final model directory does not exist: ${final_model_dir}"
        echo "Exiting..."
        exit 1
    fi
done

delimiter="+"
routing_configs=(
    "rl-online-router${delimiter}rl_agent"
    # "rl-online-router${delimiter}rl_naive"
    # "rl-online-router${delimiter}latency_predictor"
    # "rl-online-router${delimiter}prefix_cache_1"
    # "rl-online-router${delimiter}prefix_cache_2"
    # "rl-online-router${delimiter}preble"
    # "rl-online-router${delimiter}random"
)

input_workload_dirs=(
    # "workload/p4096_s1024_rps10_spp_20_ndp_100-shortversion"
    # "workload/p4096_s1024_rps10_spp_20_ndp_100-half"
    
    # "workload/p4096_s1024_rps10_spp_20_ndp_100"
    # "workload/p4096_s1024_rps10_spp_10_ndp200"

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
    # "workload/SharingRatio47%-p1024_s1024_rps8_spp_20_ndp80-p2048_s2048_rps8_spp_20_ndp80-p4096_s4096_rps3_spp_20_ndp80"
    # "workload/SharingRatio28%-p600_s1400_rps8_spp_20_ndp80-p1200_s2800_rps8_spp_20_ndp80-p2400_s5600_rps3_spp_20_ndp80"
    # "workload/SharingRatio9%-p200_s1800_rps8_spp_20_ndp80-p400_s3600_rps8_spp_20_ndp80-p800_s7200_rps3_spp_20_ndp80"

    # "workload/ten_request"
    # "workload/SharingRatio71%-p2048_s512_rps5_spp_10_ndp50-p4096_s1024_rps8_spp_10_ndp50-p8096_s2048_rps3_spp_10_ndp50-half"
    # "workload/SharingRatio47%-p1024_s1024_rps8_spp_20_ndp80-p2048_s2048_rps8_spp_20_ndp80-p4096_s4096_rps3_spp_20_ndp80-half"
    # "workload/SharingRatio28%-p600_s1400_rps8_spp_20_ndp80-p1200_s2800_rps8_spp_20_ndp80-p2400_s5600_rps3_spp_20_ndp80-half"
    # "workload/SharingRatio9%-p200_s1800_rps8_spp_20_ndp80-p400_s3600_rps8_spp_20_ndp80-p800_s7200_rps3_spp_20_ndp80-half"


    ############################
    ## local k8s for mock app ##
    ############################
    # basedir=""
    "../workload-and-experiment_results/ten_request"
    # "${basedir}/SharingRatio71%"
    # "${basedir}/SharingRatio47%"
    # "${basedir}/SharingRatio28%"
    # "${basedir}/SharingRatio9%"
)

# # Function to cleanup remote processes on interrupt
cleanup_remote() {
    echo "Interrupt received, cleaning up remote processes..."
    # Kill the background SSH process
    if [ ! -z "$ssh_pid" ] && kill -0 $ssh_pid 2>/dev/null; then
        kill $ssh_pid 2>/dev/null
    fi
    # Kill remote client processes - specifically async-client.py and kubectl logging processes
    ssh root@180.184.82.203 "pkill -f 'async-client.py' && pkill -f 'kubectl logs' && pkill -f 'run-client-only.sh'" 2>/dev/null || true
    echo "Remote cleanup completed."
    exit 1
}

repeat_times=1

for i in $(seq 1 ${repeat_times}); do
    for workload_dir in "${input_workload_dirs[@]}"; do
        # if it does not exist, exit
        if [ ! -f "${workload_dir}/workload.jsonl" ]; then
            echo "Error: ${workload_dir}/workload.jsonl does not exist. exiting..."
            echo "Exiting..."
            exit 1
        fi
        for config in "${routing_configs[@]}"; do
            for final_model_dir in "${final_model_dir_list[@]}"; do
                routing="${config%%${delimiter}*}"
                subAlgorithm="${config#*${delimiter}}"
                echo "subAlgorithm: ${subAlgorithm}"
                # if [ "${ENABLE_ONLINE_LEARNING}" = "true" ] && [ "${subAlgorithm}" != "none" ]; then
                #     echo "Skipping online learning for ${subAlgorithm} as it is not supported."
                #     continue
                # fi
                workload_name=$(echo "$workload_dir" | awk -F'/' '{print $3}')
                # workload_name=$(echo "$workload_name" | awk -F'-' '{print $1}')
                echo "* workload_dir: ${workload_dir}"
                echo "* workload_name: ${workload_name}"
                timestamp=$(date +%Y%m%d_%H%M%S)
                output_dir_in_vke_node="${config}"
                experiment_result_output_dir="../workload-and-experiment_results/${workload_name}/${subAlgorithm}"
                if [ "${subAlgorithm}" == "rl_agent" ]; then
                    postfix="iter${iterations}"
                    output_dir_in_vke_node="${output_dir_in_vke_node}-${postfix}"
                    experiment_result_output_dir="${experiment_result_output_dir}-${postfix}"
                elif [ "${subAlgorithm}" == "rl_naive" ]; then
                    trained_model_data_name=$(echo "$final_model_dir" | awk -F'training_data/' '{print $2}' | cut -d'/' -f1)
                    used_data_name=$(echo "$final_model_dir" | awk -F'training_data/' '{print $2}' | cut -d'/' -f2)
                    hyperparameter_name=$(echo "$final_model_dir" | awk -F'processed-' '{print $2}')
                    hyperparameter_name="${hyperparameter_name}-explr_${EXPLORATION_ENABLED}"
                    postfix="onlinelearning_${ENABLE_ONLINE_LEARNING}-trained_on_${trained_model_data_name}_${used_data_name}-${hyperparameter_name}-iter${iterations}"
                    output_dir_in_vke_node="${output_dir_in_vke_node}-${postfix}"
                    experiment_result_output_dir="${experiment_result_output_dir}-${postfix}"
                elif [ "${subAlgorithm}" == "latency_predictor" ]; then
                    trained_model_data_name=$(echo "$final_model_dir" | awk -F'training_data/' '{print $2}' | cut -d'/' -f1)
                    prediction_metric=$(echo "$final_model_dir" | awk -F'latency_predictor_' '{print $2}')
                    used_data_name=$(echo "$final_model_dir" | awk -F'training_data/' '{print $2}' | cut -d'/' -f2)
                    postfix="trained_on_${trained_model_data_name}_${used_data_name}-iter${iterations}"
                    output_dir_in_vke_node="${output_dir_in_vke_node}-${postfix}"
                    experiment_result_output_dir="${experiment_result_output_dir}_${prediction_metric}-${postfix}"
                fi

                output_dir_in_vke_node="${output_dir_in_vke_node}-${timestamp}"
                full_path_in_vke_node="${workload_dir}/${output_dir_in_vke_node}"
                experiment_result_output_dir="${experiment_result_output_dir}-${timestamp}"
                echo "* output_dir_in_vke_node: ${output_dir_in_vke_node}"
                # echo "* full_path_in_vke_node: ${full_path_in_vke_node}"
                echo "* experiment_result_output_dir: ${experiment_result_output_dir}"
                if [ ! -d "${experiment_result_output_dir}" ]; then
                    mkdir -p "${experiment_result_output_dir}"
                fi

                # kubectl rollout restart deployment aibrix-gateway-plugins -n aibrix-system
                # kubectl rollout restart deployment routing-agent-service -n default
                
                # kubectl rollout restart deployment llama-3-8b-instruct -n default

                # python3 check_ready.py --deployment aibrix-gateway-plugins --namespace aibrix-system
                # python3 check_ready.py --deployment routing-agent-service --namespace default
                # python3 check_ready.py --deployment llama-3-8b-instruct --namespace default
                # python3 check_ready.py --deployment mock-app --namespace default

                # ## Env var for routing-agent-service deployment
                python3 update_k8s_env.py \
                        --deployment routing-agent-service \
                        --namespace default \
                        --container routing-agent \
                        --env EXPLORATION_ENABLED=${EXPLORATION_ENABLED} \
                        --env MIN_NUM_TRAINING_DATA=${MIN_NUM_TRAINING_DATA} \
                        --env ENABLE_ONLINE_LEARNING=${ENABLE_ONLINE_LEARNING}

                ## Env var for aibrix-gateway-plugins deployment
                python3 update_k8s_env.py \
                        --deployment aibrix-gateway-plugins \
                        --namespace aibrix-system \
                        --container gateway-plugin \
                        --env ENABLE_FLUSH=${ENABLE_FLUSH} \
                        --env FLUSH_PERIOD=${FLUSH_PERIOD} \
                        --env MIN_NUM_LOG_MESSAGES_TO_FLUSH=${MIN_NUM_LOG_MESSAGES_TO_FLUSH} \
                        --env useRealRequest=1

                python ship_all.py --ship_code ${ship_code} --ship_model ${ship_model} --final_model_dir ${final_model_dir}
                sleep 5

                ###################
                ## Start logging ##
                ###################
                kubectl logs -f -n aibrix-system $(kubectl get pods -n aibrix-system | grep aibrix-gateway-plugins | awk '{print $1}') > ${experiment_result_output_dir}/all-aibrix-gateway-plugins.log.txt &
                pid_1=$!
                kubectl logs -f -n default $(kubectl get pods -n default | grep routing-agent-service | awk '{print $1}') > ${experiment_result_output_dir}/all-routing-agent-service.log.txt &
                pid_2=$!

                # Set up trap for SIGINT (Ctrl+C)
                trap cleanup_remote SIGINT

                #########################
                ## Send load from host ##
                #########################

                python3 check_ready.py --deployment aibrix-gateway-plugins --namespace aibrix-system
                python3 check_ready.py --deployment routing-agent-service --namespace default
                # python3 check_ready.py --deployment llama-3-8b-instruct --namespace default
                python3 check_ready.py --deployment mock-app --namespace default
                echo "Wait for 10 seconds..."
                sleep 15

                echo "Send load from host!!!!"
                bash local-k8s-client.sh ${subAlgorithm} ${workload_dir} ${ENABLE_ONLINE_LEARNING} ${iterations} ${max_tokens} ${experiment_result_output_dir}
                # ssh_pid=$!

                #############################
                ## Send load from vke node ##
                #############################
                # echo "SSH into vke to run client script!!!!"

                # # Run SSH command in background
                # ssh root@180.184.82.203 "cd /mnt/vdb/data-driven-routing/client && bash run-client-only.sh ${subAlgorithm} ${workload_dir} ${ENABLE_ONLINE_LEARNING} ${iterations} ${max_tokens} ${full_path_in_vke_node}" &
                # ssh_pid=$!


                # # Wait for the SSH command to complete
                # wait $ssh_pid
                # ssh_pid=""

                # # Clear the trap since we're done with this iteration
                # trap - SIGINT

                #################
                ## End logging ##
                #################
                python kubectl_cp_from_pod_to_host.py /app/final_model "${experiment_result_output_dir}/final_model" routing-agent-service default
                cat ${experiment_result_output_dir}/all-aibrix-gateway-plugins.log.txt | grep "**@latency_metrics" | grep -v "infer:" > ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins.log.csv
                echo "* all gateway log: ${experiment_result_output_dir}/all-aibrix-gateway-plugins.log.txt"
                echo "* filtered gateway log: ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins.log.csv"
                echo "* routing agent log: ${experiment_result_output_dir}/all-routing-agent-service.log.txt"
                # python plot_latency_timeseries.py ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins.log.csv
                kill $pid_1
                kill $pid_2
                
                if [ "${subAlgorithm}" != "none" ]; then
                    break
                fi
            done
        done
    done
done