#!/bin/bash

# Script to run the client in a K8s pod using kubectl exec

set -e

# Retry function for transient failures (e.g., http2: client connection lost)
retry_command() {
    local max_attempts=${RETRY_MAX_ATTEMPTS:-5}
    local delay=${RETRY_DELAY:-5}
    local attempt=1
    local exit_code=0

    while [ $attempt -le $max_attempts ]; do
        if "$@"; then
            return 0
        else
            exit_code=$?
            if [ $attempt -lt $max_attempts ]; then
                echo "⚠️  Command failed (attempt $attempt/$max_attempts). Retrying in ${delay}s..."
                echo "   Command: $*"
                sleep $delay
                # Exponential backoff with cap at 60 seconds
                delay=$((delay * 2))
                if [ $delay -gt 60 ]; then
                    delay=60
                fi
            fi
            attempt=$((attempt + 1))
        fi
    done
    echo "❌ Command failed after $max_attempts attempts: $*"
    return $exit_code
}

# Configuration
CLIENT_SERVICE_POD_NAME=client-service
CLIENT_SERVICE_CONTAINER_NAME=client
k8s_cluster="vke"
# k8s_cluster="aws"
# target_gpu="NVIDIA-L40S"
# target_gpu="NVIDIA-A10"
# Define experiment configurations
# Format: "routing_policy|workload_category|workload_name|target_gpu|rps|total_num_episodes|prefix_hit_threshold"
# Note: prefix_hit_threshold (7th field) is optional. If specified and routing_policy is "prefix_hit_threshold_or_least_request", it will be used as PREFIX_HIT_THRESHOLD. Otherwise, PREFIX_HIT_THRESHOLD defaults to 50.

target_gpu="NVIDIA-A30"
workload_mode="benchmark" # benchmark, profiling
experiment_configs=(

    #########################################################
    #########################################################
    #########################################################
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio71%|${target_gpu}|6|1|20"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio71%|${target_gpu}|6|1|40"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio71%|${target_gpu}|6|1|60"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio71%|${target_gpu}|6|1|80"

    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio71%|${target_gpu}|8|1|20"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio71%|${target_gpu}|8|1|40"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio71%|${target_gpu}|8|1|60"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio71%|${target_gpu}|8|1|80"

    #########################################################
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio47%|${target_gpu}|6|1|20"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio47%|${target_gpu}|6|1|40"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio47%|${target_gpu}|6|1|60"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio47%|${target_gpu}|6|1|80"

    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio47%|${target_gpu}|8|1|20"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio47%|${target_gpu}|8|1|40"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio47%|${target_gpu}|8|1|60"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio47%|${target_gpu}|8|1|80"

    #########################################################
    ## stopped here for now.
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio28%|${target_gpu}|6|1|20"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio28%|${target_gpu}|6|1|40"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio28%|${target_gpu}|6|1|60"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio28%|${target_gpu}|6|1|80"

    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio28%|${target_gpu}|8|1|20"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio28%|${target_gpu}|8|1|40"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio28%|${target_gpu}|8|1|60"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio28%|${target_gpu}|8|1|80"

    # #########################################################
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio9%|${target_gpu}|6|1|20"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio9%|${target_gpu}|8|1|40"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio9%|${target_gpu}|8|1|60"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio9%|${target_gpu}|8|1|80"

    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio9%|${target_gpu}|8|1|20"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio9%|${target_gpu}|8|1|40"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio9%|${target_gpu}|8|1|60"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|SharingRatio9%|${target_gpu}|8|1|80"

    #########################################################

    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|6|1|20"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|6|1|40"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|6|1|60"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|6|1|80"

    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|8|1|20"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|8|1|40"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|8|1|60"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|8|1|80"

    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|10|1|20"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|10|1|40"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|10|1|60"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|10|1|80"

    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|12|1|20"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|12|1|40"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|12|1|60"
    # "prefix_hit_threshold_or_least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|12|1|80"

    #########################################################

    # "prefix_hit_threshold_or_least_request|mooncake|conversation-2|${target_gpu}|10|2|20"
    # "prefix_hit_threshold_or_least_request|mooncake|conversation-2|${target_gpu}|10|2|40"
    # "prefix_hit_threshold_or_least_request|mooncake|conversation-2|${target_gpu}|10|2|60"
    # "prefix_hit_threshold_or_least_request|mooncake|conversation-2|${target_gpu}|10|2|80"
    
    # "prefix_hit_threshold_or_least_request|mooncake|conversation-2|${target_gpu}|15|2|20"
    # "prefix_hit_threshold_or_least_request|mooncake|conversation-2|${target_gpu}|15|2|40"
    # "prefix_hit_threshold_or_least_request|mooncake|conversation-2|${target_gpu}|15|2|60"
    # "prefix_hit_threshold_or_least_request|mooncake|conversation-2|${target_gpu}|15|2|80"

    # "prefix_hit_threshold_or_least_request|mooncake|conversation-2|${target_gpu}|20|2|20"
    # "prefix_hit_threshold_or_least_request|mooncake|conversation-2|${target_gpu}|20|2|40"
    # "prefix_hit_threshold_or_least_request|mooncake|conversation-2|${target_gpu}|20|2|60"
    # "prefix_hit_threshold_or_least_request|mooncake|conversation-2|${target_gpu}|20|2|80"

    # "prefix_hit_threshold_or_least_request|mooncake|conversation-2|${target_gpu}|25|2|20"
    # "prefix_hit_threshold_or_least_request|mooncake|conversation-2|${target_gpu}|25|2|40"
    # "prefix_hit_threshold_or_least_request|mooncake|conversation-2|${target_gpu}|25|2|60"
    # "prefix_hit_threshold_or_least_request|mooncake|conversation-2|${target_gpu}|25|2|80"

    # "prefix_hit_threshold_or_least_request|mooncake|conversation-2|${target_gpu}|30|2|20"
    # "prefix_hit_threshold_or_least_request|mooncake|conversation-2|${target_gpu}|30|2|40"
    # "prefix_hit_threshold_or_least_request|mooncake|conversation-2|${target_gpu}|30|2|60"
    # "prefix_hit_threshold_or_least_request|mooncake|conversation-2|${target_gpu}|30|2|80"

    #########################################################

    # "prefix_hit_threshold_or_least_request|mooncake|toolagent-2|${target_gpu}|10|2|20"
    # "prefix_hit_threshold_or_least_request|mooncake|toolagent-2|${target_gpu}|10|2|40"
    # "prefix_hit_threshold_or_least_request|mooncake|toolagent-2|${target_gpu}|10|2|60"
    # "prefix_hit_threshold_or_least_request|mooncake|toolagent-2|${target_gpu}|10|2|80"

    # "prefix_hit_threshold_or_least_request|mooncake|toolagent-2|${target_gpu}|15|2|20"
    # "prefix_hit_threshold_or_least_request|mooncake|toolagent-2|${target_gpu}|15|2|40"
    # "prefix_hit_threshold_or_least_request|mooncake|toolagent-2|${target_gpu}|15|2|60"
    # "prefix_hit_threshold_or_least_request|mooncake|toolagent-2|${target_gpu}|15|2|80"

    # "prefix_hit_threshold_or_least_request|mooncake|toolagent-2|${target_gpu}|20|2|20"
    # "prefix_hit_threshold_or_least_request|mooncake|toolagent-2|${target_gpu}|20|2|40"
    # "prefix_hit_threshold_or_least_request|mooncake|toolagent-2|${target_gpu}|20|2|60"
    # "prefix_hit_threshold_or_least_request|mooncake|toolagent-2|${target_gpu}|20|2|80"

    # "prefix_hit_threshold_or_least_request|mooncake|toolagent-2|${target_gpu}|25|2|20"
    # "prefix_hit_threshold_or_least_request|mooncake|toolagent-2|${target_gpu}|25|2|40"
    # "prefix_hit_threshold_or_least_request|mooncake|toolagent-2|${target_gpu}|25|2|60"
    # "prefix_hit_threshold_or_least_request|mooncake|toolagent-2|${target_gpu}|25|2|80"

    # "prefix_hit_threshold_or_least_request|mooncake|toolagent-2|${target_gpu}|30|2|20"
    # "prefix_hit_threshold_or_least_request|mooncake|toolagent-2|${target_gpu}|30|2|40"
    # "prefix_hit_threshold_or_least_request|mooncake|toolagent-2|${target_gpu}|30|2|60"
    # "prefix_hit_threshold_or_least_request|mooncake|toolagent-2|${target_gpu}|30|2|80"




    # "prefix_hit_threshold_or_least_request|mooncake|toolagent-2|${target_gpu}|11|1|20"
    # "prefix_hit_threshold_or_least_request|mooncake|toolagent-2|${target_gpu}|11|1|40"
    # "prefix_hit_threshold_or_least_request|mooncake|toolagent-2|${target_gpu}|11|1|60"
    # "prefix_hit_threshold_or_least_request|mooncake|toolagent-2|${target_gpu}|11|1|80"

    # "prefix_hit_threshold_or_least_request|mooncake|conversation-2|${target_gpu}|11|3|20"
    # "prefix_hit_threshold_or_least_request|mooncake|conversation-2|${target_gpu}|11|3|40"
    # "prefix_hit_threshold_or_least_request|mooncake|conversation-2|${target_gpu}|11|3|60"
    # "prefix_hit_threshold_or_least_request|mooncake|conversation-2|${target_gpu}|11|3|80"

    #########################################################
    #########################################################
    #########################################################
    #########################################################
    #########################################################


    # ##############################
    # ## Mooncake conversation realistic: 3658 ##
    # ##############################

    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_2|mooncake|conversation_realistic|${target_gpu}|-1|1"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear_toolagent_2|mooncake|conversation_realistic|${target_gpu}|-1|2"
    

    # toolagent_realistic_tokenized: No No
    # toolagent_realistic_tokenized_rpsscale_1: No No


    ## conversation_realistic_tokenized: Yes
    # "prefix_cache_1|mooncake|conversation_realistic_tokenized|${target_gpu}|-1|1"
    # "least_request|mooncake|conversation_realistic_tokenized|${target_gpu}|-1|1"

    ## synthetic_realistic_tokenized: Yes
    # "prefix_cache_1|mooncake|synthetic_realistic_tokenized|${target_gpu}|-1|1"
    # "least_request|mooncake|synthetic_realistic_tokenized|${target_gpu}|-1|1"

    ## synthetic_realistic_tokenized_rpsscale_1: ? ?
    # "prefix_cache_1|mooncake|synthetic_realistic_tokenized_rpsscale_1|${target_gpu}|-1|1"
    # "prefix_cache_2|mooncake|synthetic_realistic_tokenized_rpsscale_1|${target_gpu}|-1|1"
    # "least_request|mooncake|synthetic_realistic_tokenized_rpsscale_1|${target_gpu}|-1|1"


    ## conversation_realistic_tokenized_rpsscale_1: ? ?
    # "prefix_cache_1|mooncake|conversation_realistic_tokenized_rpsscale_1|${target_gpu}|-1|1"
    # "prefix_cache_2|mooncake|conversation_realistic_tokenized_rpsscale_1|${target_gpu}|-1|1"
    # "least_request|mooncake|conversation_realistic_tokenized_rpsscale_1|${target_gpu}|-1|1"

    
    #########################################################
    ## SharingRatio71%, total number of requests: 2000 ##
    #########################################################

    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|SharingRatio71%|${target_gpu}|2|2"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|SharingRatio71%|${target_gpu}|2|2"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|SharingRatio71%|${target_gpu}|4|6"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|SharingRatio71%|${target_gpu}|4|2"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|SharingRatio71%|${target_gpu}|6|6"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|SharingRatio71%|${target_gpu}|6|2"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|SharingRatio71%|${target_gpu}|8|2"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|SharingRatio71%|${target_gpu}|8|2"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|SharingRatio71%|${target_gpu}|10|2"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|SharingRatio71%|${target_gpu}|10|2"
    

    # "prefix_cache_1|gangmuk-prefix|SharingRatio71%|${target_gpu}|4|1"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio71%|${target_gpu}|6|1"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio71%|${target_gpu}|8|1"
    
    # "prefix_cache_1|gangmuk-prefix|SharingRatio71%|${target_gpu}|2|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio71%|${target_gpu}|4|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio71%|${target_gpu}|6|2"
    # "prefix_cache_2|gangmuk-prefix|SharingRatio71%|${target_gpu}|4|2"
    # "prefix_cache_2|gangmuk-prefix|SharingRatio71%|${target_gpu}|6|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio71%|${target_gpu}|8|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio71%|${target_gpu}|10|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio71%|${target_gpu}|15|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio71%|${target_gpu}|20|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio71%|${target_gpu}|30|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio71%|${target_gpu}|40|2"


    # "least_request|gangmuk-prefix|SharingRatio71%|${target_gpu}|2|2"
    # "least_request|gangmuk-prefix|SharingRatio71%|${target_gpu}|4|2"
    # "least_request|gangmuk-prefix|SharingRatio71%|${target_gpu}|6|2"
    # "least_request|gangmuk-prefix|SharingRatio71%|${target_gpu}|8|2"
    # "least_request|gangmuk-prefix|SharingRatio71%|${target_gpu}|10|2"
    # "least_request|gangmuk-prefix|SharingRatio71%|${target_gpu}|15|2"
    # "least_request|gangmuk-prefix|SharingRatio71%|${target_gpu}|20|2"
    # "least_request|gangmuk-prefix|SharingRatio71%|${target_gpu}|30|2"
    # "least_request|gangmuk-prefix|SharingRatio71%|${target_gpu}|40|2"


    #########################################################
    ## 47%, total number of requests: 2000 ##
    #########################################################

    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|SharingRatio47%|${target_gpu}|2|2"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|SharingRatio47%|${target_gpu}|2|2"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|SharingRatio47%|${target_gpu}|4|6"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|SharingRatio47%|${target_gpu}|4|2"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|SharingRatio47%|${target_gpu}|6|6"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|SharingRatio47%|${target_gpu}|6|2"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|SharingRatio47%|${target_gpu}|8|2"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|SharingRatio47%|${target_gpu}|8|2"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|SharingRatio47%|${target_gpu}|10|2"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|SharingRatio47%|${target_gpu}|10|2"

    # "prefix_cache_1|gangmuk-prefix|SharingRatio47%|${target_gpu}|2|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio47%|${target_gpu}|4|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio47%|${target_gpu}|6|2"
    # "prefix_cache_2|gangmuk-prefix|SharingRatio47%|${target_gpu}|4|2"
    # "prefix_cache_2|gangmuk-prefix|SharingRatio47%|${target_gpu}|6|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio47%|${target_gpu}|8|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio47%|${target_gpu}|10|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio47%|${target_gpu}|15|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio47%|${target_gpu}|20|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio47%|${target_gpu}|30|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio47%|${target_gpu}|40|2"

    # "least_request|gangmuk-prefix|SharingRatio47%|${target_gpu}|2|2"
    # "least_request|gangmuk-prefix|SharingRatio47%|${target_gpu}|4|2"
    # "least_request|gangmuk-prefix|SharingRatio47%|${target_gpu}|6|2"
    # "least_request|gangmuk-prefix|SharingRatio47%|${target_gpu}|8|2"
    # "least_request|gangmuk-prefix|SharingRatio47%|${target_gpu}|10|2"
    # "least_request|gangmuk-prefix|SharingRatio47%|${target_gpu}|15|2"
    # "least_request|gangmuk-prefix|SharingRatio47%|${target_gpu}|20|2"
    # "least_request|gangmuk-prefix|SharingRatio47%|${target_gpu}|30|2"
    # "least_request|gangmuk-prefix|SharingRatio47%|${target_gpu}|40|2"


    # # # #########################################################
    # # # ## SharingRatio28%, total number of requests: 2000 ##
    # # # #########################################################

    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|SharingRatio28%|${target_gpu}|2|2"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|SharingRatio28%|${target_gpu}|2|2"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|SharingRatio28%|${target_gpu}|4|6"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|SharingRatio28%|${target_gpu}|4|2"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|SharingRatio28%|${target_gpu}|6|6"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|SharingRatio28%|${target_gpu}|6|2"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|SharingRatio28%|${target_gpu}|8|2"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|SharingRatio28%|${target_gpu}|8|2"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|SharingRatio28%|${target_gpu}|10|2"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|SharingRatio28%|${target_gpu}|10|2"

    # "prefix_cache_1|gangmuk-prefix|SharingRatio28%|${target_gpu}|2|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio28%|${target_gpu}|4|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio28%|${target_gpu}|6|2"
    # "prefix_cache_2|gangmuk-prefix|SharingRatio28%|${target_gpu}|4|2"
    # "prefix_cache_2|gangmuk-prefix|SharingRatio28%|${target_gpu}|6|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio28%|${target_gpu}|8|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio28%|${target_gpu}|10|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio28%|${target_gpu}|15|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio28%|${target_gpu}|20|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio28%|${target_gpu}|30|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio28%|${target_gpu}|40|2"

    # "least_request|gangmuk-prefix|SharingRatio28%|${target_gpu}|2|2"
    # "least_request|gangmuk-prefix|SharingRatio28%|${target_gpu}|4|2"
    # "least_request|gangmuk-prefix|SharingRatio28%|${target_gpu}|6|2"
    # "least_request|gangmuk-prefix|SharingRatio28%|${target_gpu}|8|2"
    # "least_request|gangmuk-prefix|SharingRatio28%|${target_gpu}|10|2"
    # "least_request|gangmuk-prefix|SharingRatio28%|${target_gpu}|15|2"
    # "least_request|gangmuk-prefix|SharingRatio28%|${target_gpu}|20|2"
    # "least_request|gangmuk-prefix|SharingRatio28%|${target_gpu}|30|2"
    # "least_request|gangmuk-prefix|SharingRatio28%|${target_gpu}|40|2"


    # # # #########################################################
    # # # ## SharingRatio9%, total number of requests: 2000 ##
    # # # #########################################################

    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|SharingRatio9%|${target_gpu}|2|2"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|SharingRatio9%|${target_gpu}|2|2"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|SharingRatio9%|${target_gpu}|4|6"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|SharingRatio9%|${target_gpu}|4|2"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|SharingRatio9%|${target_gpu}|6|6"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|SharingRatio9%|${target_gpu}|6|2"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|SharingRatio9%|${target_gpu}|8|2"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|SharingRatio9%|${target_gpu}|8|2"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|SharingRatio9%|${target_gpu}|10|2"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|SharingRatio9%|${target_gpu}|10|2"

    # "prefix_cache_1|gangmuk-prefix|SharingRatio9%|${target_gpu}|2|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio9%|${target_gpu}|4|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio9%|${target_gpu}|6|2"
    # "prefix_cache_2|gangmuk-prefix|SharingRatio9%|${target_gpu}|4|2"
    # "prefix_cache_2|gangmuk-prefix|SharingRatio9%|${target_gpu}|6|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio9%|${target_gpu}|8|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio9%|${target_gpu}|10|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio9%|${target_gpu}|15|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio9%|${target_gpu}|20|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio9%|${target_gpu}|30|2"
    # "prefix_cache_1|gangmuk-prefix|SharingRatio9%|${target_gpu}|40|2"

    # "least_request|gangmuk-prefix|SharingRatio9%|${target_gpu}|2|2"
    # "least_request|gangmuk-prefix|SharingRatio9%|${target_gpu}|4|2"
    # "least_request|gangmuk-prefix|SharingRatio9%|${target_gpu}|6|6"
    #   "least_request|gangmuk-prefix|SharingRatio9%|${target_gpu}|8|2"
    # "least_request|gangmuk-prefix|SharingRatio9%|${target_gpu}|10|2"
    # "least_request|gangmuk-prefix|SharingRatio9%|${target_gpu}|15|2"
    # "least_request|gangmuk-prefix|SharingRatio9%|${target_gpu}|20|2"
    # "least_request|gangmuk-prefix|SharingRatio9%|${target_gpu}|30|2"
    # "least_request|gangmuk-prefix|SharingRatio9%|${target_gpu}|40|2"

    # # # ##################################################################
    # # # ## MixedSharingRatio10_30_50_70, total number of requests: 4000 ##
    # # # ##################################################################

    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|2|2"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|2|2"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|4|2"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|4|2"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|6|2"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|6|2"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|8|2"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|8|2"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|10|2"
    # "contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|10|2"

    # "prefix_cache_1|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|2|2"
    # "prefix_cache_1|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|4|2"
    # "prefix_cache_1|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|6|2"
    # "prefix_cache_1|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|8|2"
    # "prefix_cache_2|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|4|2"
    # "prefix_cache_2|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|6|2"
    # "prefix_cache_2|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|8|2"
    # "prefix_cache_1|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|10|2"
    # # "prefix_cache_1|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|15|2"
    # # "prefix_cache_1|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|20|2"
    # # "prefix_cache_1|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|30|2"
    # # "prefix_cache_1|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|40|2"

    # "least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|2|2"
    # "least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|4|2"
    # "least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|6|2"
    # "least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|8|2"
    # "least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|10|2"
    # # "least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|15|2"
    # # "least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|20|2"
    # # "least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|30|2"
    # # "least_request|gangmuk-prefix|MixedSharingRatio10_30_50_70%|${target_gpu}|40|2"
    ###################################################################################################################


    #################################
    ## Mooncake conversation: 2713 ##
    #################################

    ##############################################################################
    ##############################################################################

    ## conversation-2-extended-ver1
    # "prefix_cache_1|mooncake|conversation-2-extended-ver1|${target_gpu}|11|1"
    # "least_request|mooncake|conversation-2-extended-ver1|${target_gpu}|11|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_random-onlinelearning_1|mooncake|conversation-2-extended-ver1|${target_gpu}|11|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_2-onlinelearning_1|mooncake|conversation-2-extended-ver1|${target_gpu}|11|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_2-onlinelearning_1|mooncake|conversation-2-extended-ver1|${target_gpu}|11|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_2-onlinelearning_0|mooncake|conversation-2-extended-ver1|${target_gpu}|11|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_2-onlinelearning_0|mooncake|conversation-2-extended-ver1|${target_gpu}|11|1"

    ## conversation-2-extended-ver1
    # "prefix_cache_1|mooncake|conversation-2-extended-ver1|${target_gpu}|10|1"
    # "least_request|mooncake|conversation-2-extended-ver1|${target_gpu}|10|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_random-onlinelearning_1|mooncake|conversation-2-extended-ver1|${target_gpu}|10|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_2-onlinelearning_1|mooncake|conversation-2-extended-ver1|${target_gpu}|10|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_2-onlinelearning_1|mooncake|conversation-2-extended-ver1|${target_gpu}|10|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_2-onlinelearning_0|mooncake|conversation-2-extended-ver1|${target_gpu}|10|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_2-onlinelearning_0|mooncake|conversation-2-extended-ver1|${target_gpu}|10|1"

    ##############################################################################
    ##############################################################################
    ## toolagent-2-extended-ver1
    # "prefix_cache_1|mooncake|toolagent-2-extended-ver1|${target_gpu}|12|1"
    # "least_request|mooncake|toolagent-2-extended-ver1|${target_gpu}|12|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_random-onlinelearning_1|mooncake|toolagent-2-extended-ver1|${target_gpu}|12|1"
    "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_2-onlinelearning_1|mooncake|toolagent-2-extended-ver1|${target_gpu}|12|1"
    "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_2-onlinelearning_1|mooncake|toolagent-2-extended-ver1|${target_gpu}|12|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_2-onlinelearning_0|mooncake|toolagent-2-extended-ver1|${target_gpu}|12|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_2-onlinelearning_0|mooncake|toolagent-2-extended-ver1|${target_gpu}|12|1"

    ## toolagent-2-extended-ver1
    # "prefix_cache_1|mooncake|toolagent-2-extended-ver1|${target_gpu}|11|1"
    # "least_request|mooncake|toolagent-2-extended-ver1|${target_gpu}|11|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_random-onlinelearning_1|mooncake|toolagent-2-extended-ver1|${target_gpu}|11|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_2-onlinelearning_1|mooncake|toolagent-2-extended-ver1|${target_gpu}|11|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_2-onlinelearning_1|mooncake|toolagent-2-extended-ver1|${target_gpu}|11|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_2-onlinelearning_0|mooncake|toolagent-2-extended-ver1|${target_gpu}|11|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_2-onlinelearning_0|mooncake|toolagent-2-extended-ver1|${target_gpu}|11|1"

    ## toolagent-2-extended-ver1
    # "prefix_cache_1|mooncake|toolagent-2-extended-ver1|${target_gpu}|10|1"
    # "least_request|mooncake|toolagent-2-extended-ver1|${target_gpu}|10|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_random-onlinelearning_1|mooncake|toolagent-2-extended-ver1|${target_gpu}|10|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_2-onlinelearning_1|mooncake|toolagent-2-extended-ver1|${target_gpu}|10|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_2-onlinelearning_1|mooncake|toolagent-2-extended-ver1|${target_gpu}|10|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_2-onlinelearning_0|mooncake|toolagent-2-extended-ver1|${target_gpu}|10|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_2-onlinelearning_0|mooncake|toolagent-2-extended-ver1|${target_gpu}|10|1"

    ## toolagent-2-extended-ver1
    # "prefix_cache_1|mooncake|toolagent-2-extended-ver1|${target_gpu}|9|1"
    # "least_request|mooncake|toolagent-2-extended-ver1|${target_gpu}|9|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_random-onlinelearning_1|mooncake|toolagent-2-extended-ver1|${target_gpu}|9|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_2-onlinelearning_1|mooncake|toolagent-2-extended-ver1|${target_gpu}|9|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_2-onlinelearning_1|mooncake|toolagent-2-extended-ver1|${target_gpu}|9|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_2-onlinelearning_0|mooncake|toolagent-2-extended-ver1|${target_gpu}|9|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_2-onlinelearning_0|mooncake|toolagent-2-extended-ver1|${target_gpu}|9|1"

    ##############################################################################
    ##############################################################################
    ## synthetic-2-extended-ver2
    # "prefix_cache_1|mooncake|synthetic-2-extended-ver2|${target_gpu}|-1|1"
    # "least_request|mooncake|synthetic-2-extended-ver2|${target_gpu}|-1|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_random-onlinelearning_1|mooncake|synthetic-2-extended-ver2|${target_gpu}|-1|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_2-onlinelearning_1|mooncake|synthetic-2-extended-ver2|${target_gpu}|-1|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_2-onlinelearning_1|mooncake|synthetic-2-extended-ver2|${target_gpu}|-1|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_2-onlinelearning_0|mooncake|synthetic-2-extended-ver2|${target_gpu}|-1|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_2-onlinelearning_0|mooncake|synthetic-2-extended-ver2|${target_gpu}|-1|1"

    # "prefix_cache_1|mooncake|synthetic-2-extended-ver2|${target_gpu}|9|1"
    # "least_request|mooncake|synthetic-2-extended-ver2|${target_gpu}|9|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_random-onlinelearning_1|mooncake|synthetic-2-extended-ver2|${target_gpu}|9|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_2-onlinelearning_1|mooncake|synthetic-2-extended-ver2|${target_gpu}|9|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_2-onlinelearning_1|mooncake|synthetic-2-extended-ver2|${target_gpu}|9|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_2-onlinelearning_0|mooncake|synthetic-2-extended-ver2|${target_gpu}|9|1"
    # "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_2-onlinelearning_0|mooncake|synthetic-2-extended-ver2|${target_gpu}|9|1"
    
    ##############################################################################
    ##############################################################################
)

ship_model=0
ship_code=0
ship_data=0

llm_model="llama-3-8b-instruct"
# llm_model="qwen25-1-5b-instruct"
# llm_model="qwen3-4b-instruct"

POD_LABEL_SELECTOR="model.aibrix.ai/name=${llm_model}"

override_workload_output_length=0
force_exact_output_tokens=1 # must be always 1
max_tokens=1
max_tokens_std=0

if [ "${override_workload_output_length}" == "1" ]; then
    output_wrk_name="maxTokens_${max_tokens}-maxTokensStd_${max_tokens_std}"
else
    output_wrk_name="use_given_output_length"
fi

# ENABLE_ONLINE_LEARNING=1
RETRAIN_AT_STARTUP=1
ONLINE_TRAIN_FROM_SCRATCH=1
CB_RESET_LR_PER_ROUND=1
RECENCY_DECAY_WEIGHT_FACTOR=1.0

NORMALIZATION_MODE=zscore # zscore, fixed_range

MAX_TOTAL_DATA=10000 # 50000 1000000
FIFO_SIZE=5000
REPLAY_SIZE=5000
OLD_TRANSFER_SAMPLES_PER_WINDOW=1000
MIN_NUM_TRAINING_DATA=5000
MIN_NUM_UPDATE_DATA=1000

shuffle_requests_between_iterations=1

ENABLE_FALLBACK=0

INCLUDE_GPU_FEATURES=0
LOAD_PRETRAINED_MODEL=1
EXPLORATION_ENABLED=1
EXPLORATION_RATE=0.10
prompt_type="chat" # chat, token-ids
token_counting_mode="tiktoken" # tiktoken, llama3, word, char (1 token≈4 chars)
ENABLE_FLUSH=1
FLUSH_PERIOD=10
MIN_NUM_LOG_MESSAGES_TO_FLUSH=100

# gateway hash prefix indexer configuration
AIBRIX_PREFIX_CACHE_EVICTION_DURATION_MINS=20 # default: 20
AIBRIX_PREFIX_CACHE_BLOCK_NUMBER=400000
AIBRIX_PREFIX_CACHE_BLOCK_SIZE=4
AIBRIX_PREFIX_CACHE_EVICTION_INTERNAL_IN_SEC=1

# vLLM configuration (for the model deployment)
ENABLE_QUANTIZATION=0  # 0=disable, 1=enable
QUANTIZATION_METHOD="bitsandbytes"  # quantization method (only used if ENABLE_QUANTIZATION=1)
ENABLE_CHUNKED_PREFILL=1  # 0=disable, 1=enable
ENABLE_PREFIX_CACHING=1  # 0=disable, 1=enable

# vllm_config="quantization_${ENABLE_QUANTIZATION}_quantization_method_${QUANTIZATION_METHOD}_chunked_prefill_${ENABLE_CHUNKED_PREFILL}_prefix_caching_${ENABLE_PREFIX_CACHING}"
if [ "${ENABLE_QUANTIZATION}" == "1" ]; then
    vllm_config="with_${QUANTIZATION_METHOD}"
else
    vllm_config="without_${QUANTIZATION_METHOD}"
fi

# Configuration for the client
api_key="sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi"

# Fetch IP address with retry
ipaddr=""
for attempt in $(seq 1 5); do
    ipaddr=$(kubectl get svc -n envoy-gateway-system envoy-aibrix-system-aibrix-eg-903790dc -o jsonpath='{.spec.clusterIP}' 2>/dev/null) && break
    echo "⚠️  Failed to get service IP (attempt $attempt/5). Retrying in 5s..."
    sleep 5
done
port=80
echo "ipaddr of aibrix-system-aibrix-eg-903790dc svc: ${ipaddr}, port: ${port}"
if [ -z "${ipaddr}" ]; then
    echo "Error: ipaddr is empty"
    exit 1
fi

num_experiments=${#experiment_configs[@]}
echo "========================================="
echo "Will run ${num_experiments} experiment(s):"
for i in $(seq 0 $((num_experiments-1))); do
    IFS='|' read -r routing workload gpu rps episodes <<< "${experiment_configs[$i]}"
    echo "  $((i+1)). routing=${routing}, workload=${workload}, gpu=${gpu}, rps=${rps}, episodes=${episodes}"
done
echo "========================================="

for experiment_idx in $(seq 0 $((num_experiments-1))); do
    experiment_start_time=$(date +%s)


    # Parse experiment config with optional 7th field (prefix_hit_threshold)
    IFS='|' read -r routing_policy workload_category workload_name target_gpu rps total_num_episodes prefix_hit_threshold <<< "${experiment_configs[$experiment_idx]}"

    # if the routing policy contains "onlinelearning_1" and routing policy contains "contextual_bandit", then ENABLE_ONLINE_LEARNING=1
    # if the routing policy contains "onlinelearning_0", then ENABLE_ONLINE_LEARNING=0
    if [[ "${routing_policy}" == *"onlinelearning_1"* ]] && [[ "${routing_policy}" == *"contextual_bandit"* ]]; then
        ENABLE_ONLINE_LEARNING=1
        # remove "-onlinelearning_1" from routing policy
        routing_policy="${routing_policy%-onlinelearning_1}"
    elif [[ "${routing_policy}" == *"onlinelearning_0"* ]]; then
        ENABLE_ONLINE_LEARNING=0
        # remove "-onlinelearning_0" from routing policy
        routing_policy="${routing_policy%-onlinelearning_0}"
    else
        ENABLE_ONLINE_LEARNING=0
    fi
    echo "ENABLE_ONLINE_LEARNING: ${ENABLE_ONLINE_LEARNING}"
    if [ "${ENABLE_ONLINE_LEARNING}" == "" ]; then
        echo "Error: ENABLE_ONLINE_LEARNING is empty"
        echo "Exiting... 1"
        exit 1
    fi

    # Set PREFIX_HIT_THRESHOLD: use provided value if routing policy matches and field exists, otherwise default to 50
    if [ "${routing_policy}" == "prefix_hit_threshold_or_least_request" ] && [ -n "${prefix_hit_threshold}" ]; then
        PREFIX_HIT_THRESHOLD="${prefix_hit_threshold}"
    else
        PREFIX_HIT_THRESHOLD=50
    fi

    if [ "${routing_policy}" == "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_random" ]; then
        LOAD_PRETRAINED_MODEL=0
    else
        LOAD_PRETRAINED_MODEL=1
    fi
    echo "LOAD_PRETRAINED_MODEL: ${LOAD_PRETRAINED_MODEL}"

    if [ "${llm_model}" == "llama-3-8b-instruct" ]; then
        if [[ "${workload_name}" == *"realistic"* ]]; then
            max_input_tokens=25000
            input_tokens_std=0
            token_counting_mode="tiktoken"
        elif [[ "${workload_name}" == *"toolagent-2"* || "${workload_name}" == *"conversation-2"* || "${workload_name}" == *"synthetic-2"* ]]; then
            max_input_tokens=25000
            input_tokens_std=100
            token_counting_mode="tiktoken"
        else
            max_input_tokens=10000
            input_tokens_std=100
            token_counting_mode="word"
        fi
    elif [ "${llm_model}" == "qwen25-1-5b-instruct" ] || [ "${llm_model}" == "qwen3-4b-instruct" ]; then
        max_input_tokens=30000
        input_tokens_std=100
    else
        echo "Error: Unknown LLM model: ${llm_model}"
        echo "Exiting... 1"
        exit 1
    fi

    echo ""
    echo "========================================="
    echo "Starting Experiment $((experiment_idx+1))/${num_experiments}"
    echo "========================================="
    echo "  Routing Policy:    ${routing_policy}"
    echo "  Workload Category: ${workload_category}"
    echo "  Workload:          ${workload_name}"
    echo "  Target GPU:        ${target_gpu}"
    echo "  RPS:               ${rps}"
    echo "  Total Episodes:    ${total_num_episodes}"
    echo "========================================="
    delimiter="+"
    if [ "${routing_policy}" == "preble" ]; then
        config="preble${delimiter}${routing_policy}"
    else
        config="rl-online-router${delimiter}${routing_policy}"
    fi
    routing="${config%%${delimiter}*}"
    subAlgorithm="${config#*${delimiter}}"
    if [[ "${routing_policy}" == *"contextual_bandit"* ]]; then
        source_final_model_dir="../workload-and-experiment_results/${target_gpu}/${output_wrk_name}/${workload_category}/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear"
        echo "✓ Using contextual bandit model: ${source_final_model_dir}"
    elif [[ "${routing_policy}" == *"latency_predictor"* ]]; then
        source_final_model_dir="../workload-and-experiment_results/${target_gpu}/${output_wrk_name}/${workload_category}/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear"
        echo "✓ Using contextual bandit model: ${source_final_model_dir}"
    else
        source_final_model_dir="../workload-and-experiment_results/${target_gpu}/maxTokens_1-maxTokensStd_0/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear"
    fi

    if [ "${ship_model}" == "1" ]; then
        if [ ! -d "${source_final_model_dir}" ]; then
        echo "Error: Final model directory does not exist: ${source_final_model_dir}"
            echo "Exiting... 4"
            exit 1
        fi

        if [ ! -f "${source_final_model_dir}/model_config.json" ]; then
            echo "Error: model_config.json does not exist: ${source_final_model_dir}/model_config.json"
            echo "Exiting... 5"
            exit 1
        fi

        # Check for model weights based on routing policy
        if [[ "${routing_policy}" == *"contextual_bandit"* ]]; then
            if [ ! -f "${source_final_model_dir}/reward_net.pth" ]; then
                echo "Error: reward_net.pth does not exist: ${source_final_model_dir}/reward_net.pth"
                echo "Exiting... 6"
                exit 1
            fi
        elif [[ "${routing_policy}" == *"latency_predictor"* ]]; then
            if [ ! -f "${source_final_model_dir}/latency_predictor.pth" ]; then
                echo "Error: latency_predictor.pth does not exist: ${source_final_model_dir}/latency_predictor.pth"
                echo "Exiting... 6"
                exit 1
            fi
        fi

        if [ ! -f "${source_final_model_dir}/feature_normalization_statistics.csv" ]; then
            echo "Error: feature_normalization_statistics.csv does not exist: ${source_final_model_dir}/feature_normalization_statistics.csv"
            echo "Exiting... 7"
            exit 1
        fi
        if [ ! -f "${source_final_model_dir}/feature_distribution_statistics.csv" ]; then
            echo "Error: feature_distribution_statistics.csv does not exist: ${source_final_model_dir}/feature_distribution_statistics.csv"
            echo "Exiting... 8"
            exit 1
        fi
    fi


    if [ "${prompt_type}" == "chat" ]; then
        workload_file_name="workload.jsonl"
        workload_path_in_pod="/app/workload/${workload_category}/${workload_name}/${workload_file_name}"
        workload_path_in_host="../workload-and-experiment_results/${workload_category}/${workload_name}/${workload_file_name}"
    elif [ "${prompt_type}" == "token-ids" ]; then
        workload_file_name="workload_token.jsonl"
        workload_path_in_pod="/app/workload/${workload_category}/${workload_name}/${workload_file_name}"
        workload_path_in_host="../workload-and-experiment_results/${workload_category}/${workload_name}/${workload_file_name}"
    else
        echo "Error: Unknown prompt type: ${prompt_type}"
        echo "Exiting... 8"
        exit 1
    fi
    # if [ ! -f "${workload_path_in_host}" ]; then
    #     echo "Error: Workload file does not exist: ${workload_path_in_host}"
    #     echo "Exiting... 9"
    #     exit 1
    # fi
    output_dir="/app/output/${workload_category}/${workload_name}/${subAlgorithm}-$(date +%Y%m%d_%H%M%S)"
    output_jsonl_path="${output_dir}/output.jsonl"

    # Create local experiment result output directory
    timestamp=$(date +%Y%m%d_%H%M%S)
    if [ "${routing_policy}" == "prefix_hit_threshold_or_least_request" ]; then
        experiment_result_output_dir="../workload-and-experiment_results/${target_gpu}/${llm_model}/${output_wrk_name}/${workload_category}/${workload_name}/rps${rps}-${workload_mode}/${vllm_config}/${subAlgorithm}_threshold_${PREFIX_HIT_THRESHOLD}-iter${total_num_episodes}"
    else
        experiment_result_output_dir="../workload-and-experiment_results/${target_gpu}/${llm_model}/${output_wrk_name}/${workload_category}/${workload_name}/rps${rps}-${workload_mode}/${vllm_config}/${subAlgorithm}-iter${total_num_episodes}"
    fi
    if [[ "${routing_policy}" == *"contextual_bandit"* ]] || [[ "${routing_policy}" == *"latency_predictor"* ]]; then
        experiment_result_output_dir="${experiment_result_output_dir}-onlinelearning_${ENABLE_ONLINE_LEARNING}"
    fi
    experiment_result_output_dir="${experiment_result_output_dir}-${timestamp}"
    echo "* experiment_result_output_dir: ${experiment_result_output_dir}"
    if [ ! -d "${experiment_result_output_dir}" ]; then
        mkdir -p "${experiment_result_output_dir}"
    fi

    echo "========================================="
    echo "* workload_category: ${workload_category}"
    echo "* workload_name: ${workload_name}"
    echo "* target_gpu: ${target_gpu}"
    echo "* rps: ${rps}"
    echo "* total_num_episodes: ${total_num_episodes}"
    echo "* experiment_result_output_dir: ${experiment_result_output_dir}"
    echo "========================================="

    echo "Starting to update k8s env for routing-agent-service"
    retry_command python3 update_k8s_env.py \
        --deployment routing-agent-service \
        --namespace default \
        --container routing-agent \
        --env EXPLORATION_ENABLED=${EXPLORATION_ENABLED} \
        --env MIN_NUM_TRAINING_DATA=${MIN_NUM_TRAINING_DATA} \
        --env MIN_NUM_UPDATE_DATA=${MIN_NUM_UPDATE_DATA} \
        --env ENABLE_ONLINE_LEARNING=${ENABLE_ONLINE_LEARNING} \
        --env ONLINE_TRAIN_FROM_SCRATCH=${ONLINE_TRAIN_FROM_SCRATCH} \
        --env RETRAIN_AT_STARTUP=${RETRAIN_AT_STARTUP} \
        --env CB_RESET_LR_PER_ROUND=${CB_RESET_LR_PER_ROUND} \
        --env RECENCY_DECAY_WEIGHT_FACTOR=${RECENCY_DECAY_WEIGHT_FACTOR} \
        --env NORMALIZATION_MODE=${NORMALIZATION_MODE} \
        --env ENABLE_FALLBACK=${ENABLE_FALLBACK} \
        --env POD_LABEL_SELECTOR=${POD_LABEL_SELECTOR} \
        --env EXPLORATION_RATE=${EXPLORATION_RATE} \
        --env TARGET_GPU_MODEL=${target_gpu} \
        --env MAX_TOTAL_DATA=${MAX_TOTAL_DATA} \
        --env FIFO_SIZE=${FIFO_SIZE} \
        --env REPLAY_SIZE=${REPLAY_SIZE} \
        --env OLD_TRANSFER_SAMPLES_PER_WINDOW=${OLD_TRANSFER_SAMPLES_PER_WINDOW} \
        --env INCLUDE_GPU_FEATURES=${INCLUDE_GPU_FEATURES} \
        --env LOAD_PRETRAINED_MODEL=${LOAD_PRETRAINED_MODEL} \
        --env ROUTING_STRATEGY=${routing_policy} \
        --env OUTPUT_WRK_NAME=${output_wrk_name} \
        --env WORKLOAD_CATEGORY=${workload_category} \
        --env WORKLOAD_NAME=${workload_name} \
        --env MODEL_NAME=${llm_model}
    echo "Finished updating k8s env for routing-agent-service"
    
    echo "Starting to update k8s env for aibrix-gateway-plugins"
    retry_command python3 update_k8s_env.py \
        --deployment aibrix-gateway-plugins \
        --namespace aibrix-system \
        --container gateway-plugin \
        --env ENABLE_FLUSH=${ENABLE_FLUSH} \
        --env FLUSH_PERIOD=${FLUSH_PERIOD} \
        --env MIN_NUM_LOG_MESSAGES_TO_FLUSH=${MIN_NUM_LOG_MESSAGES_TO_FLUSH} \
        --env PREFIX_HIT_THRESHOLD=${PREFIX_HIT_THRESHOLD} \
        --env AIBRIX_PREFIX_CACHE_EVICTION_DURATION_MINS=${AIBRIX_PREFIX_CACHE_EVICTION_DURATION_MINS} \
        --env AIBRIX_PREFIX_CACHE_BLOCK_NUMBER=${AIBRIX_PREFIX_CACHE_BLOCK_NUMBER} \
        --env AIBRIX_PREFIX_CACHE_BLOCK_SIZE=${AIBRIX_PREFIX_CACHE_BLOCK_SIZE} \
        --env AIBRIX_PREFIX_CACHE_EVICTION_INTERNAL_IN_SEC=${AIBRIX_PREFIX_CACHE_EVICTION_INTERNAL_IN_SEC}

    echo "Finished updating k8s env for aibrix-gateway-plugins"

    echo "Starting to update vLLM args for ${llm_model}"
    retry_command python3 update_vllm_args.py \
        --deployment ${llm_model} \
        --namespace default \
        --container vllm-openai \
        --enable-quantization ${ENABLE_QUANTIZATION} \
        --quantization-method ${QUANTIZATION_METHOD} \
        --enable-chunked-prefill ${ENABLE_CHUNKED_PREFILL} \
        --enable-prefix-caching ${ENABLE_PREFIX_CACHING}
    echo "Finished updating vLLM args for ${llm_model}"
    sleep 2

    retry_command kubectl rollout restart deployment client-service --namespace default
    retry_command kubectl rollout restart deployment aibrix-gateway-plugins --namespace aibrix-system
    retry_command kubectl rollout restart deployment routing-agent-service --namespace default

    sleep 2
    retry_command python3 check_ready.py --deployment ${llm_model} --namespace default
    retry_command python3 check_ready.py --deployment aibrix-gateway-plugins --namespace aibrix-system
    retry_command python3 check_ready.py --deployment routing-agent-service --namespace default
    retry_command python3 check_ready.py --deployment client-service --namespace default
    sleep 2

    retry_command bash -c "kubectl get deploy ${llm_model} -o yaml > ${experiment_result_output_dir}/${llm_model}.yaml"
    retry_command bash -c "kubectl get deploy aibrix-gateway-plugins -n aibrix-system -o yaml > ${experiment_result_output_dir}/aibrix-gateway-plugins.yaml"

    ###############
    ## code ship ##
    ###############

    ship_start_time=$(date +%s)
    if [ "${ship_code}" == "1" ] || [ "${ship_model}" == "1" ]; then
        python ship_all.py --ship_code ${ship_code} --ship_model ${ship_model} --source_final_model_dir ${source_final_model_dir} --ship_data ${ship_data} --k8s_cluster ${k8s_cluster}
    fi
    
    # if [ "${routing_policy}" == "scalable_rl_agent" ]; then
    #     scalable_rl_agent_init_model_dir="../training_data/scalable_rl_agent/init_model"
    #     python kubectl_cp_from_host_to_pod.py ${scalable_rl_agent_init_model_dir} /app/final_model routing-agent-service default
    # fi

    # python kubectl_cp_from_host_to_pod.py async-client.py /app client-service default

    ship_end_time=$(date +%s)
    ship_took=$((ship_end_time - ship_start_time))
    echo "* ship_all took: ${ship_took}s"

    echo "========================================="
    echo "Running Client in K8s Pod"
    echo "========================================="
    echo "Pod:                 ${CLIENT_SERVICE_POD_NAME}"
    echo "Container:           ${CLIENT_SERVICE_CONTAINER_NAME}"
    echo "Routing Strategy:    ${routing}"
    echo "Sub-Algorithm:       ${subAlgorithm}"
    echo "Workload Category:   ${workload_category}"
    echo "Workload Name:       ${workload_name}"
    echo "Online Learning:     ${ENABLE_ONLINE_LEARNING}"
    echo "total_num_episodes:  ${total_num_episodes}"
    echo "Max Tokens:          ${max_tokens}"
    echo "Max Tokens Std:      ${max_tokens_std}"
    echo "Override Workload Output Length: ${override_workload_output_length}"
    echo "RPS:                 ${rps}"
    echo "--- vLLM Config ---"
    echo "Quantization:        ${ENABLE_QUANTIZATION} (${QUANTIZATION_METHOD})"
    echo "Chunked Prefill:     ${ENABLE_CHUNKED_PREFILL}"
    echo "Prefix Caching:      ${ENABLE_PREFIX_CACHING}"
    echo "========================================="

    # Find the actual pod name (in case of deployment with generated suffix)
    ACTUAL_POD=""
    for attempt in $(seq 1 5); do
        ACTUAL_POD=$(kubectl get pods -l app=${CLIENT_SERVICE_POD_NAME} -o jsonpath='{.items[0].metadata.name}' 2>/dev/null) && break
        echo "⚠️  Failed to get pod name (attempt $attempt/5). Retrying in 5s..."
        sleep 5
    done

    if [ -z "$ACTUAL_POD" ]; then
        echo "Error: No pod found with label app=${CLIENT_SERVICE_POD_NAME}"
        echo "Trying to use pod name directly: ${CLIENT_SERVICE_POD_NAME}"
        ACTUAL_POD=${CLIENT_SERVICE_POD_NAME}
    fi

    echo "Using pod: ${ACTUAL_POD}"

    # Wait for pod to be ready
    echo "Waiting for pod to be ready..."
    retry_command kubectl wait --for=condition=ready pod/${ACTUAL_POD} --timeout=60s || {
        echo "Error: Pod did not become ready within 60 seconds"
        kubectl describe pod ${ACTUAL_POD}
        echo "Exiting... 8"
        exit 1
    }

    # Create output directory in pod
    echo "Creating output directory in pod..."
    retry_command kubectl exec ${ACTUAL_POD} -c ${CLIENT_SERVICE_CONTAINER_NAME} -- mkdir -p ${output_dir}

    # Check if workload file exists
    echo "Checking if workload file exists..."
    retry_command kubectl exec ${ACTUAL_POD} -c ${CLIENT_SERVICE_CONTAINER_NAME} -- test -f ${workload_path_in_pod} || {
        echo "Error: Workload file ${workload_path_in_pod} not found in pod"
        echo "Available workloads:"
        kubectl exec ${ACTUAL_POD} -c ${CLIENT_SERVICE_CONTAINER_NAME} -- find /app/workload -name "*.jsonl"
        echo "Exiting... 9"
        exit 1
    }

    echo "Starting log collection..."
    
    # Follow logs continuously with reconnection on disconnect (no duplicates)
    collect_logs_continuously() {
        local namespace=$1
        local pod_pattern=$2
        local output_file=$3

        while true; do
            pod_name=$(kubectl get pods -n ${namespace} | grep ${pod_pattern} | awk '{print $1}' | head -n 1)
            if [ -n "$pod_name" ]; then
                echo "[$(date)] Starting log follow for ${pod_name}" >> ${output_file}
                # -f follows logs continuously, no --since flag to avoid duplicates
                kubectl logs -f -n ${namespace} ${pod_name} >> ${output_file} 2>&1
                echo "[$(date)] Log follow disconnected, reconnecting..." >> ${output_file}
            fi
            sleep 2  # Brief pause before reconnect attempt
        done
    }
    
    # Function to copy checkpoints periodically
    copy_checkpoints_periodically() {
        local output_dir=$1
        local interval=300  # Copy checkpoints every 5 minutes (300 seconds)
        
        # Wait a bit before first copy to let training start
        sleep 60
        
        while true; do
            checkpoint_ts=$(date +%Y%m%d_%H%M%S)
            echo "[$(date)] Copying checkpoints at ${checkpoint_ts}..."
            python kubectl_cp_from_pod_to_host.py --src /app/final_model/checkpoints --dst "${output_dir}/checkpoints_${checkpoint_ts}" --deployment routing-agent-service --namespace default 2>&1 | grep -v "tar: Removing leading"
            echo "[$(date)] Checkpoint copy completed: checkpoints_${checkpoint_ts}"
            sleep ${interval}
        done
    }
    
    # Start log collection in background
    collect_logs_continuously "aibrix-system" "aibrix-gateway-plugins" "${experiment_result_output_dir}/all-aibrix-gateway-plugins.log.txt" &
    pid_1=$!
    collect_logs_continuously "default" "routing-agent-service" "${experiment_result_output_dir}/all-routing-agent-service.log.txt" &
    pid_2=$!
    
    echo "Starting client in pod..."
    echo "Output will be saved to: ${output_dir}"

    if [ "${workload_category}" == "mooncake" ]; then
        input_token_length_scaling=2.0
    elif [ "${workload_category}" == "azure" ]; then
        input_token_length_scaling=3.0
    else
        input_token_length_scaling=1.0
    fi

    if [ "${workload_name}" == *"synthetic"* ]; then
        shuffle_requests_between_iterations=0
    else
        shuffle_requests_between_iterations=1
    fi
    echo "* shuffle_requests_between_iterations: ${shuffle_requests_between_iterations}"
    kubectl exec ${ACTUAL_POD} -c ${CLIENT_SERVICE_CONTAINER_NAME} -- \
        python3 /app/async-client.py \
            --workload_path ${workload_path_in_pod} \
            --model ${llm_model} \
            --endpoint http://${ipaddr}:${port} \
            --api_key ${api_key} \
            --output_file_path ${output_jsonl_path} \
            --routing_strategy ${routing} \
            --subAlgorithm ${subAlgorithm} \
            --max_tokens ${max_tokens} \
            --max_tokens_std ${max_tokens_std} \
            --override_workload_output_length ${override_workload_output_length} \
            --force_exact_output_tokens ${force_exact_output_tokens} \
            --output_dir ${output_dir} \
            --prompt_type ${prompt_type} \
            --token_counting_mode ${token_counting_mode} \
            --rps ${rps} \
            --poisson_arrivals \
            --shuffle_requests_between_iterations ${shuffle_requests_between_iterations} \
            --iterations ${total_num_episodes} \
            --streaming \
            --max_input_tokens ${max_input_tokens} \
            --input_tokens_std ${input_tokens_std} \
            --input_token_length_scaling ${input_token_length_scaling} \
            --output_token_length_scaling 1.0 \
            --iteration_overlap_ratio 0.0 \
            --iteration_ramp_duration 10.0 \
            --iteration_ramp_start_fraction 0.1 \
            --workload_mode ${workload_mode} \
            2>&1 | tee ${experiment_result_output_dir}/client.log.txt

    sleep 5
    # kubectl rollout restart deployment ${llm_model}

    # Process logs
    cat ${experiment_result_output_dir}/all-aibrix-gateway-plugins.log.txt | grep "**@latency_metrics" | grep -v "infer:" > ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins.log.csv
    echo "* processed gateway log: ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins.log.csv-processed.csv"
    echo "* all gateway log: ${experiment_result_output_dir}/all-aibrix-gateway-plugins.log.txt"
    echo "* filtered gateway log: ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins.log.csv"
    echo "* routing agent log: ${experiment_result_output_dir}/all-routing-agent-service.log.txt"
    echo "* client log: ${experiment_result_output_dir}/client.log.txt"
    if [ "${subAlgorithm}" == "scalable_rl_agent" ]; then
        echo "* checkpoints (periodic snapshots): ${experiment_result_output_dir}/checkpoints_*/"
        echo "* final checkpoint: ${experiment_result_output_dir}/checkpoints_${checkpoint_ts}/"
    fi

    python plot_latency_timeseries.py ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins.log.csv
    python3 plot_latency_analysis_with_client_log.py ${experiment_result_output_dir}/client.log.txt

    
    # Copy final model
    # if [ "${routing_policy}" == "contextual_bandit" ] || [ "${routing_policy}" == "latency_predictor" ]; then
    # if routing_policy has contextual_bandit or latency_predictor in its name, then copy the final model
    if [[ "${routing_policy}" == *"contextual_bandit"* ]] || [[ "${routing_policy}" == *"latency_predictor"* ]]; then
        kubectl_cp_start_time=$(date +%s)
        echo "Copying final_model from pod..."
        model_dir_in_pod="/app/${target_gpu}/${llm_model}/${output_wrk_name}/${workload_category}/final_model-${routing_policy}"
        echo "* experiment_result_output_dir: ${experiment_result_output_dir}"
        echo "* model_dir_in_pod: ${model_dir_in_pod}"
        python kubectl_cp_from_pod_to_host.py --src ${model_dir_in_pod} --dst "${experiment_result_output_dir}/final_model/${target_gpu}" --deployment routing-agent-service --namespace default --skip-files "tensor_dataset.pt" "*.pkl" "data.csv" "data-processed.csv" "data-processed-sampled.csv" "data-processed_summary.json" "data_processor_command.txt" "full_path.txt" "python_command.txt" "optimizer*" "data_processor.log.txt" "dataset_analyzer.log.txt" 
        
        kubectl_cp_end_time=$(date +%s)
        echo "* copying final_model took: $((kubectl_cp_end_time - kubectl_cp_start_time))s"
        hyperparameters_file_path=$(find "${experiment_result_output_dir}/final_model/${target_gpu}" -name "model_config.json" | head -1)
        if [ -z "$hyperparameters_file_path" ]; then
            echo "ERROR: Could not find model_config.json in ${experiment_result_output_dir}/final_model/${target_gpu}"
            # exit 1
        fi
    else
        echo "Skipping final model copying for ${routing_policy}. It does not use any learned model."
    fi
    if [ -z "$hyperparameters_file_path" ]; then
        echo "* Using hyperparameters file: $hyperparameters_file_path"
        python ../agent_codes/data_processor.py --input_file ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins.log.csv --output_file ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins-processed.log.csv --hyperparameters_file_path "$hyperparameters_file_path"
    else
        python ../agent_codes/data_processor.py --input_file ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins.log.csv --output_file ${experiment_result_output_dir}/filtered-aibrix-gateway-plugins-processed.log.csv
    fi


    # python kubectl_cp_from_pod_to_host.py /tmp/latency_metrics.log "${experiment_result_output_dir}/latency_metrics.log.txt" gateway-plugins aibrix-system

    # # Copy checkpoints (for scalable_rl_agent)
    # if [ "${subAlgorithm}" == "scalable_rl_agent" ]; then
    #     checkpoint_ts=$(date +%Y%m%d_%H%M%S)
    #     echo "Copying scalable RL agent checkpoints..."
    #     python kubectl_cp_from_pod_to_host.py /app/final_model/checkpoints "${experiment_result_output_dir}/checkpoints_${checkpoint_ts}" routing-agent-service default || echo "⚠️  No checkpoints found (agent may not have trained enough steps)"
    # fi

    
    # Kill background processes
    kill $pid_1 2>/dev/null || true
    kill $pid_2 2>/dev/null || true
    if [ "${subAlgorithm}" == "scalable_rl_agent" ] && [ -n "${pid_checkpoint}" ]; then
        kill $pid_checkpoint 2>/dev/null || true
    fi

    
    experiment_end_time=$(date +%s)
    echo ""
    echo "========================================="
    echo "Experiment $((experiment_idx+1))/${num_experiments} completed!"
    echo "* experiment took: $((experiment_end_time - experiment_start_time))s"
    echo "* workload:            ${workload_name}"
    echo "* workload category:   ${workload_category}"
    echo "* routing strategy:    ${routing}"
    echo "* sub-algorithm:       ${subAlgorithm}"
    echo "* max tokens:          ${max_tokens}"
    echo "* max tokens std:      ${max_tokens_std}"
    echo "* override workload output length: ${override_workload_output_length}"
    echo "* rps:                 ${rps}"
    echo "* quantization:        ${ENABLE_QUANTIZATION} (${QUANTIZATION_METHOD})"
    echo "* chunked prefill:     ${ENABLE_CHUNKED_PREFILL}"
    echo "* prefix caching:      ${ENABLE_PREFIX_CACHING}"
    echo "========================================="

    # kubectl rollout restart deployment client-service
    # kubectl rollout restart deployment routing-agent-service
    # kubectl rollout restart deployment aibrix-gateway-plugins -n aibrix-system
done
