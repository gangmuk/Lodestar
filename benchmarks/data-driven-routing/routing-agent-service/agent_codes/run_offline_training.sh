#!/bin/bash

# filename: run_offline_training.sh

set -e


# workload_dataset="SharingRatio9%-p200_s1800_rps8_spp_20_ndp80-p400_s3600_rps8_spp_20_ndp80-p800_s7200_rps3_spp_20_ndp80-half"
# workload_dataset="SharingRatio28%-p600_s1400_rps8_spp_20_ndp80-p1200_s2800_rps8_spp_20_ndp80-p2400_s5600_rps3_spp_20_ndp80-half"
# workload_dataset="SharingRatio47%-p1024_s1024_rps8_spp_20_ndp80-p2048_s2048_rps8_spp_20_ndp80-p4096_s4096_rps3_spp_20_ndp80-half"
# workload_dataset="SharingRatio71%-p2048_s512_rps5_spp_10_ndp50-p4096_s1024_rps8_spp_10_ndp50-p8096_s2048_rps3_spp_10_ndp50-half"

# workload_dataset_list=(
#     # "temp"
#     "merged-data"
#     # "p4096_s1024_rps20"
#     # "SharingRatio71%"
#     # "SharingRatio47%"
#     # "SharingRatio28%"
#     # "SharingRatio9%"
# )
# csv_filename="data_replaced.csv" # "data_replaced.csv", "data.csv"


routing_policy_for_data_file_list=(
    # "prefix"
    # "rl"
    # "random"
    # "latency_predictor"
    "all"
)

lr_scheduler_type="constant" # "exponential", "constant", "gradient_adaptive"
batch_size=256
training_epochs=50
lr_scheduler_gamma=0.95
excluded_pod_features="none" # "prefill_tokens", "none", "cpu_kv_cache"
no_normalize_features="none" # "kv_hit_ratio", "none"
model_type="latency_predictor" # "contextual_bandit", "latency_predictor", "rl_agent"
latency_metric="ttft" # "ttft", "avg_tpot", "e2e_latency" (for latency_predictor)
use_sampled_data=false # true, false
analyze_behavior=false # true, false
analyze_dataset=false # true, false
reward_decay_factor=0.91
hidden_dim=64 # 64, 128, 256
ttft_slo=1000
avg_tpot_slo=50
ttft_reward_weight=2.0 # ttft_reward_weight*ttft_rewards + max(0, (1-ttft_reward_weight))*tpot_rewards
REWARD_FUNCTION="linear_simple_extended" # "linear_simple", "linear_simple_extended", "piecewise_linear_steeper_gradient", "latency_optimized"
offline_learning_rate=0.001
time_stamp=$(date +%Y%m%d_%H%M%S)

# for workload_dataset in "${workload_dataset_list[@]}"; do
for routing_policy_for_data_file in "${routing_policy_for_data_file_list[@]}"; do
    # data_file="../training_data/${workload_dataset}/${routing_policy_for_data_file}/${csv_filename}"
    # data_file="../training_data/A30-8/config_sharing30%/data.csv"
    # data_file="../training_data/A30-8/config_sharing10%/data.csv"
    # data_file="../training_data/A30-8/old-version/data.csv"

    # data_file="../workload-and-experiment_results/MixedSharingRatio10_30_50_70%/prefix_cache_1-L20-rps8--iter4-20251027_114215/filtered-aibrix-gateway-plugins.log.csv"
    # data_file="../workload-and-experiment_results/MixedSharingRatio10_30_50_70%/prefix_cache_1-L20-rps10--iter8-20251027_133521/filtered-aibrix-gateway-plugins.log.csv"
    # data_file="../workload-and-experiment_results/MixedSharingRatio10_30_50_70%/prefix_cache_1-L20-rps12--iter8-20251027_163145/filtered-aibrix-gateway-plugins.log.csv"
    # data_file="../workload-and-experiment_results/MixedSharingRatio10_30_50_70%/random-L20-rps8--iter4-20251027_121944/filtered-aibrix-gateway-plugins.log.csv"
    # data_file="../workload-and-experiment_results/MixedSharingRatio10_30_50_70%/random-L20-rps10--iter8-20251027_143423/filtered-aibrix-gateway-plugins.log.csv"

    # data_file="../training_data/L20-7/merged-data/all/data_replaced.csv"

    # data_file="../training_data/L20-7/merged-data/all-with-mixed/data_replaced.csv"

    # data_file="../workload-and-experiment_results/multiturn-chat/training_data/data_replaced.csv"

    data_file="../training_data/A30-8-backup/data.csv"

    # data_file="../workload-and-experiment_results/multiturn-chat/prefix_cache_1-L20-rps16--iter2-20251027_215827/filtered-aibrix-gateway-plugins.log.csv"


    # data_file="../workload-and-experiment_results/config_sharing10%/latency_predictor_ttft-20251025_071353-trained_on_A30-8_config_sharing10%-iter2-20251025_075801/temp/filtered-aibrix-gateway-plugins.log.csv"

    data_dir=$(dirname "${data_file}")

    if [ ! -f "${data_file}" ]; then
        echo "❌ Data file not found: ${data_file}"
        exit 1
    fi
    echo "✓ Found data file: ${data_file}"

    # data_file="../workload-and-experiment_results/SharingRatio28%/latency_predictor-trained_on_merged-data_all-20250925_190503/filtered-aibrix-gateway-plugins.log.csv"

    # Generate processed CSV filename automatically
    data_basename=$(basename -- "${data_file}")
    data_name="${data_basename%.*}"  # Remove .csv extension
    processed_csv="${data_dir}/${data_name}-processed.csv"


    echo "📊 STEP 1: Processing data to structured CSV"
    echo "==========================================="
    final_model_dir="${data_dir}/final_model"
    if [ "${model_type}" == "contextual_bandit" ]; then
        final_model_dir="${final_model_dir}-${data_name}-processed-${REWARD_FUNCTION}-lr_${offline_learning_rate}-ttft_weight_${ttft_reward_weight}-ttftslo_${ttft_slo}-avgtpotslo_${avg_tpot_slo}"
        if [ "${excluded_pod_features}" != "" ]; then
            final_model_dir="${final_model_dir}-without_${excluded_pod_features}"
        fi
        final_model_dir="${final_model_dir}-hidden_dim_${hidden_dim}"

        if [ "${lr_scheduler_type}" == "gradient_adaptive" ]; then
            final_model_dir="${final_model_dir}-lrs_grad_adapt"
        elif [ "${lr_scheduler_type}" == "exponential" ]; then
            final_model_dir="${final_model_dir}-lrs_exp"
        elif [ "${lr_scheduler_type}" == "plateau" ]; then
            final_model_dir="${final_model_dir}-lrs_plateau"
        else
            echo "❌ Unknown LR scheduler type: ${lr_scheduler_type}"
            exit 1
        fi
    fi

    final_model_dir="${final_model_dir}-${model_type}"
    
    if [ "${model_type}" == "latency_predictor" ]; then
        final_model_dir="${final_model_dir}_${latency_metric}"
    fi
    final_model_dir=${final_model_dir}-${time_stamp}
    if [ -d "${final_model_dir}" ]; then
        rm -rf "${final_model_dir}"
    fi
    echo "Final model directory: ${final_model_dir}"
    mkdir -p "${final_model_dir}"


    hyper_json="${final_model_dir}/model_config.json"
    echo "📄 STEP 0: Writing hyperparameters JSON"
    python3 write_hyperparameters.py \
    --output ${hyper_json} \
    --ttft_slo ${ttft_slo} \
    --avg_tpot_slo ${avg_tpot_slo} \
    --hidden_dim ${hidden_dim} \
    --ttft_reward_weight ${ttft_reward_weight} \
    --reward_function ${REWARD_FUNCTION} \
    --offline_learning_rate ${offline_learning_rate} \
    --excluded_pod_features ${excluded_pod_features} \
    --no_normalize_features ${no_normalize_features} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --training_epochs ${training_epochs} \
    --batch_size ${batch_size} \
    --lr_scheduler_gamma ${lr_scheduler_gamma} \
    --reward_decay_factor ${reward_decay_factor} \
    --model_type ${model_type} \
    --latency_metric ${latency_metric}
    

    python3 data_processor.py --input_file ${data_file} --output_file ${processed_csv} --hyperparameters ${hyper_json}

    if [ ! -f "${processed_csv}" ]; then
        echo "❌ Failed to create processed CSV: ${processed_csv}"
        exit 1
    fi

    if [ "${analyze_dataset}" = "true" ]; then
        python3 dataset_analyzer.py --processed_csv ${processed_csv} --reward-function ${REWARD_FUNCTION} --ttft-slo ${ttft_slo} --avg-tpot-slo ${avg_tpot_slo} --ttft-reward-weight ${ttft_reward_weight} --save-sampled-dataset
    fi

    sampled_processed_csv="${processed_csv%.*}-sampled.csv"
    echo "Sampled processed CSV: ${sampled_processed_csv}"

    if [ ${use_sampled_data} = "true" ]; then
        processed_csv="${sampled_processed_csv}"
    fi

    # Step 2: Setup model directory
    training_data_dir=$(dirname "${processed_csv}")
    training_data_filename=$(basename -- "${processed_csv}")
    training_data_filename="${training_data_filename%.*}"

    # Note: final_model_dir already computed above to host model_config.json

    echo "📁 SETTING UP MODEL DIRECTORY"
    echo "=============================="
    echo "Training data directory: ${training_data_dir}"
    echo "Final model directory: ${final_model_dir}"

    echo "✅ Using model directory: ${final_model_dir}"

    # Step 3: Run training with the new streamlined pipeline
    # Build command arguments
    analyze_flag=""
    if [ "${analyze_behavior}" = "true" ]; then
        analyze_flag="--analyze_behavior"
    fi

    python_cmd="python3 offline_routing_agent.py ${processed_csv} ${analyze_flag} --final_model_dir ${final_model_dir} --hyperparameters ${hyper_json}"

    ${python_cmd} 2>&1 | tee "${final_model_dir}/output.txt"
    echo "${python_cmd}" > "${final_model_dir}/python_command.txt"
    echo "final_model_dir: ${final_model_dir}" > ${final_model_dir}/full_path.txt
    echo "data_file: ${data_file}" >> ${final_model_dir}/full_path.txt
    echo "processed_csv: ${processed_csv}" >> ${final_model_dir}/full_path.txt
    if [ "${model_type}" == "contextual_bandit" ]; then
        python csv_training_analyzer.py ${final_model_dir}/training_metrics.csv
    fi
    echo "Model saved to: ${final_model_dir}"
done
# done