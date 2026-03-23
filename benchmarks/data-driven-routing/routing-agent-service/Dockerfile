# Alternative: Use NVIDIA CUDA runtime base + install PyTorch via pip
# This might be smaller than pytorch/pytorch base image (~5-6GB vs 7.68GB)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04


WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-setuptools \
    lsof \
    && rm -rf /var/lib/apt/lists/*

# Create a symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files FIRST (they change infrequently)
# This allows Docker layer caching to work efficiently:
# - Model files are cached in their own layers
# - Python code changes won't invalidate model file layers
########################################################
#### NVIDIA-L20, llama3.1 8B, no bitsandbytes ##########
########################################################
# ## output 100 with some variance
# COPY workload-and-experiment_results/NVIDIA-L20/maxTokens_100-maxTokensStd_10/merged-data/all-with-mixed/final_model-latency_predictor_ttft NVIDIA-L20/maxTokens_100-maxTokensStd_10/final_model-latency_predictor
# COPY workload-and-experiment_results/NVIDIA-L20/maxTokens_100-maxTokensStd_10/merged-data/all-with-mixed/data_replaced-processed.csv NVIDIA-L20/maxTokens_100-maxTokensStd_10/offline_training_data.csv

########################################################
#### NVIDIA-A30, llama3.1 8B, no bitsandbytes ##########
########################################################

COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/azure/azure_code_poisson/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/azure/azure_code_poisson/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear

COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear

COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear_conversation_2 NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear_conversation_2

COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear_toolagent_2 NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear_toolagent_2

# COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear

# COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/azure/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/azure/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear

# COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/azure/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear_without_output_tokens NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/azure/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear_without_output_tokens

COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear_SharingRatio9% NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear_SharingRatio9%

COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear_SharingRatio47%_15000 NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear_SharingRatio47%_15000

COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear_SharingRatio47%_30000 NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear_SharingRatio47%_30000






# COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear_without_output_tokens NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear_without_output_tokens






COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear_without_output_tokens NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear_without_output_tokens

COPY workload-and-experiment_results/NVIDIA-A30/qwen3-4b-instruct/use_given_output_length/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear NVIDIA-A30/qwen3-4b-instruct/use_given_output_length/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear

COPY workload-and-experiment_results/NVIDIA-A30/qwen3-4b-instruct/use_given_output_length/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear NVIDIA-A30/qwen3-4b-instruct/use_given_output_length/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear

COPY workload-and-experiment_results/NVIDIA-A30/qwen3-4b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear_conversation_2 NVIDIA-A30/qwen3-4b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear_conversation_2

COPY workload-and-experiment_results/NVIDIA-A30/qwen3-4b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_2 NVIDIA-A30/qwen3-4b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_2

COPY workload-and-experiment_results/NVIDIA-A30/qwen3-4b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear_toolagent_2 NVIDIA-A30/qwen3-4b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear_toolagent_2

COPY workload-and-experiment_results/NVIDIA-A30/qwen3-4b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_2 NVIDIA-A30/qwen3-4b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_2



COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear

COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear

COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear

COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear




COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear_conversation_2 NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear_conversation_2

COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_2 NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_2

COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear_toolagent_2 NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_e2e_latency_negative_linear_toolagent_2

COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_2 NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_2




# COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/synthetic_realistic_tokenized_rpsscale_1_numtokens_300-toomuch/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_synthetic_realistic_tokenized_rpsscale_1_numtokens_300

# COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/conversation_realistic_tokenized_rpsscale_1_numtokens_300-okayish/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_realistic_tokenized_rpsscale_1_numtokens_300




COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/conversation_realistic_tokenized_rpsscale_1_numtokens_100/rps-1-benchmark/without_bitsandbytes/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_realistic_tokenized_rpsscale_1_numtokens_100


# COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/toolagent_realistic_tokenized_rpsscale_1_numtokens_100/rps-1-benchmark/without_bitsandbytes/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_realistic_tokenized_rpsscale_1_numtokens_100

# COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/toolagent_realistic_tokenized_rpsscale_1_numtokens_100/rps-1-benchmark/without_bitsandbytes/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_quantile_advantage NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_quantile_advantage_toolagent_realistic_tokenized_rpsscale_1_numtokens_100



# COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/synthetic_realistic_tokenized_rpsscale_1_numtokens_100/rps-1-benchmark/without_bitsandbytes/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_synthetic_realistic_tokenized_rpsscale_1_numtokens_100

# COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/toolagent_realistic_tokenized_rpsscale_0.6_numtokens_100/rps-1-benchmark/without_bitsandbytes/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_quantile_based NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_quantile_based_toolagent_realistic_tokenized_rpsscale_0.6_numtokens_100

# COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/toolagent_realistic_tokenized_rpsscale_0.6_numtokens_100/rps-1-benchmark/without_bitsandbytes/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_realistic_tokenized_rpsscale_0.6_numtokens_100


# COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/toolagent_realistic_tokenized_rpsscale_1_numtokens_100/rps-1-benchmark/without_bitsandbytes/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_quantile_based NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_quantile_based_toolagent_realistic_tokenized_rpsscale_1_numtokens_100

# COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/toolagent_realistic_tokenized_rpsscale_0.8_numtokens_100/rps-1-benchmark/without_bitsandbytes/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_quantile_based NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_quantile_based_toolagent_realistic_tokenized_rpsscale_0.8_numtokens_100

# COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/toolagent_realistic_tokenized_rpsscale_0.6_numtokens_100/rps-1-benchmark/without_bitsandbytes/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_quantile_based NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_quantile_based_toolagent_realistic_tokenized_rpsscale_0.6_numtokens_100



COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/conversation_realistic_tokenized_rpsscale_1_numtokens_100/rps-1-benchmark/without_bitsandbytes/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_realistic_tokenized_rpsscale_1_numtokens_100

COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/conversation_realistic_tokenized_rpsscale_1_numtokens_200/rps-1-benchmark/without_bitsandbytes/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_realistic_tokenized_rpsscale_1_numtokens_200

COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/conversation_realistic_tokenized_rpsscale_1_numtokens_300/rps-1-benchmark/without_bitsandbytes/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_realistic_tokenized_rpsscale_1_numtokens_300

COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/toolagent_realistic_tokenized_rpsscale_1_numtokens_100/rps-1-benchmark/without_bitsandbytes/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_realistic_tokenized_rpsscale_1_numtokens_100

COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/toolagent_realistic_tokenized_rpsscale_1_numtokens_200/rps-1-benchmark/without_bitsandbytes/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_realistic_tokenized_rpsscale_1_numtokens_200

COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/synthetic_realistic_tokenized_rpsscale_1_numtokens_100/rps-1-benchmark/without_bitsandbytes/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_synthetic_realistic_tokenized_rpsscale_1_numtokens_100

COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/conversation_realistic_tokenized_rpsscale_3_numtokens_100/rps-1-benchmark/without_bitsandbytes/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_realistic_tokenized_rpsscale_3_numtokens_100

COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/toolagent_realistic_tokenized_rpsscale_2_numtokens_100/rps-1-benchmark/without_bitsandbytes/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_toolagent_realistic_tokenized_rpsscale_2_numtokens_100


COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/synthetic_realistic_tokenized_rpsscale_1.5_numtokens_100_head_2.0x_tail_8.0x_after0.7/rps-1-benchmark/without_bitsandbytes/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear  NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_synthetic_realistic_tokenized_rpsscale_1.5_numtokens_100_head_2.0x_tail_8.0x_after0.7


COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/conversation-2/rps12-benchmark/without_bitsandbytes/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_conversation_2_rps12


COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/random/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear NVIDIA-A30/llama-3-8b-instruct/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_random

COPY workload-and-experiment_results/NVIDIA-A30/llama-3-8b-instruct/random/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_random

###############################################################################
#### hetero: NVIDIA-L20*7 and NVIDIA-A30*8, , llama3.1 8B, no bitsandbytes ####
###############################################################################
# COPY training_data/hetero/final_model-latency_predictor_ttft-20251101_213101-epoch60 hetero/maxTokens_100-maxTokensStd_10/final_model-latency_predictor
# COPY training_data/hetero/data-processed.csv hetero/maxTokens_100-maxTokensStd_10/offline_training_data.csv

##################################################
#### NVIDIA-A10 *7, llama3.1 8B, bitsandbytes ####
##################################################

# Copy application code LAST (changes frequently)
# This way, code changes only rebuild this layer and CMD, not model files
COPY Dockerfile /app/Dockerfile
COPY agent_codes/*.py .
COPY start.sh .
RUN chmod +x /app/start.sh
ENV PORT=8080
ENV NVIDIA_VISIBLE_DEVICES=all
ENV LOG_LEVEL=INFO
EXPOSE 8080

# CMD ["python", "routing_agent_service.py"]
CMD ["/app/start.sh"]