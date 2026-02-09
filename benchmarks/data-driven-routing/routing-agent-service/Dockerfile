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

COPY workload-and-experiment_results/NVIDIA-A30/maxTokens_100-maxTokensStd_10/final_model-latency_predictor_ttft NVIDIA-A30/maxTokens_100-maxTokensStd_10/final_model-latency_predictor
COPY workload-and-experiment_results/NVIDIA-A30/maxTokens_100-maxTokensStd_10/data-processed.csv NVIDIA-A30/maxTokens_100-maxTokensStd_10/offline_training_data.csv

## output 1
COPY workload-and-experiment_results/NVIDIA-A30/maxTokens_1-maxTokensStd_0/data-processed.csv NVIDIA-A30/maxTokens_1-maxTokensStd_0/offline_training_data.csv
COPY workload-and-experiment_results/NVIDIA-A30/maxTokens_1-maxTokensStd_0/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear NVIDIA-A30/maxTokens_1-maxTokensStd_0/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear

COPY workload-and-experiment_results/NVIDIA-A30/maxTokens_1-maxTokensStd_0/mooncake/data-processed.csv NVIDIA-A30/maxTokens_1-maxTokensStd_0/mooncake/conversation/offline_training_data.csv
COPY workload-and-experiment_results/NVIDIA-A30/maxTokens_1-maxTokensStd_0/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear NVIDIA-A30/maxTokens_1-maxTokensStd_0/mooncake/conversation/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear

# COPY workload-and-experiment_results/NVIDIA-A30/maxTokens_1-maxTokensStd_0/mooncake/conversation/data-processed.csv NVIDIA-A30/maxTokens_1-maxTokensStd_0/mooncake/conversation/offline_training_data.csv
# COPY workload-and-experiment_results/NVIDIA-A30/maxTokens_1-maxTokensStd_0/mooncake/conversation/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear NVIDIA-A30/maxTokens_1-maxTokensStd_0/mooncake/conversation/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear
# COPY workload-and-experiment_results/NVIDIA-A30/maxTokens_1-maxTokensStd_0/mooncake/toolagent/data-processed.csv NVIDIA-A30/maxTokens_1-maxTokensStd_0/mooncake/toolagent/offline_training_data.csv
# COPY workload-and-experiment_results/NVIDIA-A30/maxTokens_1-maxTokensStd_0/mooncake/toolagent/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear NVIDIA-A30/maxTokens_1-maxTokensStd_0/mooncake/toolagent/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear

COPY workload-and-experiment_results/NVIDIA-A30/maxTokens_1-maxTokensStd_0/azure/azure_code_poisson/data-processed.csv NVIDIA-A30/maxTokens_1-maxTokensStd_0/azure/azure_code_poisson/offline_training_data.csv
COPY workload-and-experiment_results/NVIDIA-A30/maxTokens_1-maxTokensStd_0/azure/azure_code_poisson/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear NVIDIA-A30/maxTokens_1-maxTokensStd_0/azure/azure_code_poisson/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear

COPY workload-and-experiment_results/NVIDIA-A30/maxTokens_1-maxTokensStd_0/mooncake/data-processed.csv NVIDIA-A30/maxTokens_1-maxTokensStd_0/mooncake/offline_training_data.csv
COPY workload-and-experiment_results/NVIDIA-A30/maxTokens_1-maxTokensStd_0/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear NVIDIA-A30/maxTokens_1-maxTokensStd_0/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear

COPY workload-and-experiment_results/NVIDIA-A30/maxTokens_1-maxTokensStd_0/gangmuk-prefix/data-processed.csv NVIDIA-A30/maxTokens_1-maxTokensStd_0/gangmuk-prefix/offline_training_data.csv

COPY workload-and-experiment_results/NVIDIA-A30/maxTokens_1-maxTokensStd_0/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear NVIDIA-A30/maxTokens_1-maxTokensStd_0/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear

COPY workload-and-experiment_results/NVIDIA-A30/maxTokens_1-maxTokensStd_0/before_latency_optimization/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear NVIDIA-A30/maxTokens_1-maxTokensStd_0/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear-before_latency_optimization

COPY workload-and-experiment_results/NVIDIA-A30/maxTokens_1-maxTokensStd_0/azure/data-processed.csv NVIDIA-A30/maxTokens_1-maxTokensStd_0/azure/offline_training_data.csv
COPY workload-and-experiment_results/NVIDIA-A30/maxTokens_1-maxTokensStd_0/azure/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear NVIDIA-A30/maxTokens_1-maxTokensStd_0/azure/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear

## use_given_output_length
COPY workload-and-experiment_results/NVIDIA-A30/use_given_output_length/azure/data-processed.csv NVIDIA-A30/use_given_output_length/azure/offline_training_data.csv
COPY workload-and-experiment_results/NVIDIA-A30/use_given_output_length/azure/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear NVIDIA-A30/use_given_output_length/azure/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear

COPY workload-and-experiment_results/NVIDIA-A30/use_given_output_length/mooncake/data-processed.csv NVIDIA-A30/use_given_output_length/mooncake/offline_training_data.csv
COPY workload-and-experiment_results/NVIDIA-A30/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear NVIDIA-A30/use_given_output_length/mooncake/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear

COPY workload-and-experiment_results/NVIDIA-A30/use_given_output_length/gangmuk-prefix/data-processed.csv NVIDIA-A30/use_given_output_length/gangmuk-prefix/offline_training_data.csv
COPY workload-and-experiment_results/NVIDIA-A30/use_given_output_length/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear-without_output_tokens NVIDIA-A30/use_given_output_length/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear-without_output_tokens

COPY workload-and-experiment_results/NVIDIA-A30/use_given_output_length/gangmuk-prefix/data-processed.csv NVIDIA-A30/use_given_output_length/gangmuk-prefix/offline_training_data.csv
COPY workload-and-experiment_results/NVIDIA-A30/use_given_output_length/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear-with_output_tokens NVIDIA-A30/use_given_output_length/gangmuk-prefix/final_model-contextual_bandit_perpodmodel_checkpoint_negative_linear-with_output_tokens

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