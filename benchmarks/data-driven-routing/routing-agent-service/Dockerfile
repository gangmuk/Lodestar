FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
# FROM pytorch/pytorch:2.1.0-cpu

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
COPY agent_codes/ /app/agent_codes/
COPY start.sh .

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


###############################################################################
#### hetero: NVIDIA-L20*7 and NVIDIA-A30*8, , llama3.1 8B, no bitsandbytes ####
###############################################################################
# COPY training_data/hetero/final_model-latency_predictor_ttft-20251101_213101-epoch60 hetero/maxTokens_100-maxTokensStd_10/final_model-latency_predictor
# COPY training_data/hetero/data-processed.csv hetero/maxTokens_100-maxTokensStd_10/offline_training_data.csv

##################################################
#### NVIDIA-A10 *7, llama3.1 8B, bitsandbytes ####
##################################################



COPY Dockerfile /app/Dockerfile
RUN chmod +x /app/start.sh
ENV PORT=8080
ENV NVIDIA_VISIBLE_DEVICES=all
ENV LOG_LEVEL=INFO
EXPOSE 8080

# CMD ["python", "routing_agent_service.py"]
CMD ["/app/start.sh"]