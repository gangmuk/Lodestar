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

COPY agent_codes/*.py .
COPY start.sh .

## model
# COPY training_data/p4096_s1024_rps20/rl+random/final_model final_model
# COPY training_data/p4096_s1024_rps20/rl+random/final_model_backup final_model
# COPY training_data/p4096_s1024_rps20/rl+random/final_model-data_replaced-processed final_model
# COPY training_data/merged-data/all/final_model-data_replaced-processed-linear_simple_extended-lr_0.001-ttft_weight_2.0-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_prefill_tokens-hidden_dim_64-lrs_exp-latency_predictor final_model
# COPY training_data/p4096_s1024_rps20/all/final_model-data_replaced-processed-linear_simple-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_none-hidden_dim_64-lrs_grad_adapt final_model
# COPY training_data/SharingRatio9%/all/final_model-data-processed final_model
# COPY training_data/SharingRatio71%/all/final_model-data-processed-linear_simple-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50 final_model

################################################################################################################
#### GPU-L3c, NVIDIA-L20
COPY training_data/L20-7/merged-data/all-with-mixed/final_model-latency_predictor_ttft-20251101_205621 final_model/NVIDIA-L20
COPY training_data/L20-7/merged-data/all-with-mixed/data_replaced-processed.csv NVIDIA-L20/offline_training_data.csv

#### NVIDIA-A30
# COPY training_data/A30-8/final_model-latency_predictor_ttft-20251028_183743 final_model/NVIDIA-A30
COPY training_data/A30-8/final_model-latency_predictor_ttft-20251101_205321 final_model/NVIDIA-A30
COPY training_data/A30-8/data-processed.csv NVIDIA-A30/offline_training_data.csv

#### hetero: NVIDIA-L20*7 and NVIDIA-A30*8
COPY training_data/hetero/final_model-latency_predictor_ttft-20251101_213101-epoch60 final_model/hetero
COPY training_data/hetero/data-processed.csv hetero/offline_training_data.csv

#### NVIDIA-A10
COPY workload-and-experiment_results/NVIDIA-A10/MixedSharingRatio10_30_50_70%/rps5/final_model-latency_predictor_ttft-20251120_173020 final_model/NVIDIA-A10/MixedSharingRatio10_30_50_70%
COPY workload-and-experiment_results/NVIDIA-A10/MixedSharingRatio10_30_50_70%/rps5/data-processed.csv offline_training_data_NVIDIA-A10_MixedSharingRatio10_30_50_70%.csv

COPY workload-and-experiment_results/NVIDIA-A10/SharingRatio71%/rps6.5/final_model-latency_predictor_ttft-20251120_172727 final_model/NVIDIA-A10/SharingRatio71%
COPY workload-and-experiment_results/NVIDIA-A10/SharingRatio71%/rps6.5/data-processed.csv offline_training_data_NVIDIA-A10_SharingRatio71%.csv

# CRITICAL FIX: Copy model directly to /app/final_model/NVIDIA-A10/ (not a subdirectory)
# to match the path expected by routing_agent_service.py
# COPY workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0/final_model-latency_predictor_ttft-0 final_model/NVIDIA-A10
COPY workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0/final_model-latency_predictor_ttft-1-without-output final_model/NVIDIA-A10/PrefillOnly/latency_predictor
COPY workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0/data-processed.csv NVIDIA-A10/PrefillOnly/offline_training_data.csv

COPY workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0/final_model-latency_predictor_ttft-1-with-output final_model/NVIDIA-A10/latency_predictor/PrefillOnly/with-output_tokens

## contextual bandit (inverse_latency reward function)
COPY workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0/final_model-contextual_bandit-linear_simple_extended final_model/NVIDIA-A10/PrefillOnly/contextual_bandit/linear_simple_extended
COPY workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0/final_model-contextual_bandit-inverse_latency final_model/NVIDIA-A10/PrefillOnly/contextual_bandit/inverse_latency
COPY workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0/final_model-contextual_bandit-latency_optimized final_model/NVIDIA-A10/PrefillOnly/contextual_bandit/latency_optimized

################################################################################################################

COPY Dockerfile /app/Dockerfile
RUN chmod +x /app/start.sh
ENV PORT=8080
ENV NVIDIA_VISIBLE_DEVICES=all
ENV LOG_LEVEL=INFO
EXPOSE 8080

# CMD ["python", "routing_agent_service.py"]
CMD ["/app/start.sh"]