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

COPY training_data/L20-7/merged-data/all-with-mixed/final_model-latency_predictor_ttft final_model/GPU-L3c
COPY training_data/L20-7/merged-data/all-with-mixed/data_replaced-processed.csv offline_training_data_GPU-L3c.csv 

# COPY training_data/A30-8/final_model-latency_predictor_ttft-20251028_183743 final_model/NVIDIA-A30
COPY training_data/A30-8/final_model-latency_predictor_ttft-20251029_231330 final_model/NVIDIA-A30
COPY training_data/A30-8/data-processed.csv offline_training_data_NVIDIA-A30.csv 

COPY workload-and-experiment_results/for_paper/hetero/final_model-latency_predictor_ttft-20251029_034844 final_model/hetero
COPY workload-and-experiment_results/for_paper/hetero/data-processed.csv offline_training_data_hetero.csv 

COPY Dockerfile /app/Dockerfile
RUN chmod +x /app/start.sh
ENV PORT=8080
ENV NVIDIA_VISIBLE_DEVICES=all
ENV LOG_LEVEL=INFO
EXPOSE 8080

# CMD ["python", "routing_agent_service.py"]
CMD ["/app/start.sh"]