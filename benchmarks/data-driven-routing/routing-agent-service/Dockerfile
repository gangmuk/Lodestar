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
COPY agent_codes/agents/*.py .
COPY agent_codes/envs/*.py .
COPY agent_codes/policies/*.py .
COPY start.sh .

## model
# COPY training_data/p4096_s1024_rps20/rl+random/final_model final_model
# COPY training_data/p4096_s1024_rps20/rl+random/final_model_backup final_model
# COPY training_data/p4096_s1024_rps20/rl+random/final_model-data_replaced-processed final_model
# COPY training_data/merged-data/all/final_model-data_replaced-processed-linear_simple_extended-lr_0.001-ttft_weight_2.0-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_prefill_tokens-hidden_dim_64-lrs_exp-latency_predictor final_model
COPY training_data/p4096_s1024_rps20/all/final_model-data_replaced-processed-linear_simple-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50-without_none-hidden_dim_64-lrs_grad_adapt final_model
# COPY training_data/SharingRatio9%/all/final_model-data-processed final_model
# COPY training_data/SharingRatio71%/all/final_model-data-processed-linear_simple-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50 final_model

# Remove the duplicate COPY command that overwrites the model above
COPY Dockerfile /app/Dockerfile
# RUN pip install -r requirements.txt
RUN chmod +x /app/start.sh

ENV PORT=8080
ENV NVIDIA_VISIBLE_DEVICES=all
ENV LOG_LEVEL=INFO

EXPOSE 8080

#########################################################################
## Use this when you don't use ship_all.py but just restart deployment ##
#########################################################################
# CMD ["python", "routing_agent_service.py"]

##########################################################
## Use this when you use ship_all.py to run the service ##
##########################################################
CMD ["/app/start.sh"]



# FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
# # FROM pytorch/pytorch:2.1.0-cpu

# WORKDIR /app

# # Install Python and pip
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3.10 \
#     python3-pip \
#     python3-setuptools \
#     && rm -rf /var/lib/apt/lists/*

# # Create a symlink for python
# RUN ln -s /usr/bin/python3 /usr/bin/python

# COPY requirements.txt .
# COPY *.py .
# COPY final_model final_model
# COPY Dockerfile /app/Dockerfile

# # RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install -r requirements.txt

# ENV PORT=8080
# ENV NVIDIA_VISIBLE_DEVICES=all
# ENV LOG_LEVEL=INFO
# # ENV MODEL_PATH=/app/latency_predictor.joblib

# EXPOSE 8080

# CMD ["python", "routing_agent_service.py"]
