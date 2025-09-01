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
COPY agent_codes/*.py .
COPY start.sh .

## model
# COPY training_data/p4096_s1024_rps20/rl+random/final_model final_model
COPY training_data/SharingRatio71%-p2048_s512_rps5_spp_10_ndp50-p4096_s1024_rps8_spp_10_ndp50-p8096_s2048_rps3_spp_10_ndp50-half/final_model-data final_model

# Remove the duplicate COPY command that overwrites the model above
COPY Dockerfile /app/Dockerfile

RUN pip install --no-cache-dir -r requirements.txt
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
