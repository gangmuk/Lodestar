FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
# FROM pytorch/pytorch:2.1.0-cpu

WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# Create a symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt .
COPY *.py .
COPY final_model final_model

# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

ENV PORT=8080
ENV NVIDIA_VISIBLE_DEVICES=all
ENV LOG_LEVEL=INFO
# ENV MODEL_PATH=/app/latency_predictor.joblib

EXPOSE 8080

# CMD ["python", "routing_agent_service.py"] # this way it makes routing_agent_service.py PID=1, and it can't be killed by SIGTERM
CMD ["sh", "-c", "python routing_agent_service.py"]