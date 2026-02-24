FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies (before workloads for better caching)
COPY vke_files/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY workload-and-experiment_results/gangmuk-prefix/SharingRatio71%/workload.jsonl /app/workload/gangmuk-prefix/SharingRatio71%/
COPY workload-and-experiment_results/gangmuk-prefix/SharingRatio47%/workload.jsonl /app/workload/gangmuk-prefix/SharingRatio47%/
COPY workload-and-experiment_results/gangmuk-prefix/SharingRatio28%/workload.jsonl /app/workload/gangmuk-prefix/SharingRatio28%/
COPY workload-and-experiment_results/gangmuk-prefix/SharingRatio9%/workload.jsonl /app/workload/gangmuk-prefix/SharingRatio9%/
COPY workload-and-experiment_results/gangmuk-prefix/MixedSharingRatio10_30_50_70%/workload.jsonl /app/workload/gangmuk-prefix/MixedSharingRatio10_30_50_70%/
COPY workload-and-experiment_results/mooncake/conversation-1/workload.jsonl /app/workload/mooncake/conversation-1/
COPY workload-and-experiment_results/mooncake/toolagent-1/workload.jsonl /app/workload/mooncake/toolagent-1/
COPY workload-and-experiment_results/mooncake/conversation-2/workload.jsonl /app/workload/mooncake/conversation-2/
COPY workload-and-experiment_results/mooncake/toolagent-2/workload.jsonl /app/workload/mooncake/toolagent-2/

COPY workload-and-experiment_results/azure/azure_code_poisson /app/workload/azure/azure_code_poisson
COPY workload-and-experiment_results/azure/azure_code /app/workload/azure/azure_code/
COPY workload-and-experiment_results/azure/azure_conv-singleturn /app/workload/azure/azure_conv-singleturn/
COPY workload-and-experiment_results/azure/azure_conv-multiturn /app/workload/azure/azure_conv-multiturn/

# Copy client script and utilities (before workloads for better caching)
COPY vke_files/async-client.py .
COPY vke_files/utils.py .
COPY vke_files/logger.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command - sleep infinity so pod stays running and we can kubectl exec into it
CMD ["sleep", "infinity"]
