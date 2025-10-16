FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY vke_files/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy client script and utilities
COPY vke_files/async-client.py .
COPY vke_files/utils.py .
COPY vke_files/logger.py .

# Create workload directory structure
RUN mkdir -p /app/workload/ten_request \
    /app/workload/hundred_request \
    /app/workload/SharingRatio71% \
    /app/workload/SharingRatio47% \
    /app/workload/SharingRatio28% \
    /app/workload/SharingRatio9% \
    /app/workload/MixedSharingRatio10_30_50_70%

# Copy workload files from parent directory
# Build context is set to parent directory in build-and-push.sh
COPY workload-and-experiment_results/ten_request/workload.jsonl /app/workload/ten_request/
COPY workload-and-experiment_results/hundred_request/workload.jsonl /app/workload/hundred_request/
COPY workload-and-experiment_results/SharingRatio71%/workload.jsonl /app/workload/SharingRatio71%/
COPY workload-and-experiment_results/SharingRatio47%/workload.jsonl /app/workload/SharingRatio47%/
COPY workload-and-experiment_results/SharingRatio28%/workload.jsonl /app/workload/SharingRatio28%/
COPY workload-and-experiment_results/SharingRatio9%/workload.jsonl /app/workload/SharingRatio9%/
COPY workload-and-experiment_results/workload-generator/gangmuk-prefix-sharing/config_sharing_mixed_10_30_50_70%/SharingRatio39%_avginput3750_globalrps12/workload.jsonl /app/workload/MixedSharingRatio10_30_50_70%/

# Create output directory
RUN mkdir -p /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command - sleep infinity so pod stays running and we can kubectl exec into it
CMD ["sleep", "infinity"]
