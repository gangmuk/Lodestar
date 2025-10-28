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




# COPY workload-and-experiment_results/workload-generator/gangmuk-prefix-sharing/old-version/input10002000/rps2030/SharingRatio10%_avginput1500_globalrps20/workload.jsonl /app/workload/old-version/input10002000/rps2030/SharingRatio10%_avginput1500_globalrps20/

# COPY workload-and-experiment_results/workload-generator/gangmuk-prefix-sharing/old-version/input10002000/rps2030/SharingRatio30%_avginput1500_globalrps20/workload.jsonl /app/workload/old-version/input10002000/rps2030/SharingRatio30%_avginput1500_globalrps20/

# COPY workload-and-experiment_results/workload-generator/gangmuk-prefix-sharing/old-version/input10002000/rps2030/SharingRatio50%_avginput1500_globalrps25/workload.jsonl /app/workload/old-version/input10002000/rps2030/SharingRatio50%_avginput1500_globalrps25/

# COPY workload-and-experiment_results/workload-generator/gangmuk-prefix-sharing/old-version/input10002000/rps2030/SharingRatio70%_avginput1500_globalrps35/workload.jsonl /app/workload/old-version/input10002000/rps2030/SharingRatio70%_avginput1500_globalrps35/




# COPY workload-and-experiment_results/workload-generator/gangmuk-prefix-sharing/old-version/input10002000/rps1520/SharingRatio10%_avginput1500_globalrps15/workload.jsonl /app/workload/old-version/input10002000/rps1520/SharingRatio10%_avginput1500_globalrps15/

# COPY workload-and-experiment_results/workload-generator/gangmuk-prefix-sharing/old-version/input10002000/rps1520/SharingRatio30%_avginput1500_globalrps15/workload.jsonl /app/workload/old-version/input10002000/rps1520/SharingRatio30%_avginput1500_globalrps15/

# COPY workload-and-experiment_results/workload-generator/gangmuk-prefix-sharing/old-version/input10002000/rps1520/SharingRatio50%_avginput1500_globalrps17/workload.jsonl /app/workload/old-version/input10002000/rps1520/SharingRatio50%_avginput1500_globalrps17/

# COPY workload-and-experiment_results/workload-generator/gangmuk-prefix-sharing/old-version/input10002000/rps1520/SharingRatio70%_avginput1500_globalrps20/workload.jsonl /app/workload/old-version/input10002000/rps1520/SharingRatio70%_avginput1500_globalrps20/




# COPY workload-and-experiment_results/workload-generator/gangmuk-prefix-sharing/old-version/input100020004000/SharingRatio10%_avginput2333_globalrps10/workload.jsonl /app/workload/old-version/input100020004000/SharingRatio10%_avginput2333_globalrps10/

# COPY workload-and-experiment_results/workload-generator/gangmuk-prefix-sharing/old-version/input100020004000/SharingRatio30%_avginput2333_globalrps10/workload.jsonl /app/workload/old-version/input100020004000/SharingRatio30%_avginput2333_globalrps10/

# COPY workload-and-experiment_results/workload-generator/gangmuk-prefix-sharing/old-version/input100020004000/SharingRatio50%_avginput2333_globalrps10/workload.jsonl /app/workload/old-version/input100020004000/SharingRatio50%_avginput2333_globalrps10/

# COPY workload-and-experiment_results/workload-generator/gangmuk-prefix-sharing/old-version/input100020004000/SharingRatio70%_avginput2333_globalrps12/workload.jsonl /app/workload/old-version/input100020004000/SharingRatio70%_avginput2333_globalrps12/

# COPY workload-and-experiment_results/workload-generator/gangmuk-prefix-sharing/config_sharing10%/SharingRatio10%_avginput3750_multirps10-15-10_rep10/workload.jsonl /app/workload/config_sharing10%/
# COPY workload-and-experiment_results/workload-generator/gangmuk-prefix-sharing/config_sharing30%/SharingRatio30%_avginput3750_multirps10-15-10_rep10/workload.jsonl /app/workload/config_sharing30%/
# COPY workload-and-experiment_results/workload-generator/gangmuk-prefix-sharing/config_sharing50%/SharingRatio50%_avginput3750_multirps12-15-12_rep10/workload.jsonl /app/workload/config_sharing50%/
# COPY workload-and-experiment_results/workload-generator/gangmuk-prefix-sharing/config_sharing70%/SharingRatio70%_avginput3750_multirps12-15-12_rep10/workload.jsonl /app/workload/config_sharing70%/


# text to sql
COPY workload-and-experiment_results/workload-generator/gangmuk-prefix-sharing/text_to_sql/SharingRatio31%_avginput4160_multirps10-12-10_rep10/workload.jsonl /app/workload/text_to_sql/

# multiturn chat
COPY workload-and-experiment_results/workload-generator/gangmuk-prefix-sharing/multiturn-chat/multiturn_sessions300_avgturns5_turnlen1000_rps10/workload.jsonl /app/workload/multiturn-chat/

# mooncake conversation
COPY workload-and-experiment_results/workload-generator/mooncake/workload-conversation/workload.jsonl /app/workload/mooncake-conversation/

# mooncake toolagent
COPY workload-and-experiment_results/workload-generator/mooncake/workload-toolagent/workload.jsonl /app/workload/mooncake-toolagent/

# test purpose
COPY workload-and-experiment_results/ten_request/workload.jsonl /app/workload/ten_request/
COPY workload-and-experiment_results/hundred_request/workload.jsonl /app/workload/hundred_request/

# working
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
