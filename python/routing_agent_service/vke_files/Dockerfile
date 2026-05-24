FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies (before workloads for better caching)

COPY workloads/gangmuk-prefix/SharingRatio70% /app/workload/gangmuk-prefix/SharingRatio70%/
COPY workloads/gangmuk-prefix/SharingRatio50% /app/workload/gangmuk-prefix/SharingRatio50%/
COPY workloads/gangmuk-prefix/SharingRatio30% /app/workload/gangmuk-prefix/SharingRatio30%/
COPY workloads/gangmuk-prefix/SharingRatio10% /app/workload/gangmuk-prefix/SharingRatio10%/
COPY workloads/gangmuk-prefix/MixedSharingRatio10_30_50_70% /app/workload/gangmuk-prefix/MixedSharingRatio10_30_50_70%/


COPY workloads/mooncake/original_toolagent-2 /app/workload/mooncake/original_toolagent-2
COPY workloads/mooncake/original_conversation-2 /app/workload/mooncake/original_conversation-2

COPY workloads/mooncake/extended_workload/conversation-2 /app/workload/mooncake/conversation-2-extended-ver1
COPY workloads/mooncake/extended_workload/toolagent-2 /app/workload/mooncake/toolagent-2-extended-ver1
COPY workloads/mooncake/synthetic_3x-numtokens_100 /app/workload/mooncake/synthetic_3x-numtokens_100

COPY workloads/gangmuk-prefix/SharingRatio0%50% /app/workload/gangmuk-prefix/SharingRatio_0%_50%
COPY workloads/gangmuk-prefix/SharingRatio5%50% /app/workload/gangmuk-prefix/SharingRatio_5%_50%

COPY workloads/overhead_test /app/workload/mooncake/overhead_test


# COPY workloads/mooncake/conversation-1 /app/workload/mooncake/conversation-1/
# COPY workloads/mooncake/toolagent-1 /app/workload/mooncake/toolagent-1/
# COPY workloads/mooncake/conversation-2 /app/workload/mooncake/conversation-2/
# COPY workloads/mooncake/toolagent-2 /app/workload/mooncake/toolagent-2/

# COPY workloads/azure/azure_code_poisson /app/workload/azure/azure_code_poisson
# COPY workloads/azure/azure_code /app/workload/azure/azure_code/
# COPY workloads/azure/azure_conv-singleturn /app/workload/azure/azure_conv-singleturn/
# COPY workloads/azure/azure_conv-multiturn /app/workload/azure/azure_conv-multiturn/

# COPY workloads/mooncake/conversation_realistic_workload /app/workload/mooncake/conversation_realistic
# COPY workloads/mooncake/toolagent_realistic_workload /app/workload/mooncake/toolagent_realistic
# COPY workloads/mooncake/synthetic_realistic_workload /app/workload/mooncake/synthetic_realistic

# COPY workloads/mooncake/conversation_realistic_workload_tokenized /app/workload/mooncake/conversation_realistic_tokenized
# COPY workloads/mooncake/toolagent_realistic_workload_tokenized /app/workload/mooncake/toolagent_realistic_tokenized
# COPY workloads/mooncake/synthetic_realistic_workload_tokenized /app/workload/mooncake/synthetic_realistic_tokenized
# COPY workloads/mooncake/extended_workload/ver0/conversation-2 /app/workload/mooncake/conversation-2-extended-ver0
# COPY workloads/mooncake/extended_workload/ver0/toolagent-2 /app/workload/mooncake/toolagent-2-extended-ver0
# COPY workloads/mooncake/synthetic_realistic_workload_tokenized-rpsscale_1-numtokens_100-outputscale_1.0-duration_2000-head_1.5x-tail_11.0x-after0.7 /app/workload/mooncake/synthetic-2-extended-ver1
# COPY workloads/mooncake/synthetic_realistic_workload_tokenized-rpsscale_1-numtokens_100-outputscale_1.0-duration_4000-head_1.7x-tail_10.0x-after0.85 /app/workload/mooncake/synthetic-2-extended-ver2
# COPY workloads/mooncake/synthetic_realistic_workload_tokenized-rpsscale_1-numtokens_100-outputscale_1.0-duration_2000-head_1.5x-tail_11.0x-after0.7-smoothed /app/workload/mooncake/synthetic_realistic_workload_tokenized-rpsscale_1-numtokens_100-outputscale_1.0-head_1.5x-tail_11.0x-after0.7-smoothed


COPY vke_files/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy client script and utilities (before workloads for better caching)
COPY vke_files/async-client.py .
COPY vke_files/utils.py .
COPY vke_files/logger.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command - sleep infinity so pod stays running and we can kubectl exec into it
CMD ["sleep", "infinity"]
