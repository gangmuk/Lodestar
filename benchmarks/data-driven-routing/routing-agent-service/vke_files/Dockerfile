FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies (before workloads for better caching)

COPY workload-and-experiment_results/gangmuk-prefix/SharingRatio71% /app/workload/gangmuk-prefix/SharingRatio71%/
COPY workload-and-experiment_results/gangmuk-prefix/SharingRatio47% /app/workload/gangmuk-prefix/SharingRatio47%/
COPY workload-and-experiment_results/gangmuk-prefix/SharingRatio28% /app/workload/gangmuk-prefix/SharingRatio28%/
COPY workload-and-experiment_results/gangmuk-prefix/SharingRatio9% /app/workload/gangmuk-prefix/SharingRatio9%/
COPY workload-and-experiment_results/gangmuk-prefix/MixedSharingRatio10_30_50_70% /app/workload/gangmuk-prefix/MixedSharingRatio10_30_50_70%/
COPY workload-and-experiment_results/mooncake/conversation-1 /app/workload/mooncake/conversation-1/
COPY workload-and-experiment_results/mooncake/toolagent-1 /app/workload/mooncake/toolagent-1/
COPY workload-and-experiment_results/mooncake/conversation-2 /app/workload/mooncake/conversation-2/
COPY workload-and-experiment_results/mooncake/toolagent-2 /app/workload/mooncake/toolagent-2/

COPY workload-and-experiment_results/azure/azure_code_poisson /app/workload/azure/azure_code_poisson
COPY workload-and-experiment_results/azure/azure_code /app/workload/azure/azure_code/
COPY workload-and-experiment_results/azure/azure_conv-singleturn /app/workload/azure/azure_conv-singleturn/
COPY workload-and-experiment_results/azure/azure_conv-multiturn /app/workload/azure/azure_conv-multiturn/

COPY workload-and-experiment_results/workload-generator/mooncake/conversation_realistic_workload /app/workload/mooncake/conversation_realistic
COPY workload-and-experiment_results/workload-generator/mooncake/toolagent_realistic_workload /app/workload/mooncake/toolagent_realistic
COPY workload-and-experiment_results/workload-generator/mooncake/synthetic_realistic_workload /app/workload/mooncake/synthetic_realistic

COPY workload-and-experiment_results/workload-generator/mooncake/conversation_realistic_workload_tokenized /app/workload/mooncake/conversation_realistic_tokenized
COPY workload-and-experiment_results/workload-generator/mooncake/toolagent_realistic_workload_tokenized /app/workload/mooncake/toolagent_realistic_tokenized
COPY workload-and-experiment_results/workload-generator/mooncake/synthetic_realistic_workload_tokenized /app/workload/mooncake/synthetic_realistic_tokenized

COPY workload-and-experiment_results/workload-generator/mooncake/conversation_realistic_workload_tokenized-rpsscale_2-numtokens_100-outputscale_1-duration_600 /app/workload/mooncake/conversation_realistic_tokenized_rpsscale_2



COPY workload-and-experiment_results/workload-generator/mooncake/conversation_realistic_workload_tokenized-rpsscale_1-numtokens_100-outputscale_1-duration_1200 /app/workload/mooncake/conversation_realistic_tokenized_rpsscale_1_numtokens_100

COPY workload-and-experiment_results/workload-generator/mooncake/synthetic_realistic_workload_tokenized-rpsscale_1-numtokens_100-outputscale_1-duration_1200 /app/workload/mooncake/synthetic_realistic_tokenized_rpsscale_1_numtokens_100

COPY workload-and-experiment_results/workload-generator/mooncake/toolagent_realistic_workload_tokenized-rpsscale_1-numtokens_100-outputscale_1-duration_1200 /app/workload/mooncake/toolagent_realistic_tokenized_rpsscale_1_numtokens_100


COPY workload-and-experiment_results/workload-generator/mooncake/conversation_realistic_workload_tokenized-rpsscale_1-numtokens_300-outputscale_1-duration_1200 /app/workload/mooncake/conversation_realistic_tokenized_rpsscale_1_numtokens_300

COPY workload-and-experiment_results/workload-generator/mooncake/synthetic_realistic_workload_tokenized-rpsscale_1-numtokens_300-outputscale_1-duration_1200 /app/workload/mooncake/synthetic_realistic_tokenized_rpsscale_1_numtokens_300

COPY workload-and-experiment_results/workload-generator/mooncake/toolagent_realistic_workload_tokenized-rpsscale_1-numtokens_300-outputscale_1-duration_1200 /app/workload/mooncake/toolagent_realistic_tokenized_rpsscale_1_numtokens_300



COPY workload-and-experiment_results/workload-generator/mooncake/conversation_realistic_workload_tokenized-rpsscale_1-numtokens_200-outputscale_1-duration_1200 /app/workload/mooncake/conversation_realistic_tokenized_rpsscale_1_numtokens_200

COPY workload-and-experiment_results/workload-generator/mooncake/synthetic_realistic_workload_tokenized-rpsscale_1-numtokens_200-outputscale_1-duration_1200 /app/workload/mooncake/synthetic_realistic_tokenized_rpsscale_1_numtokens_200

COPY workload-and-experiment_results/workload-generator/mooncake/toolagent_realistic_workload_tokenized-rpsscale_1-numtokens_200-outputscale_1-duration_1200 /app/workload/mooncake/toolagent_realistic_tokenized_rpsscale_1_numtokens_200


COPY workload-and-experiment_results/workload-generator/mooncake/conversation_realistic_workload_tokenized-rpsscale_1-numtokens_500-outputscale_1-duration_1200 /app/workload/mooncake/conversation_realistic_tokenized_rpsscale_1_numtokens_500

COPY workload-and-experiment_results/workload-generator/mooncake/synthetic_realistic_workload_tokenized-rpsscale_1-numtokens_500-outputscale_1-duration_1200 /app/workload/mooncake/synthetic_realistic_tokenized_rpsscale_1_numtokens_500

COPY workload-and-experiment_results/workload-generator/mooncake/toolagent_realistic_workload_tokenized-rpsscale_1-numtokens_500-outputscale_1-duration_1200 /app/workload/mooncake/toolagent_realistic_tokenized_rpsscale_1_numtokens_500




COPY workload-and-experiment_results/workload-generator/mooncake/toolagent_realistic_workload_tokenized-rpsscale_0.8-numtokens_100-outputscale_1-duration_1200 /app/workload/mooncake/toolagent_realistic_tokenized_rpsscale_0.8_numtokens_100

COPY workload-and-experiment_results/workload-generator/mooncake/toolagent_realistic_workload_tokenized-rpsscale_0.6-numtokens_100-outputscale_1-duration_1200 /app/workload/mooncake/toolagent_realistic_tokenized_rpsscale_0.6_numtokens_100




# COPY workload-and-experiment_results/workload-generator/mooncake/conversation_realistic_workload_tokenized-rpsscale_2-numtokens_100-outputscale_1-duration_1200 /app/workload/mooncake/conversation_realistic_tokenized_rpsscale_2_numtokens_100

# COPY workload-and-experiment_results/workload-generator/mooncake/synthetic_realistic_workload_tokenized-rpsscale_2-numtokens_100-outputscale_1-duration_1200 /app/workload/mooncake/synthetic_realistic_tokenized_rpsscale_2_numtokens_100

# COPY workload-and-experiment_results/workload-generator/mooncake/toolagent_realistic_workload_tokenized-rpsscale_2-numtokens_100-outputscale_1-duration_1200 /app/workload/mooncake/toolagent_realistic_tokenized_rpsscale_2_numtokens_100

COPY workload-and-experiment_results/workload-generator/mooncake/conversation_realistic_workload_tokenized-rpsscale_2-numtokens_100-outputscale_1-duration_600 /app/workload/mooncake/conversation_realistic_tokenized_rpsscale_2_numtokens_100

COPY workload-and-experiment_results/workload-generator/mooncake/synthetic_realistic_workload_tokenized-rpsscale_2-numtokens_100-outputscale_1-duration_600 /app/workload/mooncake/synthetic_realistic_tokenized_rpsscale_2_numtokens_100

COPY workload-and-experiment_results/workload-generator/mooncake/toolagent_realistic_workload_tokenized-rpsscale_2-numtokens_100-outputscale_1-duration_600 /app/workload/mooncake/toolagent_realistic_tokenized_rpsscale_2_numtokens_100



COPY workload-and-experiment_results/workload-generator/mooncake/conversation_realistic_workload_tokenized-rpsscale_3-numtokens_100-outputscale_1-duration_600 /app/workload/mooncake/conversation_realistic_tokenized_rpsscale_3_numtokens_100

COPY workload-and-experiment_results/workload-generator/mooncake/synthetic_realistic_workload_tokenized-rpsscale_3-numtokens_100-outputscale_1-duration_600 /app/workload/mooncake/synthetic_realistic_tokenized_rpsscale_3_numtokens_100

COPY workload-and-experiment_results/workload-generator/mooncake/toolagent_realistic_workload_tokenized-rpsscale_3-numtokens_100-outputscale_1-duration_600 /app/workload/mooncake/toolagent_realistic_tokenized_rpsscale_3_numtokens_100


COPY workload-and-experiment_results/workload-generator/mooncake/conversation_realistic_workload_tokenized-rpsscale_3-numtokens_100-outputscale_1-duration_600 /app/workload/mooncake/conversation_realistic_tokenized_rpsscale_3_numtokens_100

COPY workload-and-experiment_results/workload-generator/mooncake/synthetic_realistic_workload_tokenized-rpsscale_3-numtokens_100-outputscale_1-duration_600 /app/workload/mooncake/synthetic_realistic_tokenized_rpsscale_3_numtokens_100

COPY workload-and-experiment_results/workload-generator/mooncake/toolagent_realistic_workload_tokenized-rpsscale_3-numtokens_100-outputscale_1-duration_600 /app/workload/mooncake/toolagent_realistic_tokenized_rpsscale_3_numtokens_100



COPY workload-and-experiment_results/workload-generator/mooncake/synthetic_realistic_workload_tokenized-rpsscale_3-numtokens_100-outputscale_1.0-duration_1800-head_2.0x-tail_4.0x-after0.7 /app/workload/mooncake/synthetic_realistic_tokenized_rpsscale_3_numtokens_100_head_2.0x_tail_4.0x_after0.7


COPY workload-and-experiment_results/workload-generator/mooncake/toolagent_realistic_workload_tokenized-rpsscale_2-numtokens_200-outputscale_1-duration_600 /app/workload/mooncake/toolagent_realistic_tokenized_rpsscale_2_numtokens_200


COPY workload-and-experiment_results/workload-generator/mooncake/synthetic_realistic_workload_tokenized-rpsscale_1-numtokens_100-outputscale_1.0-duration_1800-head_2.0x-tail_8.0x-after0.7 /app/workload/mooncake/synthetic_realistic_tokenized_rpsscale_1_numtokens_100_head_2.0x_tail_8.0x_after0.7
COPY workload-and-experiment_results/workload-generator/mooncake/synthetic_realistic_workload_tokenized-rpsscale_1.5-numtokens_100-outputscale_1.0-duration_1800-head_2.0x-tail_8.0x-after0.7 /app/workload/mooncake/synthetic_realistic_tokenized_rpsscale_1.5_numtokens_100_head_2.0x_tail_8.0x_after0.7

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
