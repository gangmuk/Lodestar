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

########################################################
#### NVIDIA-L20, llama3.1 8B, no bitsandbytes ##########
########################################################
COPY training_data/L20-7/merged-data/all-with-mixed/final_model-latency_predictor_ttft-20251101_205621 NVIDIA-L20/Aggregated/final_model/latency_predictor
COPY training_data/L20-7/merged-data/all-with-mixed/data_replaced-processed.csv NVIDIA-L20/Aggregated/offline_training_data.csv

########################################################
#### NVIDIA-A30, llama3.1 8B, no bitsandbytes ##########
########################################################
#### NVIDIA-A30
# COPY training_data/A30-8/final_model-latency_predictor_ttft-20251028_183743 final_model/NVIDIA-A30
COPY training_data/A30-8/final_model-latency_predictor_ttft-20251101_205321 NVIDIA-A30/Aggregated/final_model/latency_predictor
COPY training_data/A30-8/data-processed.csv NVIDIA-A30/Aggregated/offline_training_data.csv


###############################################################################
#### hetero: NVIDIA-L20*7 and NVIDIA-A30*8, , llama3.1 8B, no bitsandbytes ####
###############################################################################
COPY training_data/hetero/final_model-latency_predictor_ttft-20251101_213101-epoch60 hetero/Aggregated/final_model/latency_predictor
COPY training_data/hetero/data-processed.csv hetero/Aggregated/offline_training_data.csv

##################################################
#### NVIDIA-A10 *7, llama3.1 8B, bitsandbytes ####
##################################################
COPY workload-and-experiment_results/NVIDIA-A10/MixedSharingRatio10_30_50_70%/rps5/final_model-latency_predictor_ttft-20251120_173020 NVIDIA-A10/Aggregated/final_model/latency_predictorMixedSharingRatio10_30_50_70%
COPY workload-and-experiment_results/NVIDIA-A10/MixedSharingRatio10_30_50_70%/rps5/data-processed.csv NVIDIA-A10/Aggregated/MixedSharingRatio10_30_50_70%/offline_training_data.csv

COPY workload-and-experiment_results/NVIDIA-A10/SharingRatio71%/rps6.5/final_model-latency_predictor_ttft-20251120_172727 NVIDIA-A10/Aggregated/final_model/latency_predictorSharingRatio71%
COPY workload-and-experiment_results/NVIDIA-A10/SharingRatio71%/rps6.5/data-processed.csv NVIDIA-A10/Aggregated/SharingRatio71%/offline_training_data.csv



## NVIDIA-A10, offline data, PrefillOnly
COPY workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0/data-processed.csv NVIDIA-A10/PrefillOnly/offline_training_data.csv

## latency predictor
COPY workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0/final_model-latency_predictor_ttft-1-without-output NVIDIA-A10/PrefillOnly/final_model/latency_predictor
COPY workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0/final_model-latency_predictor_ttft-1-with-output NVIDIA-A10/PrefillOnly/final_model/latency_predictor/with-output_tokens

## contextual bandit

## source dir
# final_model-contextual_bandit_context_aware_clustermodel
# final_model-contextual_bandit_inverse_latency_clustermodel
# final_model-contextual_bandit_latency_optimized_clustermodel
# final_model-contextual_bandit_linear_simple_extended_clustermodel
# final_model-contextual_bandit_negative_linear_clustermodel
# final_model-contextual_bandit_negative_reciprocal_clustermodel
# final_model-contextual_bandit_negative_squared_clustermodel
# final_model-contextual_bandit_quantile_based_clustermodel
# final_model-contextual_bandit_simple_latency_minimization_clustermodel

# final_model-contextual_bandit_negative_linear_perpodmodel
# final_model-contextual_bandit_negative_squared_perpodmodel
# final_model-contextual_bandit_quantile_based_perpodmodel


COPY workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0/final_model-contextual_bandit_linear_simple_extended_clustermodel NVIDIA-A10/PrefillOnly/final_model/contextual_bandit_linear_simple_extended_clustermodel
COPY workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0/final_model-contextual_bandit_latency_optimized_clustermodel NVIDIA-A10/PrefillOnly/final_model/contextual_bandit_latency_optimized_clustermodel
COPY workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0/final_model-contextual_bandit_simple_latency_minimization_clustermodel NVIDIA-A10/PrefillOnly/final_model/contextual_bandit_simple_latency_minimization_clustermodel
COPY workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0/final_model-contextual_bandit_quantile_based_clustermodel NVIDIA-A10/PrefillOnly/final_model/contextual_bandit_quantile_based_clustermodel
COPY workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0/final_model-contextual_bandit_negative_linear_clustermodel NVIDIA-A10/PrefillOnly/final_model/contextual_bandit_negative_linear_clustermodel

COPY workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0/final_model-contextual_bandit_negative_linear_perpodmodel NVIDIA-A10/PrefillOnly/final_model/contextual_bandit_negative_linear_perpodmodel
COPY workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0/final_model-contextual_bandit_negative_squared_perpodmodel NVIDIA-A10/PrefillOnly/final_model/contextual_bandit_negative_squared_perpodmodel
COPY workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0/final_model-contextual_bandit_quantile_based_perpodmodel NVIDIA-A10/PrefillOnly/final_model/contextual_bandit_quantile_based_perpodmodel


COPY workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0/final_model-contextual_bandit_quantile_based_perpodmodel_advanced NVIDIA-A10/PrefillOnly/final_model/contextual_bandit_quantile_based_perpodmodel_advanced

COPY workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0/final_model-contextual_bandit_quantile_based_perpodmodel_policygradient NVIDIA-A10/PrefillOnly/final_model/contextual_bandit_quantile_based_perpodmodel_policygradient



################################################################################################################

COPY Dockerfile /app/Dockerfile
RUN chmod +x /app/start.sh
ENV PORT=8080
ENV NVIDIA_VISIBLE_DEVICES=all
ENV LOG_LEVEL=INFO
EXPOSE 8080

# CMD ["python", "routing_agent_service.py"]
CMD ["/app/start.sh"]