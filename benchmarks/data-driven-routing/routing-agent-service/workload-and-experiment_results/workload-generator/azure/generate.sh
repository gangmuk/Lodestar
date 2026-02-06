#!/bin/bash

# code workload
python azure_workload_generator.py \
    --azure-csv AzureLLMInferenceTrace_code.csv \
    --shared-proportion 0.3 \
    --shared-proportion-std 0.1 \
    --num-requests-per-prefix 10 \
    --shared-proportion-std 0.2 \
    --num-requests-per-prefix-std 3 \
    --target-avg-rps 10 \
    --num-requests 5000 \
    --max-rps-multiplier 3.0 \
    --rps-pattern poisson \
    --generate-plots \
    --seed 777 \
    --access-pattern sequential

# single turn conversation workload
python azure_workload_generator.py \
    --azure-csv AzureLLMInferenceTrace_conv.csv \
    --shared-proportion 0.5 \
    --shared-proportion-std 0.2 \
    --num-requests-per-prefix 10 \
    --shared-proportion-std 0.2 \
    --num-requests-per-prefix-std 3 \
    --target-avg-rps 10 \
    --num-requests 5000 \
    --max-rps-multiplier 3.0 \
    --rps-pattern poisson \
    --generate-plots \
    --seed 777 \
    --access-pattern sequential

# multi turn conversation workload
python azure_workload_generator.py \
    --azure-csv AzureLLMInferenceTrace_conv.csv \
    --shared-proportion 0.1 \
    --shared-proportion-std 0.1 \
    --num-requests-per-prefix 2 \
    --shared-proportion-std 0.2 \
    --num-requests-per-prefix-std 3 \
    --target-avg-rps 10 \
    --num-requests 5000 \
    --max-rps-multiplier 3.0 \
    --rps-pattern poisson \
    --generate-plots \
    --seed 777 \
    --access-pattern sequential