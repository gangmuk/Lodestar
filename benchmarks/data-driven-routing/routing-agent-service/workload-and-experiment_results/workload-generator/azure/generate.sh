#!/bin/bash

    # --azure-csv AzureLLMInferenceTrace_code.csv \
python azure_workload_generator.py \
    --azure-csv AzureLLMInferenceTrace_conv.csv \
    --target-avg-rps 10 \
    --num-requests 5000 \
    --max-rps-multiplier 3.0 \
    --rps-pattern poisson \
    --generate-plots \
    --seed 777 \
    --num-requests-per-prefix 10 \
    --num-requests-per-prefix-std 3 \
    --shared-proportion 0.3 \
    --shared-proportion-std 0.1 \
    --access-pattern sequential \
    # --access-pattern random \
    # --access-pattern normal \
    # --normal-mean-ratio 0.5 \
    # --normal-std-ratio 0.2 \