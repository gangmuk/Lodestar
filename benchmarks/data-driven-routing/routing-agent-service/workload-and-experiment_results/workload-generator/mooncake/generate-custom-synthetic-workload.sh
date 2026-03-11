#!/bin/bash

# Custom stretched synthetic workload generator
# Stretches both head (low-sharing) and tail (high-sharing) independently.
#
# --stretch-after-pct 0.7: head/tail split at 70% of trace
# --head-stretch-factor 2.0: head is 2x longer
# --tail-stretch-factor 3.0: tail is 3x longer
#
# Use both factors=1.0 to get identical output to realistic_workload_generator.py

STRETCH_AFTER_PCT=${STRETCH_AFTER_PCT:-0.7}
HEAD_STRETCH=${HEAD_STRETCH:-2.0}
TAIL_STRETCH=${TAIL_STRETCH:-8.0}
RPS_SCALE=${RPS_SCALE:-1}
NUM_TOKENS=${NUM_TOKENS:-100}
OUTPUT_SCALE=${OUTPUT_SCALE:-1.0}
DURATION=${DURATION:-1800}
TRACE=${TRACE:-Mooncake_synthetic_trace.jsonl}

OUTPUT_DIR="synthetic_realistic_workload_tokenized-rpsscale_${RPS_SCALE}-numtokens_${NUM_TOKENS}-outputscale_${OUTPUT_SCALE}-duration_${DURATION}-head_${HEAD_STRETCH}x-tail_${TAIL_STRETCH}x-after${STRETCH_AFTER_PCT}"

python custom_synthetic_workload_generator.py \
    --rps-scale "${RPS_SCALE}" \
    --num-tokens-per-hash-id "${NUM_TOKENS}" \
    --output-length-scale "${OUTPUT_SCALE}" \
    --mooncake-trace "${TRACE}" \
    --output-dir "${OUTPUT_DIR}" \
    --duration "${DURATION}" \
    --vocab-csv vocab.csv \
    --stretch-after-pct "${STRETCH_AFTER_PCT}" \
    --head-stretch-factor "${HEAD_STRETCH}" \
    --tail-stretch-factor "${TAIL_STRETCH}"
