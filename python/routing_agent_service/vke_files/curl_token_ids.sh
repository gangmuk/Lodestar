#!/bin/bash

message="What kind of fun things are there in the world?"

llm_model="llama-3-8b-instruct"
subAlgorithm="latency_predictor"
random_request_id=$((RANDOM % 1000))
routing_policy="rl-online-router"
# routing_policy="preble"

# Convert message to token IDs using Python
# Note: This uses GPT-2 tokenizer as an example. You may need to adjust based on your model.
token_ids=$(python3 -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
token_ids = tokenizer.encode('${message}', add_special_tokens=False)
print(' '.join(map(str, token_ids)))
")

echo "Message: ${message}"
echo "Token IDs: ${token_ids}"
echo ""

# Convert space-separated token IDs to JSON array
token_ids_json=$(echo "${token_ids}" | python3 -c "
import sys
token_ids = sys.stdin.read().strip().split()
print('[' + ','.join(token_ids) + ']')
")

echo "Sending request with token IDs..."
echo ""

curl -i -v http://115.190.203.81:80/v1/completions \
    -H "Content-Type: application/json" \
    -H "request-id: ${random_request_id}" \
    -H "Authorization: Bearer sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi" \
    -H "routing-strategy: ${routing_policy}" \
    -d '{
        "model": "'"${llm_model}"'",
        "subAlgorithm": "'"${subAlgorithm}"'",
        "prompt": "",
        "prompt_token_ids": '"${token_ids_json}"',
        "temperature": 0.0,
        "max_tokens": 50,
        "min_tokens": 50,
        "ignore_eos": true
    }'
