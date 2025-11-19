#!/bin/bash

message="What kind of fun things are there in the world?"

# llm_model="llama-3-8b-instruct"
# subAlgorithm="latency_predictor"
llm_model=$1
subAlgorithm=$2
random_request_id=$((RANDOM % 1000))
routing_policy="rl-online-router"
# routing_policy="preble"

curl -i -v http://115.190.180.7:80/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "request-id: ${random_request_id}" \
    -H "Authorization: Bearer sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi" \
    -H "routing-strategy: ${routing_policy}" \
    -d '{
        "model": "'"${llm_model}"'",
        "subAlgorithm": "'"${subAlgorithm}"'",
        "messages": [
            {
                "role": "user",
                "content": "'"${message}"'"
            }
        ],
        "temperature": 0.0,
        "max_tokens": 1,
        "min_tokens": 1,
        "ignore_eos": true
    }'