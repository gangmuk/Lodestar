#!/bin/bash

message=$(for i in $(seq 1 2000); do echo -n "hello "; done)

llm_model="llama-3-8b-instruct"
subAlgorithm="least_latency"
# llm_model=$1
# subAlgorithm=$2
random_request_id=$((RANDOM % 1000))
routing_policy="rl-online-router"
# routing_policy="preble"

for i in {1..1}; do
    prompt="${message} ${random_request_id}"
    echo "Sending request ${i}"
    sleep 0.5
    curl -i http://115.190.203.81/v1/chat/completions \
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
                    "content": "'"${prompt}"'"
                }
            ],
            "temperature": 0.0,
            "max_tokens": 50,
            "min_tokens": 50,
            "ignore_eos": true
        }' &
done