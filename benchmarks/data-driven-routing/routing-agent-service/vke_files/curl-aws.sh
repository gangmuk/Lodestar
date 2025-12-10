#!/bin/bash

message="What kind of fun things are there in the world?"

llm_model="llama-3-8b-instruct"
subAlgorithm="latency_predictor"
# subAlgorithm="contextual_bandit"
random_request_id=$((RANDOM % 1000))
routing_policy="rl-online-router"
# routing_policy="preble"

# k get svc -n envoy-gateway envoy-aibrix-system-aibrix-eg-903790dc
# envoy_gateway_ip=172.20.198.174:80  # ClusterIP - only works from inside cluster

# kubectl port-forward svc/envoy-aibrix-system-aibrix-eg-903790dc 8080:80 -n envoy-gateway-system &
# port_forward_pid=$!
envoy_gateway_ip=localhost:8080  # Port-forwarded

curl -i -v http://${envoy_gateway_ip}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi" \
    -H "request-id: ${random_request_id}" \
    -H "routing-strategy: ${routing_policy}" \
    -H "subAlgorithm: ${subAlgorithm}" \
    -d '{
        "model": "'"${llm_model}"'",
        "messages": [
            {
                "role": "user",
                "content": "'"${message}"'"
            }
        ],
        "temperature": 0.0,
        "max_tokens": 100,
        "min_tokens": 100,
        "ignore_eos": true
    }'

# kill $port_forward_pid
