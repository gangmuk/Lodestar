#!/bin/bash

# message=$(for i in $(seq 1 100); do echo -n "hello "; done)
message="What fun things can I do in Champaign, IL"

routing_policy="rl-online-router"
subAlgorithm="random"
# subAlgorithm="least_latency"
# subAlgorithm="contextual_bandit_perpodmodel_checkpoint_negative_linear"

# llm_model="llama-3-8b-instruct-v100"
llm_model="llama-3-8b-instruct"
# llm_model="qwen25-1-5b-instruct"
random_request_id=$((RANDOM % 1000))
# routing_policy="preble"

envoy_gateway_external_ip=$(kubectl get svc -n envoy-gateway-system envoy-aibrix-system-aibrix-eg-903790dc -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "envoy_gateway_external_ip: ${envoy_gateway_external_ip}"

for i in {1..10}; do
    prompt="${message} ${random_request_id}"
    echo "Sending request ${i}"
    sleep 0.5
    curl -i http://${envoy_gateway_external_ip}/v1/chat/completions \
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
        }'
    sleep 1
done