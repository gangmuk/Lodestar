#!/bin/bash

## It requires port forwarding
## kubectl -n envoy-gateway-system port-forward service/envoy-aibrix-system-aibrix-eg-903790dc 8888:80

num_requests=$1
if [ -z "$num_requests" ]; then
    num_requests=1
fi

# ROUTING_STRATEGY="random"
ROUTING_STRATEGY="rl-online-router"
subAlgorithm="none"
# subAlgorithm="prefix-cache-1"
# subAlgorithm="prefix-cache-2"

# randomly generate request id from 0 to 1000
for i in $(seq 1 $num_requests); do
    request_id=$((RANDOM % 1000))
    echo "Request ${request_id}"
    curl -i -v http://localhost:8888/v1/chat/completions \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi" \
        -H "routing-strategy: ${ROUTING_STRATEGY}" \
        -H "request-id: ${request_id}" \
        -d '{"model": "llama2-7b", "subAlgorithm": "'"${subAlgorithm}"'", "messages": [{"role": "user", "content": "I like apple, orange juice, golden kiwi, cilantro, pineapple, watermelon, blueberry, strawberry, peach, graph!!! All are amazing!!! What fruits would I like more? Please give me some recommendations."}], "temperature": 0.0, "max_tokens": 500}'
done