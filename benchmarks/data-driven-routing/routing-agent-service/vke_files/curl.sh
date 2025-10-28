#!/bin/bash

ROUTING_STRATEGY="rl-online-router"
subAlgorithm="none"
request_id=$((RANDOM % 1000))
llm_model="llama-3-8b-instruct"

# curl -i -v http://localhost:8888/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -H "request-id: ${request_id}" \
#     -H "Authorization: Bearer sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi" \
#     -H "routing-strategy: ${ROUTING_STRATEGY}" \
#     -d '{"model": "llama3-1-8b", "subAlgorithm": "'"${subAlgorithm}"'", "messages": [{"role": "user", "content": "Where is Champaign? I am planning to visit the town. Recommend me some interesting things to do."}], "temperature": 0.0, "max_tokens": 10000, "min_tokens": 10000, "ignore_eos": true}'

random_request_id=$((RANDOM % 1000))
routing_policy="preble"

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
                "content": "Where is Champaign? I am planning to visit the town. Recommend me some interesting things to do."
            }
        ],
        "temperature": 0.0,
        "max_tokens": 20,
        "min_tokens": 20,
        "ignore_eos": true
    }'