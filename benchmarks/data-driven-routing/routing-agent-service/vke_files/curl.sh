#!/bin/bash

ROUTING_STRATEGY="rl-online-router"
subAlgorithm="none"
request_id=$((RANDOM % 1000))

curl -i -v http://localhost:8888/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "request-id: ${request_id}" \
    -H "Authorization: Bearer sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi" \
    -H "routing-strategy: ${ROUTING_STRATEGY}" \
    -d '{"model": "llama3-1-8b", "subAlgorithm": "'"${subAlgorithm}"'", "messages": [{"role": "user", "content": "Where is Champaign? I am planning to visit the town. Recommend me some interesting things to do."}], "temperature": 0.0, "max_tokens": 10000}'


# curl -i -v http://115.190.180.7:80/v1/chat/completions -H "Content-Type: application/json" -H "request-id: 0" -H "Authorization: Bearer sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi" -H "routing-strategy: rl-online-router" -d '{"model": "llama3-1-8b", "subAlgorithm": "none", "messages": [{"role": "user", "content": "Where is Champaign? I am planning to visit the town. Recommend me some interesting things to do."}], "temperature": 0.0, "max_tokens": 10000}'