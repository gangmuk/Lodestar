#!/bin/bash

ROUTING_STRATEGY="rl-online-router"
subAlgorithm="none"

curl -i -v http://localhost:8888/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi" \
    -H "routing-strategy: ${ROUTING_STRATEGY}" \
    -d '{"model": "llama-3-8b-instruct", "subAlgorithm": "'"${subAlgorithm}"'", "messages": [{"role": "user", "content": "Where is Beijing"}], "temperature": 0.0, "max_tokens": 100}'


# curl -i -v http://localhost:8888/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -H "Authorization: Bearer sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi" \
#     -H "routing-strategy: random" \
#     -d '{"model": "llama2-7b", "messages": [{"role": "user", "content": "Where is Beijing"}], "temperature": 0.0, "max_tokens": 100}'
