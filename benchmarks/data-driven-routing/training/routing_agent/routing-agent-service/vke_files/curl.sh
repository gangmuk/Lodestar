#!/bin/bash


curl -i -v http://localhost:8888/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi" \
    -H "routing-strategy: rl-online-router" \
    -d '{"model": "llama-3-8b-instruct", "messages": [{"role": "user", "content": "Where is Beijing"}], "temperature": 0.0, "max_tokens": 100}'



# curl -i -v http://localhost:8888/v1/completions \
#     -H "Content-Type: application/json" \
#     -H "Authorization: Bearer sk-kFJ12nKsFVfVmGpj3QzX65s4RbN2xJqWzPYCjYu7wT3BlbLi" \
#     -H "routing-strategy: rl-online-router" \
#     -d '{"model": "llama-3-8b-instruct", "prompt": "Where is Beijing", "temperature": 0.0, "max_tokens": 100}'

#     # -H "routing-strategy: prefix-cache-and-load" \
#     # -H "routing-strategy: random" \
