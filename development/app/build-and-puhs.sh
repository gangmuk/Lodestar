#!/bin/bash

target=$1

if [ -z "$target" ]; then
  echo "Please provide a target (e.g., 'remote' or 'local')."
  exit 1
fi

# if [ "$target" == "remote" ]; then
#     platform=linux/amd64
# elif [ "$target" == "local" ]; then
#     platform=linux/arm64
# else
#     echo "Invalid target. Use 'remote' or 'local'."
#     exit 1
# fi

tag=latest

# docker buildx build --platform ${platform} -t aibrix/vllm-mock:${tag} -f Dockerfile .
docker build -t aibrix/vllm-mock:${tag} -f Dockerfile .

if [ "$target" == "remote" ]; then
    docker tag aibrix/vllm-mock:${tag} aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/vllm-mock:${tag}
    docker push aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/vllm-mock:${tag}
elif [ "$target" == "local" ]; then

    docker tag aibrix/vllm-mock:${tag} gangmuk/vllm-mock:${tag}

    ## load image to kind cluster
    # kind load docker-image gangmuk/vllm-mock:${tag}

    ## use dockerhub
    docker push gangmuk/vllm-mock:${tag}

    kubectl set image deployment/mock-app llm-engine=gangmuk/vllm-mock:${tag}

    kubectl patch deployment mock-app -p '{"spec":{"template":{"spec":{"containers":[{"name":"llm-engine","imagePullPolicy":"Always"}]}}}}'

    kubectl rollout restart deploy mock-app
fi