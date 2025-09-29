#!/bin/bash

build=$1
if [ -z "${build}" ]; then
  echo "Please provide a build (e.g., 'remote' or 'local')."
  exit 1
fi

docker build -t aibrix/vllm-mock:latest -f Dockerfile .

if [ "${build}" == "vke" ]; then
    docker tag aibrix/vllm-mock:latest aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/vllm-mock:latest
    docker push aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/vllm-mock:latest
elif [ "${build}" == "local" ]; then
    docker tag aibrix/vllm-mock:latest gangmuk/vllm-mock:latest
    docker push gangmuk/vllm-mock:latest
    kubectl set image deployment/mock-app llm-engine=gangmuk/vllm-mock:latest
    kubectl patch deployment mock-app -p '{"spec":{"template":{"spec":{"containers":[{"name":"llm-engine","imagePullPolicy":"Always"}]}}}}'
    kubectl rollout restart deploy mock-app
fi