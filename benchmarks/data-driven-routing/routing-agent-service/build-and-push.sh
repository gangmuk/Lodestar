#!/bin/bash

build=$1

if [ -z "$build" ]; then
  echo "Please provide a build (e.g., 'remote' or 'local')."
  exit 1
fi

# docker buildx build --platform ${platform} --no-cache -t aibrix/gangmuk-routing-agent:${tag} .

if [ "$build" == "remote" ]; then
    dockerfile=Dockerfile
    tag=latest-linux
    docker buildx build --platform linux/amd64 -f ${dockerfile} -t aibrix/gangmuk-routing-agent:${tag} .
    docker tag aibrix/gangmuk-routing-agent:${tag} aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gangmuk-routing-agent:${tag}
    docker push aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gangmuk-routing-agent:${tag}
    # kubectl set env deployment/routing-agent-service POD_LABEL_SELECTOR="model.aibrix.ai/name=llama-3-8b-instruct"
elif [ "$build" == "local" ]; then
    dockerfile=Dockerfile
    tag=latest-mac
    docker build --platform linux/arm64 -f ${dockerfile} -t aibrix/gangmuk-routing-agent:${tag} .

    ## load image to kind cluster
    # kind load docker-image aibrix/gangmuk-routing-agent:${tag}

    ## create local docker registry and connect it to kind
    # docker tag aibrix/gangmuk-routing-agent:${tag} localhost:5001/aibrix/gangmuk-routing-agent:${tag}
    # docker push localhost:5001/aibrix/gangmuk-routing-agent:${tag}

    ## use dockerhub...
    docker tag aibrix/gangmuk-routing-agent:${tag} gangmuk/gangmuk-routing-agent:${tag}
    docker push gangmuk/gangmuk-routing-agent:${tag}

    kubectl set image deployment/routing-agent-service routing-agent=gangmuk/gangmuk-routing-agent:${tag}

    kubectl patch deployment routing-agent-service -p '{"spec":{"template":{"spec":{"containers":[{"name":"routing-agent","imagePullPolicy":"Always"}]}}}}'

    kubectl set env deployment/routing-agent-service POD_LABEL_SELECTOR="model.aibrix.ai/name=llama2-7b"

    kubectl rollout restart deploy routing-agent-service
else
    echo "Invalid build. Use 'remote' or 'local'."
    exit 1
fi