#!/bin/bash

build=$1

# docker buildx build --platform ${platform} --no-cache -t aibrix/gangmuk-routing-agent:${tag} .

if [ "$build" == "vke" ]; then
    tag=latest-vke-gangmuk
    sudo docker build --platform linux/amd64 -f Dockerfile -t aibrix/gangmuk-routing-agent:${tag} .
    sudo docker tag aibrix/gangmuk-routing-agent:${tag} aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gangmuk-routing-agent:${tag}
    sudo docker push aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gangmuk-routing-agent:${tag}
    kubectl set image deployment/routing-agent-service routing-agent=aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gangmuk-routing-agent:${tag}
    # kubectl set env deployment/routing-agent-service POD_LABEL_SELECTOR="model.aibrix.ai/name=llama-3-8b-instruct"
else
    if [ "$build" == "local-linux" ]; then
        tag=latest-linux
    elif [ "$build" == "local-mac" ]; then
        tag=latest-mac
    else
        echo "Unknown build type. Defaulting to latest-mac."
        exit
    fi
    POD_LABEL_SELECTOR="model.aibrix.ai/name=llama2-7b"
    # you don't need to use platform and buildx since sudo docker will build based on the current machine type automatically.
    sudo docker build -f Dockerfile -t aibrix/gangmuk-routing-agent:${tag} .
    sudo docker tag aibrix/gangmuk-routing-agent:${tag} gangmuk/gangmuk-routing-agent:${tag}
    sudo docker push gangmuk/gangmuk-routing-agent:${tag} # push to dockerhub
    kubectl set image deployment/routing-agent-service routing-agent=gangmuk/gangmuk-routing-agent:${tag}
    kubectl patch deployment routing-agent-service -p '{"spec":{"template":{"spec":{"containers":[{"name":"routing-agent","imagePullPolicy":"Always"}]}}}}'
    kubectl set env deployment/routing-agent-service POD_LABEL_SELECTOR=${POD_LABEL_SELECTOR}
    kubectl rollout restart deploy routing-agent-service
fi