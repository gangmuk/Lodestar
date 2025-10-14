#!/bin/bash

build=$1
# tag=gangmuk-20250415-gatewaylog-00572df

if [ -z "$build" ]; then
    echo "build argument is empty"
    echo "Usage: ./build-gateway.sh <vke|local-mac|local-linux>"
    echo "exiting..."
    exit 1
fi

# if build is none of vke, local-mac, local linux, exit


if [ "$build" == "vke" ]; then
    tag=latest-vke-gangmuk
    sudo docker buildx build --platform linux/amd64 -t aibrix/gateway-plugins:nightly -f build/container/Dockerfile.gateway .
    sudo docker tag aibrix/gateway-plugins:nightly aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gateway-plugins:${tag}
    sudo docker push aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gateway-plugins:${tag}
else
    if [ "$build" == "local-linux" ]; then
        tag=latest-linux
    elif [ "$build" == "local-mac" ]; then
        tag=latest-mac
    fi
    make docker-build-gateway-plugins # it will build based on the current machine's type
    sudo docker tag aibrix/gateway-plugins:nightly gangmuk/gateway-plugins:${tag}
    sudo docker push gangmuk/gateway-plugins:${tag} # dockerhub
    kubectl set image deployment/aibrix-gateway-plugins gateway-plugin=gangmuk/gateway-plugins:${tag} -n aibrix-system
    kubectl patch deployment aibrix-gateway-plugins -n aibrix-system -p '{"spec":{"template":{"spec":{"containers":[{"name":"gateway-plugin","imagePullPolicy":"Always"}]}}}}'
    kubectl rollout restart deployment/aibrix-gateway-plugins -n aibrix-system
fi
