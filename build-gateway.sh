#!/bin/bash

build=$1
# tag=gangmuk-20250415-gatewaylog-00572df

if [ -z "$build" ]; then
    echo "build argument is empty"
    echo "Usage: ./build-gateway.sh <local|remote>"
    echo "exiting..."
    exit 1
fi

if [ "$build" == "remote" ]; then
    ## Remote push
    # build
    # make docker-build-gateway-plugins-amd64 # Use it when you build it on a mac but for intel server (vke).
    tag=latest-linux-gangmuk

    docker buildx build --platform linux/amd64 -t aibrix/gateway-plugins:nightly -f build/container/Dockerfile.gateway .

    # tag
    docker tag aibrix/gateway-plugins:nightly aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gateway-plugins:${tag}

    # push
    docker push aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gateway-plugins:${tag}
else
    tag=latest-mac
    ## for local docker registry only
    make docker-build-gateway-plugins
    docker tag aibrix/gateway-plugins:nightly gangmuk/gateway-plugins:${tag}
    docker push gangmuk/gateway-plugins:${tag}

    ## load image to kind cluster
    # kind load docker-image aibrix/gateway-plugins:${tag}

    ## use dockerhub...
    kubectl set image deployment/aibrix-gateway-plugins gateway-plugin=gangmuk/gateway-plugins:${tag} -n aibrix-system

    kubectl patch deployment aibrix-gateway-plugins -n aibrix-system -p '{"spec":{"template":{"spec":{"containers":[{"name":"gateway-plugin","imagePullPolicy":"Always"}]}}}}'

    kubectl rollout restart deployment/aibrix-gateway-plugins -n aibrix-system
fi

kubectl rollout restart deploy aibrix-gateway-plugins -n aibrix-system