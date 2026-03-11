#!/bin/bash

build=$1

# docker buildx build --platform ${platform} --no-cache -t aibrix/gangmuk-routing-agent:${tag} .

# Check if buildx is available
if sudo docker buildx version >/dev/null 2>&1; then
    USE_BUILDX=true
    BUILDX_CMD="sudo docker buildx build"
else
    USE_BUILDX=false
    BUILDX_CMD="sudo docker build"
    echo "Warning: docker buildx not available, using legacy docker build"
    echo "Install buildx to avoid deprecation warnings: https://docs.docker.com/go/buildx/"
fi

if [ "$build" == "vke" ]; then
    tag=latest-vke-gangmuk-recovery
    if [ "$USE_BUILDX" = true ]; then
        # sudo docker buildx build --platform linux/amd64 -f Dockerfile -t aibrix/gangmuk-routing-agent:${tag} --load .
        sudo docker buildx build --platform linux/amd64 -f Dockerfile -t aibrix/gangmuk-routing-agent:${tag} --load .
    else
        # sudo docker build --platform linux/amd64 -f Dockerfile -t aibrix/gangmuk-routing-agent:${tag} .
        echo "error: docker buildx not available, using legacy docker build"
        echo "Install buildx to avoid deprecation warnings: https://docs.docker.com/go/buildx/"
        echo "exiting..."
        exit 1
    fi

   sudo docker tag aibrix/gangmuk-routing-agent:${tag} aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gangmuk-routing-agent:${tag}
   sudo docker push aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gangmuk-routing-agent:${tag}
   kubectl set image deployment/routing-agent-service routing-agent=aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gangmuk-routing-agent:${tag}
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
    if [ "$USE_BUILDX" = true ]; then
        sudo docker buildx build -f Dockerfile -t aibrix/gangmuk-routing-agent:${tag} --load .
    else
        sudo docker build -f Dockerfile -t aibrix/gangmuk-routing-agent:${tag} .
    fi
    sudo docker tag aibrix/gangmuk-routing-agent:${tag} gangmuk/gangmuk-routing-agent:${tag}
    sudo docker push gangmuk/gangmuk-routing-agent:${tag} # push to dockerhub
    kubectl set image deployment/routing-agent-service routing-agent=gangmuk/gangmuk-routing-agent:${tag}
    kubectl patch deployment routing-agent-service -p '{"spec":{"template":{"spec":{"containers":[{"name":"routing-agent","imagePullPolicy":"Always"}]}}}}'
    kubectl set env deployment/routing-agent-service POD_LABEL_SELECTOR=${POD_LABEL_SELECTOR}
    kubectl rollout restart deploy routing-agent-service
fi
