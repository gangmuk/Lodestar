#!/bin/bash

build=$1

if [ -z "$build" ]; then
  echo "Please provide a build type:"
  echo "  vke          - Build for VKE cluster"
  echo "  dockerhub    - Build for DockerHub"
  echo "  local-linux  - Build for local Linux K8s"
  echo "  local-mac    - Build for local Mac K8s"
  exit 1
fi

# docker buildx build --platform ${platform} --no-cache -t aibrix/gangmuk-client:${tag} .

if [ "$build" == "vke" ]; then
    tag=latest-vke-gangmuk
    echo "Building client image for VKE cluster..."
    # Build from parent directory to include workload files
    cd .. && sudo docker buildx build --platform linux/amd64 -f vke_files/Dockerfile -t aibrix/gangmuk-client:${tag} .
    sudo docker tag aibrix/gangmuk-client:${tag} aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gangmuk-client:${tag}
    sudo docker push aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gangmuk-client:${tag}
    kubectl set image deployment/client-service client=aibrix-container-registry-cn-beijing.cr.volces.com/aibrix/gangmuk-client:${tag}
else
    if [ "$build" == "local-linux" ]; then
        tag=latest-linux
    elif [ "$build" == "local-mac" ]; then
        tag=latest-mac
    else
        echo "Unknown build type. Please use 'vke', 'local-linux', or 'local-mac'."
        exit 1
    fi
    
    echo "Building client image for local K8s (${build})..."
    # Build from parent directory to include workload files
    # you don't need to use platform and buildx since sudo docker will build based on the current machine type automatically.
    cd .. && sudo docker build -f vke_files/Dockerfile -t aibrix/gangmuk-client:${tag} .
    sudo docker tag aibrix/gangmuk-client:${tag} gangmuk/gangmuk-client:${tag}
    sudo docker push gangmuk/gangmuk-client:${tag} # push to dockerhub
    echo "✓ Image pushed to DockerHub"
    
    # Update deployment if it exists
    if kubectl get deployment client-service -n default &> /dev/null; then
        echo "Updating client-service deployment..."
        kubectl set image deployment/client-service client=gangmuk/gangmuk-client:${tag}
        kubectl patch deployment client-service -p '{"spec":{"template":{"spec":{"containers":[{"name":"client","imagePullPolicy":"Always"}]}}}}'
        echo "✓ Deployment updated"
    else
        echo "⚠️  Deployment 'client-service' not found. You'll need to create it manually."
    fi
fi

# kubectl rollout restart deploy client-service