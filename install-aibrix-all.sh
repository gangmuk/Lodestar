#!/bin/bash

kubectl create -f https://github.com/vllm-project/aibrix/releases/download/v0.4.1/aibrix-dependency-v0.4.1.yaml

kubectl create -f https://github.com/vllm-project/aibrix/releases/download/v0.4.1/aibrix-core-v0.4.1.yaml

## Remove taint in your node if you want to schedule pods in your control-plane cluster. e.g., when you have only one node which is control-plane in your cluster. 

# kubectl taint nodes node0.gangmuk-266786.mlproxy-pg0.clemson.cloudlab.us node-role.kubernetes.io/control-plane:NoSchedule-

node_name=node1.gangmuk-272174.mlproxy-pg0.clemson.cloudlab.us
kubectl label node $node_name machine.cluster.vke.volcengine.com/gpu-name=NVIDIA-L20

## RL routing agent service
kubectl apply -f benchmarks/data-driven-routing/routing-agent-service/k8s/routing-agent/routing-agent-service.yaml

# kubectl apply -f ~/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/k8s/routing-agent/deployment-routing-agent-service.yaml

kubectl apply -f benchmarks/data-driven-routing/routing-agent-service/k8s/routing-agent/svc-routing-agent-service.yaml

kubectl apply -f ~/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/k8s/routing-agent/sa-clusterrole-rolebinding.yaml

## Mock application

kubectl apply -f benchmarks/data-driven-routing/routing-agent-service/k8s/mock-app-yamls/mock-deployment-svc-sa.yaml 

bash development/app/build-and-push.sh local

## Building aibrix-gateway-plugin
##  ~/projects/aibrix-gangmuk/build-gateway.sh <build>
## Allocate more than 8 cores for limits field

## Building routing agent service
##  ~/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/build-and-push.sh <build>

## Building mock-app
##  ~/projects/aibrix-gangmuk/development/app/build-and-push.sh <build>