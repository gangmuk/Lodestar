#!/bin/bash

kubectl port-forward svc/envoy-aibrix-system-aibrix-eg-903790dc 8080:80 -n envoy-gateway-system