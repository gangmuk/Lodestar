# Check pods for port 8080 in specific namespaces only
for namespace in default aibrix-system dynamo; do
  echo "=== Checking namespace: $namespace ==="
  kubectl get pods -n $namespace -o wide | grep Running | while read pod rest; do
    echo "Checking $namespace/$pod..."
    kubectl exec -n $namespace $pod -- netstat -tlnp 2>/dev/null | grep :8080 && echo "  ^^^ Found port 8080 in $namespace/$pod"
  done
  echo
done